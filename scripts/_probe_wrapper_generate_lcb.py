"""LCB-realistic A/B: wrapper.generate(T=1, v6E) vs base.generate() byte-equality.

Sequential design (no concurrent base + wrapper on GPU):
  Phase A: load base alone, generate all 30 LCB completions, save to disk, free.
  Phase B: load a FRESH base, build the wrapper around it, generate all 30,
           save to disk, free.
  Phase C: load both saved completion lists, diff per-problem, report first_div.

This pattern matches what humaneval_smoke does (one model on GPU at a time),
which is the only configuration we know fits in 24 GB at bs=4 for DS-V2-Lite
NF4. The earlier "concurrent" probe OOM'd because base + wrapper-extras
together exceeded the 23.6 GB ceiling once a 1.6 GB fp32 attn-scores tensor
got allocated during base.generate().

Override the attention impl with MYTHIC_ATTN=flash_attention_2 once flash-attn
is installed in the env (eliminates the eager O(N²) scores tensor — frees
~3-5 GB of peak).
"""
from __future__ import annotations
import gc
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mythic_rdt.configuration import MythicRDTDeepseekV2Config
from mythic_rdt.modeling import MythicRDTDeepseekV2ForCausalLM
from mythic_rdt.training import inject_depth_lora
from mythic_rdt.training.trainer import TRAINABLE_STATE_FN, _load_trainable_state

from humaneval_smoke import load_lcb, build_lcb_chat_prompts

BASE = "base/DeepSeek-Coder-V2-Lite-Instruct"
CKPT = "checkpoints/phase1_v6a_dual_t/checkpoint-200"
N_NEW = 384
N_PROBLEMS = 30
BATCH_SIZE = 4

ATTN_IMPL = os.environ.get("MYTHIC_ATTN", "eager")
DEVICE = "cuda"
DTYPE = torch.bfloat16
QUANT = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=DTYPE)

TOKENS_BASE = Path("/tmp/probe_lcb_base_tokens.pt")
TOKENS_WRAP = Path("/tmp/probe_lcb_wrap_tokens.pt")


def _free_gpu(name: str) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"[probe] freed {name}: GPU free={free/1e9:.2f}G / {total/1e9:.2f}G")


def gen_one(model, tokenizer, prompts, **gen_kwargs) -> list[list[int]]:
    """Greedy generate for prompts in batches of BATCH_SIZE."""
    completions: list[list[int]] = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=2048,
                        add_special_tokens=False)
        ids = enc.input_ids.to(DEVICE)
        attn = enc.attention_mask.to(DEVICE)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=attn,
                                 max_new_tokens=N_NEW, do_sample=False,
                                 pad_token_id=pad_id, use_cache=True,
                                 **gen_kwargs)
        for r in range(len(batch)):
            completions.append(out[r, ids.shape[1]:].tolist())
        del enc, ids, attn, out
    return completions


def phase_a_base(prompts, tokenizer) -> None:
    print(f"\n[probe-A] === BASE ALONE === (attn={ATTN_IMPL})")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, trust_remote_code=True, quantization_config=QUANT,
        torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation=ATTN_IMPL)
    base.eval()
    free, _ = torch.cuda.mem_get_info()
    print(f"[probe-A] base loaded; GPU free={free/1e9:.2f}G")
    base_tokens = gen_one(base, tokenizer, prompts)
    torch.save(base_tokens, TOKENS_BASE)
    print(f"[probe-A] saved {len(base_tokens)} base completions -> {TOKENS_BASE}")
    del base
    _free_gpu("base")


def phase_b_wrapper(prompts, tokenizer) -> None:
    print(f"\n[probe-B] === WRAPPER v6E T=1 === (attn={ATTN_IMPL})")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, trust_remote_code=True, quantization_config=QUANT,
        torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation=ATTN_IMPL)
    base.eval()

    cfg = MythicRDTDeepseekV2Config(
        prelude_layers=4, coda_layers=4, recurrent_layer_idx=10,
        recurrent_block_start=4, recurrent_block_end=22,
        block_mode=True, first_iter_identity=True,
        gate_init_bias=0.0, layerscale_init=1e-4, layerscale_clamp_max=1e-2,
        train_loop_iters=1, max_loop_iters=4, base_model_path=BASE)
    wrapper = MythicRDTDeepseekV2ForCausalLM(cfg, base=base).to(DEVICE)
    wrapper.eval()
    inject_depth_lora(wrapper,
        targets=["self_attn.q_proj_or_q_a", "self_attn.o_proj"],
        rank=8, alpha=16.0, lora_dtype=DTYPE)
    state = torch.load(f"{CKPT}/{TRAINABLE_STATE_FN}",
                       map_location="cpu", weights_only=True)
    loaded, missing, unexpected = _load_trainable_state(wrapper, state)
    print(f"[probe-B] wrapper loaded={loaded} miss={len(missing)} unex={len(unexpected)}")
    free, _ = torch.cuda.mem_get_info()
    print(f"[probe-B] wrapper ready; GPU free={free/1e9:.2f}G")

    wrap_tokens = gen_one(wrapper, tokenizer, prompts, T=1)
    torch.save(wrap_tokens, TOKENS_WRAP)
    print(f"[probe-B] saved {len(wrap_tokens)} wrapper completions -> {TOKENS_WRAP}")
    del wrapper, base
    _free_gpu("wrapper")


def phase_c_compare(prompts, tokenizer) -> None:
    base_tokens = torch.load(TOKENS_BASE)
    wrap_tokens = torch.load(TOKENS_WRAP)
    print("\n[probe-C] === PER-PROBLEM DIFF ===")
    n_clean = 0
    diverged_idx: list[int] = []
    for i in range(len(prompts)):
        bt, wt = base_tokens[i], wrap_tokens[i]
        n_total = min(len(bt), len(wt))
        first_div = next((j for j in range(n_total) if bt[j] != wt[j]), -1)
        n_match = sum(1 for j in range(n_total) if bt[j] == wt[j])
        clean = first_div == -1 and len(bt) == len(wt)
        marker = "✓" if clean else "✗"
        if clean:
            n_clean += 1
        else:
            diverged_idx.append(i)
        print(f"  {marker} problem {i}: prompt_chars={len(prompts[i])}  "
              f"base_len={len(bt)}  wrap_len={len(wt)}  "
              f"matching={n_match}/{n_total}  first_div={first_div}")
        if first_div >= 0:
            ctx = tokenizer.decode(bt[:first_div], skip_special_tokens=True)
            print(f"    base[{first_div}]={bt[first_div]} ({tokenizer.decode([bt[first_div]])!r})  "
                  f"wrap[{first_div}]={wt[first_div]} ({tokenizer.decode([wt[first_div]])!r})")
            print(f"    shared prefix tail: ...{ctx[-80:]!r}")

    print(f"\n[probe-C] SUMMARY: {n_clean}/{len(prompts)} byte-identical; "
          f"diverged: {diverged_idx}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tokenizer.padding_side = "left"

    print(f"[probe] loading {N_PROBLEMS} LCB-medium prompts...")
    problems = load_lcb(limit=N_PROBLEMS, difficulty="medium",
                       min_date="2024-10-01", testtype="functional")
    prompts = build_lcb_chat_prompts(problems, tokenizer)
    print(f"[probe] got {len(prompts)} prompts; lengths (chars): {[len(p) for p in prompts]}")

    phase_a_base(prompts, tokenizer)
    phase_b_wrapper(prompts, tokenizer)
    phase_c_compare(prompts, tokenizer)


if __name__ == "__main__":
    main()
