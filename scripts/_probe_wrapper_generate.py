"""End-to-end probe: wrapper.generate() vs base.generate() byte-equality at v6E + T=1.

If v6E + T=1 truly equals base mathematically AND the wrapper exposes a proper
HF GenerationMixin interface, then:

  base.generate(ids, max_new_tokens=N) == wrapper.generate(ids, max_new_tokens=N, T=1)

byte-for-byte. This is the unblocker for trusting smoke numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mythic_rdt.configuration import MythicRDTDeepseekV2Config
from mythic_rdt.modeling import MythicRDTDeepseekV2ForCausalLM
from mythic_rdt.training import inject_depth_lora
from mythic_rdt.training.trainer import TRAINABLE_STATE_FN, _load_trainable_state


BASE = "base/DeepSeek-Coder-V2-Lite-Instruct"
CKPT = "checkpoints/phase1_v6a_dual_t/checkpoint-200"
N_NEW = 64  # short enough to be fast, long enough to surface divergence


def main():
    device = "cuda"
    dtype = torch.bfloat16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    )
    print(f"[probe] loading base from {BASE}...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, trust_remote_code=True, quantization_config=quant,
        torch_dtype=dtype, device_map=device,
    )
    base.eval()

    cfg = MythicRDTDeepseekV2Config(
        prelude_layers=4, coda_layers=4, recurrent_layer_idx=10,
        recurrent_block_start=4, recurrent_block_end=22,
        block_mode=True, first_iter_identity=True,
        gate_init_bias=0.0, layerscale_init=1e-4, layerscale_clamp_max=1e-2,
        train_loop_iters=1, max_loop_iters=4, base_model_path=BASE,
    )
    print("[probe] building v6E wrapper (first_iter_identity=True, max_loop_iters=4)...")
    wrapper = MythicRDTDeepseekV2ForCausalLM(cfg, base=base).to(device)
    wrapper.eval()

    print("[probe] injecting DepthLoRA + loading v6A ckpt-200 trainable state...")
    inject_depth_lora(
        wrapper,
        targets=["self_attn.q_proj_or_q_a", "self_attn.o_proj"],
        rank=8, alpha=16.0, lora_dtype=dtype,
    )
    state = torch.load(f"{CKPT}/{TRAINABLE_STATE_FN}",
                       map_location="cpu", weights_only=True)
    loaded, missing, unexpected = _load_trainable_state(wrapper, state)
    print(f"[probe] loaded={loaded}  missing={len(missing)}  unexpected={len(unexpected)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    text = (
        "def fibonacci(n):\n"
        "    \"\"\"Return the nth Fibonacci number using iteration.\"\"\"\n"
    )
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    print(f"[probe] input_ids.shape = {ids.shape}")

    print("\n[probe] === base.generate ===")
    with torch.no_grad():
        base_out = base.generate(
            input_ids=ids, attention_mask=attn,
            max_new_tokens=N_NEW, do_sample=False,
            pad_token_id=pad_id, use_cache=True,
        )
    base_new = base_out[0, ids.shape[1]:].tolist()
    print(f"  generated {len(base_new)} new tokens")
    print(f"  decoded: {tokenizer.decode(base_new, skip_special_tokens=True)!r}")

    print("\n[probe] === wrapper.generate(T=1) ===")
    try:
        with torch.no_grad():
            wrap_out = wrapper.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=N_NEW, do_sample=False,
                pad_token_id=pad_id, use_cache=True,
                T=1,
            )
        wrap_new = wrap_out[0, ids.shape[1]:].tolist()
        print(f"  generated {len(wrap_new)} new tokens")
        print(f"  decoded: {tokenizer.decode(wrap_new, skip_special_tokens=True)!r}")
    except Exception as e:
        print(f"  ✗ wrapper.generate FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return

    # Compare
    n_match = sum(1 for a, b in zip(base_new, wrap_new) if a == b)
    n_total = min(len(base_new), len(wrap_new))
    print(f"\n[probe] === COMPARISON ===")
    print(f"  base len={len(base_new)}  wrapper len={len(wrap_new)}")
    print(f"  matching tokens: {n_match}/{n_total}")
    if n_match == n_total and len(base_new) == len(wrap_new):
        print("  ✓ BYTE-IDENTICAL — wrapper.generate(T=1) ≡ base.generate() under v6E")
    else:
        # Find first divergence
        for i in range(n_total):
            if base_new[i] != wrap_new[i]:
                print(f"  ✗ first divergence at position {i}:")
                print(f"      base[{i}]={base_new[i]} ({tokenizer.decode([base_new[i]])!r})")
                print(f"      wrap[{i}]={wrap_new[i]} ({tokenizer.decode([wrap_new[i]])!r})")
                ctx = tokenizer.decode(base_new[:i], skip_special_tokens=True)
                print(f"      shared prefix: ...{ctx[-60:]!r}")
                break


if __name__ == "__main__":
    main()
