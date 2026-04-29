"""Direct A/B: wrapper v6E (first_iter_identity, LoRA disabled at t=0) at T=1 vs base.

Loads v6A ckpt-200 trainable weights into a v6E wrapper, runs one forward pass
on a fixed batch, compares logits to base. With v6E semantics, T=1 wrapper output
must be base output byte-for-byte (modulo BF16 numerics).

Usage:
  PYTHONDONTWRITEBYTECODE=1 python scripts/_probe_v6e_identity.py
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
from mythic_rdt.training.trainer import (
    TRAINABLE_STATE_FN, _load_trainable_state,
)
from mythic_rdt.loop_state import set_loop_t, get_loop_t


BASE = "base/DeepSeek-Coder-V2-Lite-Instruct"
CKPT = "checkpoints/phase1_v6a_dual_t/checkpoint-200"

# Instrument set_loop_t to log every call
_orig_set = set_loop_t
_call_log: list[int] = []
def _instr_set(t):
    _call_log.append(int(t))
    _orig_set(t)
import mythic_rdt.loop_state as _ls
_ls.set_loop_t = _instr_set
import mythic_rdt.modeling as _modeling
_modeling.set_loop_t = _instr_set


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
        prelude_layers=4, coda_layers=4,
        recurrent_layer_idx=10,
        recurrent_block_start=4, recurrent_block_end=22,
        block_mode=True, first_iter_identity=True,
        gate_init_bias=0.0, layerscale_init=1e-4, layerscale_clamp_max=1e-2,
        train_loop_iters=1, max_loop_iters=4,
        base_model_path=BASE,
    )
    print("[probe] building v6E wrapper (first_iter_identity=True, max_loop_iters=4)...")
    wrapper = MythicRDTDeepseekV2ForCausalLM(cfg, base=base).to(device)
    wrapper.eval()

    print(f"[probe] injecting DepthLoRA scaffold...")
    inject_depth_lora(
        wrapper,
        targets=["self_attn.q_proj_or_q_a", "self_attn.o_proj"],
        rank=8, alpha=16.0,
        lora_dtype=dtype,
    )
    state = torch.load(f"{CKPT}/{TRAINABLE_STATE_FN}",
                       map_location="cpu", weights_only=True)
    loaded, missing, unexpected = _load_trainable_state(wrapper, state)
    print(f"[probe] loaded={loaded}  missing={len(missing)}  unexpected={len(unexpected)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    text = "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    print(f"[probe] input_ids.shape = {ids.shape}")

    def _to_logits(o):
        return o.logits if hasattr(o, "logits") else o

    _call_log.clear()
    print("\n[probe] === BASE forward ===")
    with torch.no_grad():
        base_out = _to_logits(base(ids))
    print(f"  base_out.shape={tuple(base_out.shape)}  dtype={base_out.dtype}")

    _call_log.clear()
    print("\n[probe] === WRAPPER v6E T=1 forward ===")
    with torch.no_grad():
        wrap_out = _to_logits(wrapper(ids, T=1))
    print(f"  wrap_out.shape={tuple(wrap_out.shape)}  dtype={wrap_out.dtype}")
    print(f"  set_loop_t(...) calls during wrapper forward: {_call_log}")
    if -1 in _call_log:
        print("  ✓ set_loop_t(-1) was called — v6E LoRA-disable path fired")
    else:
        print("  ✗ set_loop_t(-1) was NOT called — v6E disable did not fire (BUG)")

    # Compare last-token logits
    diff = (base_out - wrap_out).float()
    abs_diff = diff.abs()
    print(f"\n[probe] === LOGIT DIFF (wrapper - base) ===")
    print(f"  shape: {tuple(abs_diff.shape)}")
    print(f"  max |diff|: {abs_diff.max().item():.6e}")
    print(f"  mean |diff|: {abs_diff.mean().item():.6e}")
    print(f"  >0.01 fraction: {(abs_diff > 0.01).float().mean().item():.6f}")
    print(f"  >0.1  fraction: {(abs_diff > 0.1).float().mean().item():.6f}")

    # Compare argmax (next-token prediction at last position)
    base_argmax = base_out[0, -1].argmax().item()
    wrap_argmax = wrap_out[0, -1].argmax().item()
    print(f"\n[probe] last-token argmax: base={base_argmax} ({tokenizer.decode([base_argmax])!r})  "
          f"wrap={wrap_argmax} ({tokenizer.decode([wrap_argmax])!r})  "
          f"match={base_argmax == wrap_argmax}")

    # All positions argmax match?
    base_argmaxes = base_out[0].argmax(-1)
    wrap_argmaxes = wrap_out[0].argmax(-1)
    n_match = (base_argmaxes == wrap_argmaxes).sum().item()
    n_total = base_argmaxes.shape[0]
    print(f"  per-position argmax match: {n_match}/{n_total}")


if __name__ == "__main__":
    main()
