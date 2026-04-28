#!/usr/bin/env python
"""Phase 0 hard gate (Stage 1, DS-Coder-V2-Lite-Instruct).

MASTER_PLAN.md §5: at T=1 with `force_gate_zero=True`, the wrapper's
output must be bit-exact with the manual three-layer base pipeline:

    embed -> layer[0] -> layer[recurrent_idx] -> layer[26] -> norm -> lm_head

This script:
  1. Loads the base DS-Coder-V2-Lite-Instruct once (~30 GB).
  2. For each candidate `recurrent_layer_idx` in {10, 13, 16}, builds a
     `MythicRDTDeepseekV2ForCausalLM(prelude=1, coda=1, idx=<c>)`.
  3. Runs both the wrapper (T=1, force_gate_zero=True) and the manual
     reference pipeline on the same input.
  4. Asserts max-abs-diff == 0.0 in fp32 (bit-exact). Hard gate.

Usage:
    conda activate mythic-rdt
    python scripts/phase0_sanity.py \\
        --base base/DeepSeek-Coder-V2-Lite-Instruct \\
        --candidates 10 13 16 \\
        --dtype bfloat16 \\
        --device auto \\
        --seq-len 32 --batch-size 1

Bit-exactness: floating-point ops are deterministic given identical
inputs and identical operation order. Both code paths run the SAME
layer modules in the SAME order on the SAME tensors, so byte-equality
of the resulting logits is the test (not "close enough").

Notes on dtype:
    - bfloat16 is fine for bit-exactness too -- both paths share the
      same dtype, so rounding is identical.
    - fp32 is more tolerant if a downstream backend introduces TF32 or
      flash-attention nondeterminism.
    - The script defaults to bfloat16 (matches base on-disk dtype) and
      reports max-abs-diff so any non-zero is loudly visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make `src/mythic_rdt/...` importable when running the script from repo root
# without installing the package.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mythic_rdt.configuration import MythicRDTDeepseekV2Config  # noqa: E402
from mythic_rdt.modeling import MythicRDTDeepseekV2ForCausalLM  # noqa: E402


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mythic-RDT Stage 1 Phase 0 sanity gate")
    p.add_argument(
        "--base",
        type=str,
        default="base/DeepSeek-Coder-V2-Lite-Instruct",
        help="Path to the base DS-Coder-V2-Lite-Instruct checkpoint dir.",
    )
    p.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=[10, 13, 16],
        help="Recurrent layer indices to probe (default: 10 13 16).",
    )
    p.add_argument(
        "--prelude-layers",
        type=int,
        default=1,
        help="Number of base layers used verbatim before recurrence.",
    )
    p.add_argument(
        "--coda-layers",
        type=int,
        default=1,
        help="Number of base layers used verbatim after recurrence.",
    )
    p.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP))
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'cpu', 'cuda', 'cuda:N', or 'auto' (cuda if available else cpu).",
    )
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def fmt_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "a_max_abs": float(a.float().abs().max().item()),
        "b_max_abs": float(b.float().abs().max().item()),
    }


def make_inputs(
    tokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Random token ids inside the model's vocab. Avoids special tokens."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab_size = tokenizer.vocab_size if tokenizer is not None else 100_000
    # Stay clear of the very low ids (special tokens) and high reserved range.
    low = 100
    high = max(low + 1, vocab_size - 100)
    ids = torch.randint(
        low, high, (batch_size, seq_len), generator=g, dtype=torch.long
    )
    return ids.to(device)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    base_path = Path(args.base)
    if not base_path.exists():
        print(f"ERROR: base path does not exist: {base_path}", file=sys.stderr)
        return 2
    print(f"[phase0] base={base_path}")
    print(f"[phase0] device={device} dtype={dtype}")
    print(f"[phase0] candidates={args.candidates}")
    print(f"[phase0] prelude={args.prelude_layers} coda={args.coda_layers}")

    print("[phase0] loading base model (this is the slow part)...")
    base = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base.eval()
    base.to(device)
    print(
        f"[phase0] base loaded: hidden_size={base.config.hidden_size} "
        f"layers={base.config.num_hidden_layers}"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_path), trust_remote_code=True
        )
    except Exception as exc:  # pragma: no cover - tokenizer is best-effort here
        print(f"[phase0] tokenizer load failed ({exc!r}); using random ids only")
        tokenizer = None

    input_ids = make_inputs(
        tokenizer, args.batch_size, args.seq_len, device, args.seed
    )

    overall_pass = True
    for cand in args.candidates:
        print(f"\n[phase0] === recurrent_layer_idx = {cand} ===")
        cfg = MythicRDTDeepseekV2Config(
            prelude_layers=args.prelude_layers,
            coda_layers=args.coda_layers,
            recurrent_layer_idx=cand,
            train_loop_iters=1,
            max_loop_iters=8,
            base_model_path=str(base_path),
        )
        try:
            wrapper = MythicRDTDeepseekV2ForCausalLM(cfg, base=base).to(device)
        except ValueError as exc:
            print(f"[phase0] config rejected: {exc}")
            overall_pass = False
            continue

        wrapper.eval()
        with torch.no_grad():
            wrapper_logits = wrapper(input_ids, T=1, force_gate_zero=True)
            ref_logits = wrapper.base_three_layer_pass(input_ids)

        stats = fmt_diff(wrapper_logits, ref_logits)
        print(
            f"[phase0] wrapper vs ref:  "
            f"max_abs_diff={stats['max_abs_diff']:.3e}  "
            f"mean_abs_diff={stats['mean_abs_diff']:.3e}  "
            f"|logits|_max={stats['a_max_abs']:.3e}"
        )

        bit_exact = stats["max_abs_diff"] == 0.0
        if bit_exact:
            print(f"[phase0] PASS: bit-exact at recurrent_layer_idx={cand}")
        else:
            print(
                f"[phase0] FAIL: not bit-exact at recurrent_layer_idx={cand} "
                f"(diff={stats['max_abs_diff']:.3e})"
            )
            overall_pass = False

        # Also assert we did not accidentally collide with base's first
        # forward when force_gate_zero is OFF: at default init (gate~0,
        # ls=1e-4) the loop must move the residual SLIGHTLY but not by
        # much. This is a separate, weaker check we log but don't gate.
        with torch.no_grad():
            normal_logits = wrapper(input_ids, T=1, force_gate_zero=False)
        normal_stats = fmt_diff(normal_logits, ref_logits)
        print(
            f"[phase0] T=1 default-init vs ref:  "
            f"max_abs_diff={normal_stats['max_abs_diff']:.3e}  "
            f"mean_abs_diff={normal_stats['mean_abs_diff']:.3e}  "
            "(non-zero expected; loop is near-identity, not identity)"
        )

    print()
    if overall_pass:
        print("[phase0] OVERALL: PASS - wrapper plumbing is correct.")
        return 0
    print("[phase0] OVERALL: FAIL - investigate before any fine-tune work.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
