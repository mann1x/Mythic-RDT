#!/usr/bin/env python
"""Measure mean cross-entropy loss of the BASE model on the same packed
data stream the trainer uses. Diagnostic to compare against the wrapper
loss plateau and decide whether the wrapper has converged to base or
is still under-trained.

Quick: ~20 batches at the same seq_len/batch_size as training. NF4 base.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(THIS_FILE.parent))

from mythic_rdt.training.data import build_packed_dataset  # noqa: E402
from _dscoder_compat import dtype_kwarg, load_dscoder_tokenizer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="base/DeepSeek-Coder-V2-Lite-Instruct")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--data-seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16

    print(f"[base-loss] loading base ({args.base}) NF4")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, device_map="cuda",
        quantization_config=bnb, low_cpu_mem_usage=True, **dtype_kwarg(dtype),
    )
    base.eval()
    tokenizer = load_dscoder_tokenizer(args.base)

    print(f"[base-loss] streaming {args.n_batches} batches "
          f"(batch={args.batch_size} seq={args.seq_len})")
    ds = build_packed_dataset(
        tokenizer=tokenizer, seq_len=args.seq_len, seed=args.data_seed,
    )
    it = iter(ds)
    losses = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(args.n_batches):
            batch = []
            for _ in range(args.batch_size):
                batch.append(next(it)["input_ids"])
            input_ids = torch.stack(batch).to("cuda")
            labels = input_ids.clone()
            out = base(input_ids=input_ids, labels=labels)
            losses.append(float(out.loss.item()))
            print(f"[base-loss]   batch {i+1}/{args.n_batches}  loss={losses[-1]:.4f}")
    print(f"[base-loss] elapsed {time.time()-t0:.1f}s")
    import statistics as st
    print(f"\n[base-loss] === SUMMARY ===")
    print(f"  n={len(losses)}  mean={st.mean(losses):.4f}  "
          f"median={st.median(losses):.4f}  "
          f"min={min(losses):.4f}  max={max(losses):.4f}  "
          f"stdev={st.stdev(losses):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
