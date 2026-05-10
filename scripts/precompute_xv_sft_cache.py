"""Per-prompt cross-vocab teacher cache for v6V joint SFT+recurrence.

Iterates the funcsig corpus (same JSONL the joint trainer reads), runs the
teacher (QC-14B at NF4) on prompt+completion text, and produces a per-
example top-K teacher cache PROJECTED to the student vocab via CTD's
VocabMapper. Output is aligned to the student's tokenized
chat-template-wrapped prompt + completion sequence — drop-in for the
joint trainer's --teacher-logits-xv flag.

Output schema:
    {
      "indices":        int32 tensor [N_examples, max_total_len, top_k]
                        (in student vocab)
      "values":         float16 tensor [N_examples, max_total_len, top_k]
                        log-probabilities over the top-K support
      "alignment_mask": bool tensor [N_examples, max_total_len]
                        True where the position has a valid projected
                        teacher logit; False for prompt positions, padded
                        positions, or alignment failures
      "meta": {...}
    }

Index ordering MUST match TeacherCompletionDataset's __getitem__ — i.e.
example i in the cache corresponds to records[i]. The joint trainer's
DataLoader exposes the original index via __ds_idx__ for indexing.

Cost: ~5-10 s per example on NF4 QC-14B (1 forward at L≈1000 + alignment
+ projection + cache write). Funcsig has 474 prompts → ~30-60 min total.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

_CTD_DEFAULT = "/workspace/cross-tokenizer-distill"
_ctd_path = os.environ.get("CTD_REPO", _CTD_DEFAULT)
if Path(_ctd_path).exists():
    sys.path.insert(0, _ctd_path)
    sys.path.insert(0, str(Path(_ctd_path) / "experiments" / "validation"))

from ctd import VocabMapper  # noqa: E402
from ctd.alignment import build_alignment  # noqa: E402
from ctd.precompute import _project_or_passthrough  # noqa: E402

# Borrow the same dataset class so prompt+completion construction is identical
# to what the trainer sees. Index alignment is by integer position in records.
import importlib.util as _ilu  # noqa: E402
_ctd_sft_path = Path(_ctd_path) / "experiments" / "validation" / "06_train_sft_on_teacher.py"
_spec = _ilu.spec_from_file_location("ctd_sft", _ctd_sft_path)
_ctd_sft_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ctd_sft_mod)
TeacherCompletionDataset = _ctd_sft_mod.TeacherCompletionDataset


def _load_teacher(path: str, dtype: torch.dtype, quant: str):
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "cuda",
        "torch_dtype": dtype,
    }
    if quant != "none":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[xv-sft] loading teacher {quant.upper()}")
    return AutoModelForCausalLM.from_pretrained(path, **load_kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-tokenizer", required=True)
    ap.add_argument("--student-tokenizer", required=True)
    ap.add_argument("--corpus", required=True, help="funcsig JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--max-prompt-len", type=int, default=384)
    ap.add_argument("--max-total-len", type=int, default=1024)
    ap.add_argument("--chat-template", action="store_true",
                    help="Wrap prompt with student's chat template — MUST "
                         "match the trainer's --chat-template setting.")
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--multi-token", default="distribute",
                    choices=["strict", "distribute", "first_token"])
    ap.add_argument("--alignment", default="student_offset",
                    choices=["student_offset", "byte_anchor"])
    ap.add_argument("--quant", default="nf4", choices=["none", "nf4", "fp4"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    ap.add_argument("--shard-every", type=int, default=100)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    # Tokenizers + mapper.
    print("[xv-sft] loading tokenizers...")
    teacher_tok = AutoTokenizer.from_pretrained(
        args.teacher_tokenizer, trust_remote_code=True
    )
    student_tok = AutoTokenizer.from_pretrained(
        args.student_tokenizer, trust_remote_code=True
    )
    print(f"[xv-sft] teacher_v={teacher_tok.vocab_size} "
          f"student_v={student_tok.vocab_size}")

    print(f"[xv-sft] building VocabMapper(multi_token={args.multi_token})...")
    mapper = VocabMapper.from_tokenizers(
        teacher_tokenizer=teacher_tok,
        student_tokenizer=student_tok,
        multi_token=args.multi_token,
        progress=True,
    )
    print(mapper.coverage_report() if hasattr(mapper, "coverage_report") else mapper._report)

    # Teacher.
    teacher = _load_teacher(args.teacher, dtype=dtype, quant=args.quant)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    device = next(teacher.parameters()).device

    # Build the same dataset the trainer will use (index parity).
    if student_tok.pad_token is None and student_tok.eos_token is not None:
        student_tok.pad_token = student_tok.eos_token
    ds = TeacherCompletionDataset(
        args.corpus, student_tok,
        max_prompt_len=args.max_prompt_len,
        max_total_len=args.max_total_len,
        code_only_mask=False,  # not relevant for cache build
        chat_template=args.chat_template,
        system_prompt=args.system_prompt,
    )
    print(f"[xv-sft] dataset N={len(ds)}")

    K = args.top_k
    L_max = args.max_total_len
    N = len(ds)

    out_indices = torch.zeros((N, L_max, K), dtype=torch.int32)
    out_values = torch.zeros((N, L_max, K), dtype=torch.float16)
    out_mask = torch.zeros((N, L_max), dtype=torch.bool)

    n_aligned_total = 0
    n_completion_total = 0
    n_dropped = 0
    t_start = time.time()
    last_log = t_start

    for i in range(N):
        item = ds[i]
        student_ids = item["input_ids"]            # list[int], len ≤ L_max
        prompt_len = item["prompt_len"]            # int

        # Reconstruct text for teacher input.
        # We reconstruct from the student's tokenized form so the teacher
        # sees the exact same "raw" content (decoded from student tokens).
        text = student_tok.decode(student_ids, skip_special_tokens=False)

        # Re-tokenize with teacher.
        teacher_ids = teacher_tok.encode(text, add_special_tokens=False)
        if not teacher_ids:
            n_dropped += 1
            continue

        # Run teacher.
        try:
            t_input = torch.tensor([teacher_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                t_out = teacher(input_ids=t_input, use_cache=False)
                t_logits = t_out.logits[0]  # [L_t, V_t]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_dropped += 1
            continue

        # Build alignment.
        try:
            table = build_alignment(
                text=text,
                teacher_token_ids=teacher_ids,
                student_token_ids=list(student_ids),
                teacher_tokenizer=teacher_tok,
                student_tokenizer=student_tok,
                mode=args.alignment,
                suffix_reencode=False,  # cheaper; per-prompt corpus is short
            )
        except Exception as exc:
            print(f"[xv-sft] ex {i} alignment failed: {exc!r}")
            n_dropped += 1
            continue

        # Per-position projection — only on COMPLETION positions (>= prompt_len).
        # Prompt positions get mask=False (they're loss-masked anyway).
        n_aligned_block = 0
        n_completion_block = 0
        for pos, entry in enumerate(table.entries):
            if pos >= L_max:
                break
            if pos < prompt_len:
                continue  # skip prompt-side positions
            n_completion_block += 1
            if not entry.valid or entry.suffix_token_ids is not None:
                # We disabled suffix_reencode → suffix entries never appear,
                # but defensively skip them. Invalid alignment also skipped.
                continue
            try:
                t_logit = t_logits[entry.teacher_pos, :]  # [V_t]
                log_proj, ids_proj = _project_or_passthrough(
                    logit_or_topk=t_logit,
                    topk_indices=None,
                    top_k=K,
                    projection=mapper,
                    project_at_write_time=True,
                )
                out_indices[i, pos, :] = ids_proj.to(torch.int32).cpu()
                out_values[i, pos, :] = log_proj.to(torch.float16).cpu()
                out_mask[i, pos] = True
                n_aligned_block += 1
            except Exception as exc:
                if pos == prompt_len:
                    print(f"[xv-sft] ex {i} pos {pos}: {exc!r}")
                continue

        n_aligned_total += n_aligned_block
        n_completion_total += n_completion_block

        del t_input, t_out, t_logits
        if (i + 1) % 16 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Progress.
        now = time.time()
        if now - last_log >= 30 or (i + 1) >= N:
            elapsed = now - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (N - i - 1) / max(rate, 1e-6)
            cov = n_aligned_total / max(n_completion_total, 1)
            print(f"[xv-sft] ex {i+1:5d}/{N}  ({100*(i+1)/N:5.1f}%)  "
                  f"{rate:.2f} ex/s  ETA={eta/60:.1f} min  "
                  f"completion_alignment={cov:.2%}  dropped={n_dropped}",
                  flush=True)
            last_log = now

        # Periodic shard.
        if args.shard_every > 0 and (i + 1) % args.shard_every == 0 and (i + 1) < N:
            shard = out_path.with_name(out_path.stem + f".shard{i+1:06d}.pt")
            torch.save({
                "indices": out_indices[:i+1].clone(),
                "values": out_values[:i+1].clone(),
                "alignment_mask": out_mask[:i+1].clone(),
                "meta": {"shard_until": i + 1, "n_target": N},
            }, shard)
            print(f"[xv-sft] wrote shard {shard.name}")

    coverage = n_aligned_total / max(n_completion_total, 1)
    meta = {
        "teacher_model": args.teacher,
        "teacher_tokenizer": args.teacher_tokenizer,
        "student_tokenizer": args.student_tokenizer,
        "corpus": args.corpus,
        "alignment": args.alignment,
        "projection_strategy": args.multi_token,
        "project_at_write_time": True,
        "top_k": args.top_k,
        "max_total_len": args.max_total_len,
        "max_prompt_len": args.max_prompt_len,
        "n_examples": N,
        "n_dropped": n_dropped,
        "completion_alignment_coverage": float(coverage),
        "chat_template": args.chat_template,
        "ctd_version": getattr(__import__("ctd"), "__version__", "unknown"),
        "dtype": args.dtype,
        "quant": args.quant,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    print(f"[xv-sft] writing final {out_path} (N={N}, "
          f"completion_coverage={coverage:.2%})")
    torch.save({
        "indices": out_indices,
        "values": out_values,
        "alignment_mask": out_mask,
        "meta": meta,
    }, out_path)
    print(f"[xv-sft] DONE in {(time.time()-t_start)/60:.1f} min")
    print(f"[xv-sft] meta: {json.dumps(meta, default=str, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
