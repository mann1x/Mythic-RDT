#!/usr/bin/env python
"""Phase 0 layer-quality probe (Stage 1, DS-Coder-V2-Lite-Instruct).

MASTER_PLAN.md §5 phase 0 step "pick recurrent_layer_idx":

    Probe layers 10/13/16 untrained at T=1/T=4/T=8 on 100 prompts (mix
    of HumanEval-style code + FineWeb-Edu prose). Choose cleanest
    behavior. Decision gate: if T=8 untrained drops > 50 % PPL or
    produces gibberish on > 20 % of prompts, the architecture choice
    needs revisiting before phase 1.

Per-layer × per-T metrics (untrained wrapper, default RDT init):

  - **PPL ratio vs T=1**: mean (PPL_T / PPL_{T=1}). At init the loop is
    near-identity (gate ~ 5e-6), so the prelude output trickles through
    almost unchanged across T. PPL should be roughly stable across T;
    a > 1.5x ratio is the "> 50 % drop" trip.
  - **gibberish rate**: fraction of generations classified as gibberish
    by simple structural heuristics (high non-printable ratio, repeated
    chars, empty / whitespace-only). > 20 % is the trip.

Cost note: runs on CPU since DS-Coder bf16 (~30 GB) overflows a single
3090 (24 GB). The wrapper does T-1 extra forwards through one layer per
prompt-token vs the base, so T=8 with 50 generated tokens at seq=64
takes 30-90 s per prompt on CPU. Defaults are tiny (10 prompts, gen=20
tokens) so a smoke run finishes in minutes; pass --num-prompts and
--gen-tokens to scale up on a real machine.

Usage:
    conda activate mythic-rdt
    # quick smoke (minutes on CPU)
    python scripts/phase0_probe_layers.py --candidates 13 --T-values 1 \\
        --num-prompts 2 --gen-tokens 10

    # full probe (intended on a multi-GPU box; hours on CPU)
    python scripts/phase0_probe_layers.py --candidates 10 13 16 \\
        --T-values 1 4 8 --num-prompts 100 --gen-tokens 64

The decision criterion fires only if `--T-values` includes 8. With
fewer T values the script reports observed metrics but does not gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from mythic_rdt.configuration import MythicRDTDeepseekV2Config  # noqa: E402
from mythic_rdt.modeling import MythicRDTDeepseekV2ForCausalLM  # noqa: E402
from _dscoder_compat import dtype_kwarg, load_dscoder_tokenizer  # noqa: E402


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


# A small in-script prompt pack. Bigger runs should pass --prompts-file.
# Code prompts mirror HumanEval signature+docstring style; prose prompts
# mirror FineWeb-Edu educational paragraphs.
DEFAULT_PROMPTS_CODE: list[str] = [
    'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number, False otherwise."""\n',
    'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number (0-indexed)."""\n',
    'def reverse_string(s: str) -> str:\n    """Return the input string reversed."""\n',
    'def gcd(a: int, b: int) -> int:\n    """Return the greatest common divisor of two non-negative ints."""\n',
    'def count_vowels(text: str) -> int:\n    """Return the number of vowels (aeiou, case-insensitive) in text."""\n',
    'def flatten(nested: list) -> list:\n    """Recursively flatten a nested list of arbitrary depth."""\n',
    'def is_palindrome(s: str) -> bool:\n    """Return True iff s reads the same forward and backward, ignoring case."""\n',
    'def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n    """Merge two sorted ascending lists into one sorted ascending list."""\n',
    'def sum_digits(n: int) -> int:\n    """Return the sum of decimal digits of a non-negative integer n."""\n',
    'def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    """Return the transpose of a 2D matrix represented as a list of lists."""\n',
]

DEFAULT_PROMPTS_PROSE: list[str] = [
    "Photosynthesis is the process by which green plants and certain other organisms convert light energy, usually from the sun, into chemical energy stored in glucose. The overall reaction can be summarized as ",
    "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth. Key stages include evaporation, condensation, precipitation, and ",
    "Newton's three laws of motion form the foundation of classical mechanics. The first law states that an object at rest stays at rest and an object in motion stays in motion unless ",
    "DNA, or deoxyribonucleic acid, is a long molecule that contains an organism's unique genetic code. It carries the instructions a cell needs to make ",
    "The mitochondria are membrane-bound organelles found in the cytoplasm of nearly all eukaryotic cells. They are often called the powerhouse of the cell because their primary function is to ",
    "Plate tectonics is the scientific theory that the Earth's lithosphere is divided into a number of large slabs called tectonic plates that move relative to one another. The boundaries between plates are sites of ",
    "In computer science, a binary search tree is a data structure where each node has at most two children, and the left subtree of a node contains only nodes with values less than ",
    "Machine learning is a branch of artificial intelligence that focuses on systems that learn from data rather than from explicit programming. A common workflow involves splitting the data into ",
    "The Roman Republic was the era of classical Roman civilization beginning with the overthrow of the Roman Kingdom and ending with the establishment of the Roman Empire. During this period, political power was held by ",
    "An ecosystem consists of all the organisms living in a particular area together with the non-living components of their environment. Energy flows through an ecosystem starting from ",
]


# ---------------------------------------------------------------------------
# Gibberish heuristic
# ---------------------------------------------------------------------------


REPEATED_CHAR_RUN = re.compile(r"(.)\1{6,}")  # 7+ identical chars in a row
WHITESPACE_ONLY = re.compile(r"^\s*$")


def is_gibberish(text: str) -> tuple[bool, str]:
    """Cheap structural gibberish detector.

    Returns (is_gibberish, reason). A continuation is flagged when:
      - empty / whitespace-only
      - > 50 % non-printable (excluding standard whitespace) characters
      - contains a 7+ char run of a single repeated character

    These catch the failure modes we actually expect from an untrained
    recurrence loop (mode collapse to a single token, NaN-driven
    nonsense, empty generation). They do NOT catch subtle semantic
    drift -- that is left to PPL.
    """
    if not text or WHITESPACE_ONLY.match(text):
        return True, "empty/whitespace"
    printable = sum(
        1 for c in text if c.isprintable() or c in ("\n", "\t")
    )
    nonprintable_ratio = 1 - printable / max(len(text), 1)
    if nonprintable_ratio > 0.5:
        return True, f"non-printable={nonprintable_ratio:.2f}"
    if REPEATED_CHAR_RUN.search(text):
        m = REPEATED_CHAR_RUN.search(text)
        return True, f"repeated-char-run={m.group(0)[:8]!r}"
    return False, "ok"


# ---------------------------------------------------------------------------
# PPL on a forward pass
# ---------------------------------------------------------------------------


@torch.no_grad()
def teacher_forced_ppl_batch(
    wrapper: MythicRDTDeepseekV2ForCausalLM,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    T: int,
) -> list[float]:
    """Per-row mean cross-entropy. Pads are excluded from the average.

    Returns one float per row (length = batch_size). Pad tokens are
    masked out via attention_mask so a short prompt's mean PPL is not
    diluted by predicting <pad> -> <pad>.
    """
    logits = wrapper(input_ids, attention_mask=attention_mask, T=T)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()  # [B, S-1]

    # Per-position CE.
    B, Sm1, V = shift_logits.shape
    flat_loss = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(B, Sm1)
    # Mask + mean over valid positions per row.
    valid = shift_mask.sum(dim=1).clamp_min(1.0)
    per_row = (flat_loss * shift_mask).sum(dim=1) / valid
    return per_row.tolist()


# ---------------------------------------------------------------------------
# Greedy generation -- manual since the wrapper is plain nn.Module
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_generate_batch(
    wrapper: MythicRDTDeepseekV2ForCausalLM,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    T: int,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Batched greedy decoding through the wrapper, no KV cache.

    Each step does a full forward over the (growing) sequence -- v0
    wrapper has no cache. Per-row "done" tracked via eos_token_id; once
    a row is done we still keep it in the batch (HF-standard) but pad
    its newly produced positions with `pad_token_id` so it doesn't
    influence downstream printing.

    Returns:
        cur: [B, prompt_len + max_new_tokens] generated ids.
        cur_mask: matching attention mask (1 = real, 0 = pad).
    """
    pad = pad_token_id if pad_token_id is not None else 0
    cur = input_ids.clone()
    cur_mask = attention_mask.clone()
    B = cur.size(0)
    done = torch.zeros(B, dtype=torch.bool, device=cur.device)

    for _ in range(max_new_tokens):
        logits = wrapper(cur, attention_mask=cur_mask, T=T)
        next_id = logits[:, -1, :].argmax(dim=-1)  # [B]
        # Frozen rows keep emitting pad so we don't churn their text.
        next_id = torch.where(done, torch.full_like(next_id, pad), next_id)
        if eos_token_id is not None:
            done = done | (next_id == eos_token_id)
        next_col = next_id.unsqueeze(1)
        cur = torch.cat([cur, next_col], dim=1)
        # New column is real for not-yet-done rows; pad mask = 0 for done.
        new_mask_col = (~done).long().unsqueeze(1)
        cur_mask = torch.cat([cur_mask, new_mask_col], dim=1)
        if done.all():
            break
    return cur, cur_mask


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    layer: int
    T: int
    n_prompts: int
    mean_ppl: float
    gibberish_rate: float
    elapsed_sec: float
    gibberish_examples: list[dict]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mythic-RDT Stage 1 Phase 0 layer-quality probe"
    )
    p.add_argument(
        "--base",
        type=str,
        default="base/DeepSeek-Coder-V2-Lite-Instruct",
    )
    p.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=[10, 13, 16],
        help="Recurrent layer indices to probe.",
    )
    p.add_argument(
        "--T-values",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Loop iteration counts to test at each candidate layer.",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts (split half code / half prose by default).",
    )
    p.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional JSONL with one {'text': ...} per line. Overrides "
        "the in-script default prompt pack.",
    )
    p.add_argument("--gen-tokens", type=int, default=20)
    p.add_argument("--prompt-truncate-tokens", type=int, default=64)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Prompts per forward pass. On a 48 GB A6000 with seq~128 "
        "and bf16 base, batch=8 typically fits comfortably; the bottleneck "
        "is hidden-state activations during generation, not weights.",
    )
    p.add_argument("--prelude-layers", type=int, default=1)
    p.add_argument("--coda-layers", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write the per-(layer,T) results dict to this JSON path.",
    )
    return p.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_prompts(args) -> list[str]:
    if args.prompts_file:
        prompts: list[str] = []
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                prompts.append(json.loads(line)["text"])
        return prompts[: args.num_prompts]
    half = max(1, args.num_prompts // 2)
    code = DEFAULT_PROMPTS_CODE[:half]
    prose = DEFAULT_PROMPTS_PROSE[: args.num_prompts - len(code)]
    return code + prose


def probe_one(
    wrapper: MythicRDTDeepseekV2ForCausalLM,
    tokenizer,
    prompts: list[str],
    layer: int,
    T: int,
    gen_tokens: int,
    truncate_tokens: int,
    batch_size: int,
    device: torch.device,
) -> ProbeResult:
    t0 = time.time()
    ppls: list[float] = []
    n_gibberish = 0
    gib_examples: list[dict] = []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        # Left-pad for generation: real tokens at the right edge so the
        # last logit is always over a real token. Tokenizer handles this
        # via padding_side="left".
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=truncate_tokens,
                add_special_tokens=False,
            )
        finally:
            tokenizer.padding_side = old_side

        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)
        if input_ids.shape[1] < 2:
            continue

        # PPL over each row (mask-aware).
        ppls.extend(
            teacher_forced_ppl_batch(wrapper, input_ids, attn_mask, T=T)
        )

        # Greedy generation for each row.
        gen_ids, gen_mask = greedy_generate_batch(
            wrapper,
            input_ids,
            attn_mask,
            T=T,
            max_new_tokens=gen_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_id,
        )
        prompt_len = input_ids.shape[1]
        for r, prompt_text in enumerate(batch_prompts):
            cont_ids = gen_ids[r, prompt_len:].tolist()
            cont = tokenizer.decode(cont_ids, skip_special_tokens=True)
            gib, reason = is_gibberish(cont)
            if gib:
                n_gibberish += 1
                if len(gib_examples) < 3:
                    gib_examples.append(
                        {
                            "prompt_idx": batch_start + r,
                            "reason": reason,
                            "continuation": cont[:120],
                        }
                    )

    elapsed = time.time() - t0
    return ProbeResult(
        layer=layer,
        T=T,
        n_prompts=len(ppls),
        mean_ppl=sum(ppls) / max(1, len(ppls)),
        gibberish_rate=n_gibberish / max(1, len(prompts)),
        elapsed_sec=elapsed,
        gibberish_examples=gib_examples,
    )


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    base_path = Path(args.base)
    if not base_path.exists():
        print(f"ERROR: base path does not exist: {base_path}", file=sys.stderr)
        return 2

    print(f"[probe] base={base_path}")
    print(f"[probe] device={device} dtype={dtype}")
    print(f"[probe] candidates={args.candidates} T-values={args.T_values}")
    print(
        f"[probe] num_prompts={args.num_prompts} gen_tokens={args.gen_tokens} "
        f"truncate={args.prompt_truncate_tokens} batch_size={args.batch_size}"
    )

    import transformers as _tf
    print(f"[probe] transformers={_tf.__version__}")
    print("[probe] loading base (device_map='cuda' for fast load)...")
    # device_map="cuda" lands safetensors directly in VRAM and avoids the
    # ~5min nn.Module._apply Python loop over DS-Coder's 5291 weight tensors.
    # Falls back to plain CPU/MPS path if device is not cuda.
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        **dtype_kwarg(dtype),
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = "cuda"
    base = AutoModelForCausalLM.from_pretrained(str(base_path), **load_kwargs)
    base.eval()
    if device.type != "cuda":
        base.to(device)
    print(
        f"[probe] base loaded: hidden_size={base.config.hidden_size} "
        f"layers={base.config.num_hidden_layers}"
    )

    # CRITICAL: AutoTokenizer falls back to broken slow LlamaTokenizer for
    # this base. Use the trust-remote-code DeepseekTokenizerFast directly.
    # See scripts/_dscoder_compat.py and memory/project_dscoder_5x_blocker.md.
    tokenizer = load_dscoder_tokenizer(base_path)
    print(f"[probe] tokenizer class: {type(tokenizer).__name__}")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = load_prompts(args)
    # Prepend BOS so the model sees the start-of-sequence marker it was
    # trained with. DS-Coder's tokenizer has add_bos_token=True by default,
    # but we use add_special_tokens=False below (so truncation cuts only
    # real text, not BOS/EOS), so we add BOS manually here.
    bos = tokenizer.bos_token or ""
    prompts = [bos + p for p in prompts]
    print(f"[probe] loaded {len(prompts)} prompts (BOS={tokenizer.bos_token!r} prepended)")

    all_results: list[ProbeResult] = []
    for cand in args.candidates:
        print(f"\n[probe] === recurrent_layer_idx = {cand} ===")
        cfg = MythicRDTDeepseekV2Config(
            prelude_layers=args.prelude_layers,
            coda_layers=args.coda_layers,
            recurrent_layer_idx=cand,
            train_loop_iters=1,
            max_loop_iters=max(args.T_values),
            base_model_path=str(base_path),
        )
        try:
            wrapper = MythicRDTDeepseekV2ForCausalLM(cfg, base=base).to(device)
        except ValueError as exc:
            print(f"[probe] config rejected: {exc}")
            continue
        wrapper.eval()

        for T in args.T_values:
            print(f"[probe]   T={T} ...", end=" ", flush=True)
            res = probe_one(
                wrapper,
                tokenizer,
                prompts,
                layer=cand,
                T=T,
                gen_tokens=args.gen_tokens,
                truncate_tokens=args.prompt_truncate_tokens,
                batch_size=args.batch_size,
                device=device,
            )
            all_results.append(res)
            print(
                f"PPL={res.mean_ppl:.3f}  gib={res.gibberish_rate*100:.1f}%  "
                f"({res.elapsed_sec:.1f}s, n={res.n_prompts})"
            )

    print("\n[probe] === summary ===")
    print(f"{'layer':>6} {'T':>3} {'mean_PPL':>10} {'gib_rate':>10} {'sec':>8}")
    for r in all_results:
        print(
            f"{r.layer:>6} {r.T:>3} {r.mean_ppl:>10.3f} "
            f"{r.gibberish_rate*100:>9.1f}% {r.elapsed_sec:>8.1f}"
        )

    print("\n[probe] === decision (per MASTER_PLAN.md §5) ===")
    layers_passed: list[int] = []
    layers_failed: list[tuple[int, str]] = []
    for cand in args.candidates:
        rs = [r for r in all_results if r.layer == cand]
        if not rs:
            continue
        # Compare T=8 (or max T) against T=1
        r_t1 = next((r for r in rs if r.T == 1), None)
        r_max = max(rs, key=lambda r: r.T)
        if 8 not in args.T_values:
            print(
                f"  layer {cand}: skipping decision (T=8 not in --T-values)."
            )
            continue
        ratio = r_max.mean_ppl / r_t1.mean_ppl if r_t1 and r_t1.mean_ppl > 0 else float("inf")
        gib = r_max.gibberish_rate
        verdict = "PASS"
        reasons = []
        if ratio > 1.5:
            verdict = "FAIL"
            reasons.append(f"PPL_T={r_max.T}/PPL_T=1 = {ratio:.2f}x > 1.5x")
        if gib > 0.20:
            verdict = "FAIL"
            reasons.append(f"gibberish={gib*100:.1f}% > 20%")
        line = (
            f"  layer {cand}: T={r_max.T} ppl_ratio={ratio:.3f} "
            f"gibberish={gib*100:.1f}%  -> {verdict}"
        )
        if reasons:
            line += "  (" + "; ".join(reasons) + ")"
        print(line)
        if verdict == "PASS":
            layers_passed.append(cand)
        else:
            layers_failed.append((cand, "; ".join(reasons)))

    if 8 in args.T_values:
        if layers_passed:
            print(f"\n[probe] candidate layers passing the gate: {layers_passed}")
            print(
                "[probe] recommendation: pick the lowest-PPL layer from the "
                "above (re-inspect summary table)."
            )
        else:
            print(
                "\n[probe] NO layer passed. Revisit middle-layer choice + "
                "gate init before phase 1 (MASTER_PLAN.md §5 hard call)."
            )

    if args.output_json:
        out = [asdict(r) for r in all_results]
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "results": out,
                    "passed": layers_passed,
                    "failed": layers_failed,
                },
                f,
                indent=2,
            )
        print(f"\n[probe] wrote results -> {args.output_json}")

    return 0 if (8 not in args.T_values or layers_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
