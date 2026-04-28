#!/usr/bin/env python
"""HumanEval-20 smoke test — Phase 1 sanity gate (Stage 1, DS-Coder-V2-Lite).

Per MASTER_PLAN.md §5 Phase 1:
> First eval: HumanEval --limit 20 at T=1 vs base — should be near-parity
> (~80 % since at T=1 we're effectively a 3-layer slice of DS-Coder). If
> wildly off, the surgery is wrong.

This script:
  1. Loads DS-Coder-V2-Lite-Instruct on GPU (using device_map="cuda" so
     the safetensors land directly in VRAM, ~5x faster than .to("cuda")).
  2. Pulls the first --limit HumanEval problems from HF.
  3. Generates greedy completions with the BASE model (model.generate).
  4. Builds the Mythic-RDT wrapper at the chosen layer + T values, runs
     greedy_generate (no KV cache in v0) on the same problems.
  5. Scores via exec(prompt + completion + test) — the HumanEval canonical
     scorer. Counts pass@1 per configuration.
  6. Reports a comparison table: base | wrapper T=1 | wrapper T=4 | T=8.

Phase 1 gate (per MASTER_PLAN.md):
  - wrapper_T1 within ~1 pp of base   -> wrapper plumbing is correct
  - wrapper at T=4, T=8 runs without crashes -> stability OK
  - if wrapper_T1 - base > 5 pp drift, surgery has a bug

CRITICAL (parent project bug-015): NEVER eval HumanEval via chat-style
APIs that wrap completions in ```python fences. We are using direct
PyTorch generation here so this isn't a risk, but the scorer still has
to handle the case where the model emits markdown -- we strip leading
``` fences before exec.

Sandbox note: HumanEval scoring exec()s untrusted-ish code. We run with
a 5 s timeout per problem and a restricted globals dict. Anyone running
this on a multi-tenant pod should still be aware that the model could
emit `os.system("rm -rf /")` and have it executed. Acceptable for a
single-user vast.ai pod we control; do NOT run this against arbitrary
checkpoints in production.

Usage:
    conda activate mythic-rdt
    python scripts/humaneval_smoke.py \\
        --base base/DeepSeek-Coder-V2-Lite-Instruct \\
        --limit 20 --recurrent-layer-idx 10 \\
        --T-values 1 4 8 --gen-tokens 384 \\
        --output-json results/humaneval_smoke.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mythic_rdt.configuration import MythicRDTDeepseekV2Config  # noqa: E402
from mythic_rdt.modeling import MythicRDTDeepseekV2ForCausalLM  # noqa: E402


def _load_dscoder_tokenizer(base_path: str):
    """Load DS-Coder-V2-Lite-Instruct's tokenizer correctly.

    The repo declares `tokenizer_class: LlamaTokenizerFast` in
    tokenizer_config.json but does NOT register an `auto_map` entry for
    its custom DeepseekTokenizerFast. As a result, AutoTokenizer silently
    falls back to the slow LlamaTokenizer, which produces broken
    tokenization (drops spaces, drops non-ASCII chars, encode/decode
    round-trip is lossy).

    Verified failure mode (transformers 5.6 + tokenizers 0.22.2):
      Encoding "Please complete the following Python function." with the
      slow LlamaTokenizer gives tokens that decode back to
      "PleasecompletethefollowingPythonfunction." (no spaces).
      Encoding any Chinese is silently dropped.

    Fix: import DeepseekTokenizerFast (subclass of LlamaTokenizerFast)
    directly from the model's own trust_remote_code module.
    """
    sys.path.insert(0, base_path)
    try:
        from tokenization_deepseek_fast import DeepseekTokenizerFast
    finally:
        sys.path.pop(0)
    return DeepseekTokenizerFast.from_pretrained(base_path)


# ---------------------------------------------------------------------------
# HumanEval loading + scoring
# ---------------------------------------------------------------------------


def load_humaneval(limit: int) -> list[dict]:
    """Pull the canonical HumanEval (164 problems) and return the first `limit`."""
    import pyarrow.parquet as pq
    he_path = hf_hub_download(
        repo_id="openai_humaneval",
        repo_type="dataset",
        filename="openai_humaneval/test-00000-of-00001.parquet",
    )
    tbl = pq.read_table(he_path).to_pylist()
    return tbl[:limit]


# Markdown-fence handling for instruct-mode (chat-template) generations.
# DS-Coder-Instruct emits responses like:
#     ```python
#     <full function definition incl. signature + body>
#     ```
# So we extract the FIRST fenced block, then peel off the signature lines
# (everything up to and including the docstring) so it lines up with the
# canonical HumanEval scorer (which expects `prompt + completion`).
FENCED_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
LEADING_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n?", re.IGNORECASE)
TRAILING_FENCE_RE = re.compile(r"\n?```\s*$", re.IGNORECASE)


def _extract_function_body(code: str, entry_point: str) -> str:
    """Given a chunk of code that *includes* `def entry_point(...)`, return
    the body alone (everything after the signature + docstring)."""
    lines = code.splitlines(keepends=True)
    # locate `def entry_point(` line
    def_idx = None
    for i, ln in enumerate(lines):
        if re.match(rf"\s*def\s+{re.escape(entry_point)}\s*\(", ln):
            def_idx = i
            break
    if def_idx is None:
        return code  # no signature found, return as-is and hope for the best
    # body starts after the signature line; skip docstring if present
    body_start = def_idx + 1
    # crude docstring skip: if next non-blank line starts with """ or ''',
    # consume until matching close
    i = body_start
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            # single-line docstring?
            rest = stripped[3:]
            if quote in rest:
                body_start = i + 1
            else:
                # multi-line: consume until close
                j = i + 1
                while j < len(lines) and quote not in lines[j]:
                    j += 1
                body_start = j + 1
    return "".join(lines[body_start:])


def clean_completion(completion: str, prompt: str, entry_point: str = "") -> str:
    """Convert a model generation into something the HumanEval scorer can
    concatenate after `prompt`.

    Two paths:
      1. Chat-template mode: model emits ```python\\n<full def + body>\\n```.
         Extract the fenced block, drop the signature/docstring, return body.
      2. Raw-completion mode (legacy): model continues the prompt directly.
         Drop echoed prompt + strip stray fences.
    """
    # --- path 1: full fenced block present ---
    m = FENCED_BLOCK_RE.search(completion)
    if m:
        block = m.group(1)
        if entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", block):
            return _extract_function_body(block, entry_point)
        # fence present but no def -- it's already a body
        return block
    # --- path 2: raw completion (no fences) ---
    if completion.startswith(prompt):
        completion = completion[len(prompt):]
    completion = LEADING_FENCE_RE.sub("", completion)
    completion = TRAILING_FENCE_RE.sub("", completion)
    # Clip at the first new def/class/import boundary so we don't drag in
    # extra hallucinated definitions that shadow `entry_point`.
    cut_idx = len(completion)
    for marker in ("\n\nclass ", "\n\nimport ", "\n\nfrom ", "\nif __name__", "\n#"):
        i = completion.find(marker)
        if 0 <= i < cut_idx:
            cut_idx = i
    completion = completion[:cut_idx]
    return completion


def _score_worker(prompt: str, completion: str, test: str, entry_point: str, q):
    """Run inside a child process so a hung infinite loop doesn't kill us."""
    try:
        program = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
        ns: dict = {}
        exec(program, ns)
        q.put(("pass", ""))
    except Exception as exc:
        q.put(("fail", f"{type(exc).__name__}: {exc}"))


def score_problem(
    prompt: str, completion: str, test: str, entry_point: str, timeout: float = 5.0
) -> tuple[bool, str]:
    """Return (passed, reason). Runs the scoring exec in a child process."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_score_worker, args=(prompt, completion, test, entry_point, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
        return False, f"timeout>{timeout}s"
    if q.empty():
        return False, "child crashed (no result)"
    status, msg = q.get_nowait()
    return status == "pass", msg


# ---------------------------------------------------------------------------
# LiveCodeBench (function_call subset) loading + scoring
# ---------------------------------------------------------------------------

LCB_INSTRUCT_TEMPLATE = (
    "Solve the following Python coding problem. Respond with ONLY the "
    "completed Solution class in a Python markdown block, no explanations.\n\n"
    "{question}\n\n"
    "```python\n{starter}\n```"
)


def load_lcb(limit: int, difficulty: str = "medium",
             min_date: str = "2024-10-01",
             testtype: str = "functional") -> list[dict]:
    """Load LiveCodeBench problems filtered to function_call style.

    Returns a list of normalized problem dicts:
      - task_id, question_content, starter_code, method_name, difficulty,
        public_tests (list of {input, output, testtype}).

    Filtering:
      - difficulty match (default "medium" — easy is too easy, hard runs are slow)
      - contest_date >= min_date (contamination control: defaults post-2024-10
        which is after DS-Coder-V2-Lite's training cutoff)
      - testtype == "functional" (skip stdin/stdout style for smoke; full LCB
        eval at v4 end will use lcb-runner which handles stdin properly)

    Note on loading: datasets>=4.0 dropped support for trust_remote_code-based
    dataset scripts and LCB ships its data behind such a script. We bypass it
    by downloading the underlying JSONL release files directly via
    huggingface_hub. As of 2026-04 the release set is test{,2..6}.jsonl and
    contains ~1055 problems total; the smoke filter typically yields ~55
    medium / ~34 easy / ~38 hard candidates post-2024-10.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[lcb] WARN: `huggingface_hub` not installed; skipping LCB.")
        return []
    print(f"[lcb] loading livecodebench/code_generation_lite "
          f"(difficulty={difficulty}, min_date={min_date}, testtype={testtype})...")
    release_files = ["test.jsonl", "test2.jsonl", "test3.jsonl",
                     "test4.jsonl", "test5.jsonl", "test6.jsonl"]
    out: list[dict] = []
    for fn in release_files:
        try:
            path = hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                repo_type="dataset",
                filename=fn,
            )
        except Exception as exc:
            print(f"[lcb]   skip {fn}: {exc}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("difficulty") != difficulty:
                    continue
                contest_date = (row.get("contest_date") or "")[:10]
                if contest_date and contest_date < min_date:
                    continue
                public_raw = row.get("public_test_cases", "[]")
                if isinstance(public_raw, str):
                    try:
                        public = json.loads(public_raw)
                    except (json.JSONDecodeError, ValueError):
                        continue
                else:
                    public = public_raw or []
                if not public:
                    continue
                if public[0].get("testtype") != testtype:
                    continue
                starter = row.get("starter_code", "") or ""
                m = re.search(r"def\s+(\w+)\s*\(\s*self", starter)
                if not m:
                    continue
                out.append({
                    "task_id": f"lcb/{row.get('platform','?')}/{row.get('question_id','?')}",
                    "question_content": row.get("question_content", ""),
                    "starter_code": starter,
                    "method_name": m.group(1),
                    "difficulty": row.get("difficulty", ""),
                    "contest_date": contest_date,
                    "public_tests": public,
                })
                if len(out) >= limit:
                    break
        if len(out) >= limit:
            break
    print(f"[lcb] loaded {len(out)} problems "
          f"(difficulty={difficulty}, post {min_date}, functional)")
    return out


def build_lcb_chat_prompts(problems: list[dict], tokenizer) -> list[str]:
    """Render each LCB problem as a chat-template prompt."""
    rendered: list[str] = []
    for prob in problems:
        msg = [{"role": "user", "content": LCB_INSTRUCT_TEMPLATE.format(
            question=prob["question_content"],
            starter=prob["starter_code"],
        )}]
        rendered.append(tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True,
        ))
    return rendered


def clean_lcb_completion(completion: str, starter_code: str) -> str:
    """Extract the Solution class from a fenced block; fall back to raw."""
    m = FENCED_BLOCK_RE.search(completion)
    if m:
        return m.group(1)
    # No fence found; strip trailing fence if partial, return as-is.
    return TRAILING_FENCE_RE.sub("", completion)


def _score_lcb_worker(code: str, tests: list, method_name: str, q):
    """Run inside child process. Imports are inside the function so they
    survive the fork in subprocesses without re-importing module globals."""
    import ast as _ast
    try:
        # Pre-populate namespace with common typing + stdlib symbols so
        # starter_code with `List[int]`, `Optional[str]`, etc. exec()s without
        # NameError (LCB starter signatures use these heavily).
        preamble = (
            "from typing import List, Dict, Tuple, Set, Optional, Union, "
            "Any, Callable, Iterator, Iterable, Sequence\n"
            "from collections import defaultdict, deque, Counter, OrderedDict\n"
            "from math import inf, gcd, floor, ceil, sqrt, log, log2, factorial\n"
            "from heapq import heappush, heappop, heapify, nlargest, nsmallest\n"
            "from bisect import bisect_left, bisect_right, insort\n"
            "from itertools import accumulate, combinations, permutations, product\n"
            "from functools import lru_cache, cache, reduce\n"
        )
        ns: dict = {}
        exec(preamble + code, ns)
        Solution = ns.get("Solution")
        if Solution is None:
            q.put(("fail", "no Solution class defined"))
            return
        for i, t in enumerate(tests):
            inp_str = (t.get("input") or "").strip()
            exp_str = (t.get("output") or "").strip()
            # LCB encodes input as Python literals. Two formats observed:
            #   - Single-line: the WHOLE string is one arg literal (e.g.
            #     "[1,2,3]" means one List[int] arg, NOT three int args).
            #   - Multi-line: each line is one positional arg literal.
            if "\n" in inp_str:
                args = []
                for line in inp_str.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        args.append(_ast.literal_eval(line))
                    except (ValueError, SyntaxError):
                        args.append(line)
                args = tuple(args)
            else:
                try:
                    args = (_ast.literal_eval(inp_str),)
                except (ValueError, SyntaxError):
                    args = (inp_str,)
            try:
                expected = _ast.literal_eval(exp_str)
            except (ValueError, SyntaxError):
                expected = exp_str
            sol = Solution()
            method = getattr(sol, method_name, None)
            if method is None:
                q.put(("fail", f"Solution has no method `{method_name}`"))
                return
            result = method(*args)
            if result != expected:
                q.put(("fail",
                       f"test {i}: got {result!r} expected {expected!r}"))
                return
        q.put(("pass", ""))
    except Exception as exc:
        q.put(("fail", f"{type(exc).__name__}: {exc}"))


def score_lcb_problem(code: str, tests: list, method_name: str,
                      timeout: float = 10.0) -> tuple[bool, str]:
    """Sandbox the LCB scoring exec in a child process with a timeout."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_score_lcb_worker, args=(code, tests, method_name, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
        return False, f"timeout>{timeout}s"
    if q.empty():
        return False, "child crashed (no result)"
    status, msg = q.get_nowait()
    return status == "pass", msg


def run_eval_lcb(name: str, completions: list[str],
                 problems: list[dict]) -> "RunResult":
    t0 = time.time()
    n_pass = 0
    failures: list[dict] = []
    for prob, comp in zip(problems, completions):
        cleaned = clean_lcb_completion(comp, prob["starter_code"])
        passed, reason = score_lcb_problem(
            cleaned, prob["public_tests"], prob["method_name"],
        )
        if passed:
            n_pass += 1
        else:
            if len(failures) < 5:
                failures.append({
                    "task_id": prob["task_id"],
                    "method": prob["method_name"],
                    "difficulty": prob["difficulty"],
                    "reason": reason[:160],
                    "completion_head": cleaned[:160],
                })
    elapsed = time.time() - t0
    return RunResult(
        name=name,
        pass_at_1=n_pass / max(1, len(problems)),
        n_pass=n_pass,
        n_total=len(problems),
        elapsed_sec=elapsed,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _prepend_bos(prompts: list[str], tokenizer) -> list[str]:
    """DS-Coder's tokenizer has add_bos_token=False, so add_special_tokens=True
    is a no-op. Manually prefix the BOS string so the model sees the start-of-
    sequence marker it was trained with -- without it generation degenerates to
    high-vocab gibberish (multilingual BPE tokens).
    """
    bos = tokenizer.bos_token or ""
    return [bos + p for p in prompts]


# DeepSeek-Coder-Instruct HumanEval prompting recipe:
# Wrap the raw HumanEval prompt as a user instruction. The official
# DeepSeek-Coder evaluation harness uses a similar template; the key idea
# is to ask the model to RETURN the completed function (in a fenced block),
# rather than autoregress-continue the function signature.
HE_INSTRUCT_TEMPLATE = (
    "Please complete the following Python function. "
    "Respond with ONLY the completed code in a Python markdown block, "
    "no explanations.\n\n"
    "```python\n{prompt}```"
)


def build_chat_prompts(prompts: list[str], tokenizer) -> list[str]:
    """Render each HumanEval prompt through the tokenizer's chat template.

    The chat template handles BOS, role markers, and the assistant-turn
    prefix. We do NOT additionally prepend BOS here -- the template already
    includes it (verified on DS-Coder-V2-Lite-Instruct).
    """
    rendered: list[str] = []
    for p in prompts:
        msg = [{"role": "user", "content": HE_INSTRUCT_TEMPLATE.format(prompt=p)}]
        rendered.append(
            tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
        )
    return rendered


@torch.no_grad()
def base_generate(
    base, tokenizer, prompts: list[str], gen_tokens: int, device: torch.device,
    batch_size: int = 1,
) -> list[str]:
    """Greedy generation through the base model using model.generate (KV-cached).

    `prompts` MUST already be chat-template-rendered (with BOS baked in) --
    do not pass raw HumanEval signatures here.
    """
    completions: list[str] = []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=2048, add_special_tokens=False,  # template already has BOS
            )
            input_ids = enc.input_ids.to(device)
            attn = enc.attention_mask.to(device)
            # KV cache works correctly under transformers 4.46. Under 5.x
            # the entire forward path is broken (not just cache), so this
            # script must be run under the transformers-4.x sidecar venv.
            out = base.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=gen_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            for r, p_text in enumerate(batch):
                gen_only = out[r, input_ids.shape[1]:].tolist()
                completions.append(tokenizer.decode(gen_only, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_side
    return completions


@torch.no_grad()
def wrapper_generate(
    wrapper: MythicRDTDeepseekV2ForCausalLM,
    tokenizer,
    prompts: list[str],
    T: int,
    gen_tokens: int,
    device: torch.device,
    batch_size: int = 1,
    force_bypass: bool = False,
    no_kv_cache: bool = False,
) -> list[str]:
    """Greedy generation through the wrapper.

    Uses KV cache when T==1 (huge speedup vs naive recompute -- a 200-token
    completion goes from ~50,000 token-forwards to ~200). For T>1 the
    wrapper currently doesn't support KV cache (would need T separate cache
    slots per recurrent block layer); falls back to recompute-per-token.
    """
    use_cache = not no_kv_cache
    completions: list[str] = []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    eos_id = tokenizer.eos_token_id
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=2048, add_special_tokens=False,  # template already has BOS
            )
            input_ids = enc.input_ids.to(device)
            attn = enc.attention_mask.to(device)
            done = torch.zeros(input_ids.size(0), dtype=torch.bool, device=device)
            generated = torch.empty((input_ids.size(0), 0),
                                    dtype=torch.long, device=device)

            if use_cache:
                # ----- T=1 KV-cache path -----
                # Initial pass: full prefix in, get logits + cache.
                out = wrapper(
                    input_ids, attention_mask=attn, T=T,
                    force_bypass=force_bypass,
                    use_cache=True, return_dict=True,
                )
                logits = out.logits
                past = out.past_key_values
                cur_mask = attn
                next_id = logits[:, -1, :].argmax(dim=-1)
                if eos_id is not None:
                    done = done | (next_id == eos_id)
                generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
                cur_mask = torch.cat(
                    [cur_mask, (~done).long().unsqueeze(1)], dim=1
                )
                # Incremental: feed only the new token each step.
                for _ in range(gen_tokens - 1):
                    if done.all():
                        break
                    out = wrapper(
                        next_id.unsqueeze(1), attention_mask=cur_mask, T=T,
                        force_bypass=force_bypass,
                        past_key_values=past, use_cache=True, return_dict=True,
                    )
                    past = out.past_key_values
                    next_id = out.logits[:, -1, :].argmax(dim=-1)
                    next_id = torch.where(done, torch.full_like(next_id, pad_id), next_id)
                    if eos_id is not None:
                        done = done | (next_id == eos_id)
                    generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
                    cur_mask = torch.cat(
                        [cur_mask, (~done).long().unsqueeze(1)], dim=1
                    )
            else:
                # ----- T>1 recompute path (legacy) -----
                cur = input_ids
                cur_mask = attn
                for _ in range(gen_tokens):
                    logits = wrapper(
                        cur, attention_mask=cur_mask, T=T, force_bypass=force_bypass
                    )
                    if not torch.is_tensor(logits):
                        logits = logits.logits
                    next_id = logits[:, -1, :].argmax(dim=-1)
                    next_id = torch.where(done, torch.full_like(next_id, pad_id), next_id)
                    if eos_id is not None:
                        done = done | (next_id == eos_id)
                    cur = torch.cat([cur, next_id.unsqueeze(1)], dim=1)
                    cur_mask = torch.cat(
                        [cur_mask, (~done).long().unsqueeze(1)], dim=1
                    )
                    if done.all():
                        break
                generated = cur[:, input_ids.shape[1]:]
            for r in range(len(batch)):
                gen_only = generated[r].tolist()
                completions.append(tokenizer.decode(gen_only, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_side
    return completions


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    name: str
    pass_at_1: float
    n_pass: int
    n_total: int
    elapsed_sec: float
    failures: list[dict]


def run_eval(
    name: str,
    completions: list[str],
    problems: list[dict],
) -> RunResult:
    t0 = time.time()
    n_pass = 0
    failures: list[dict] = []
    for prob, comp in zip(problems, completions):
        cleaned = clean_completion(comp, prob["prompt"], prob["entry_point"])
        passed, reason = score_problem(
            prob["prompt"], cleaned, prob["test"], prob["entry_point"]
        )
        if passed:
            n_pass += 1
        else:
            if len(failures) < 5:
                failures.append({
                    "task_id": prob.get("task_id"),
                    "entry_point": prob["entry_point"],
                    "reason": reason[:120],
                    "completion_head": cleaned[:120],
                })
    elapsed = time.time() - t0
    return RunResult(
        name=name,
        pass_at_1=n_pass / max(1, len(problems)),
        n_pass=n_pass,
        n_total=len(problems),
        elapsed_sec=elapsed,
        failures=failures,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HumanEval-N smoke test for Mythic-RDT")
    p.add_argument("--base", type=str, default="base/DeepSeek-Coder-V2-Lite-Instruct")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--recurrent-layer-idx", type=int, default=10)
    p.add_argument("--recurrent-block-start", type=int, default=None,
                   help="v3+ block mode: start index (inclusive) of the "
                        "consecutive recurrent block.")
    p.add_argument("--recurrent-block-end", type=int, default=None,
                   help="v3+ block mode: end index (inclusive).")
    p.add_argument("--block-mode", action="store_true",
                   help="Use v3 recurrence formula (block_out passes "
                        "through; LTI is purely additive).")
    p.add_argument("--prelude-layers", type=int, default=1)
    p.add_argument("--coda-layers", type=int, default=1)
    p.add_argument("--gate-init-bias", type=float, default=0.0,
                   help="Match training-time gate init for the wrapper "
                        "build. v3 default 0.0; v0-v2 used -3.0.")
    p.add_argument("--layerscale-init", type=float, default=1e-4)
    p.add_argument("--layerscale-clamp-max", type=float, default=None)
    p.add_argument("--force-bypass", action="store_true",
                   help="Phase 0 sanity: zero gate AND layerscale at every "
                        "iteration so block_out flows through unmodified. "
                        "With T=1 and no LoRA, output should be bit-exact "
                        "with running prelude+block+coda once. Used to "
                        "verify wrapper plumbing on a fresh build.")
    p.add_argument("--T-values", type=int, nargs="+", default=[1])
    p.add_argument("--gen-tokens", type=int, default=384)
    p.add_argument("--no-kv-cache", action="store_true",
                   help="Disable wrapper KV cache (recompute full prefix each "
                        "step). Diagnostic only -- ~10x slower. Use to isolate "
                        "whether long-generation regressions come from a cache "
                        "bug vs from wrapper logit drift.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--skip-base", action="store_true",
                   help="Skip the base-model run (only useful for re-runs).")
    # Checkpoint-loading: required to eval a Phase 1 fine-tune. When omitted
    # the wrapper runs at init weights (LoRA-B=0, gate=-3) which is functionally
    # bypass-mode -- only useful for plumbing-correctness sanity, not quality.
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a Trainer checkpoint dir (containing "
                        "mythic_rdt_trainable.pt). Loads trained LoRA + "
                        "recurrence params on top of the freshly-built wrapper.")
    p.add_argument("--lora-rank", type=int, default=8,
                   help="Must match the rank used at training time (the "
                        "checkpoint's tensors expect this shape).")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", type=str, nargs="+",
                   default=["self_attn.q_proj_or_q_a", "self_attn.o_proj"],
                   help="Must match the targets used at training time.")
    p.add_argument("--quant", type=str, default="none",
                   choices=["none", "nf4", "fp4"],
                   help="Match the training-time quant for VRAM-tight eval. "
                        "Pure bf16 base + trained adapters works too if it fits.")
    # LiveCodeBench mini (function_call subset) — optional second eval
    # alongside HumanEval-N. Tighter contamination control + harder problems
    # with real headroom for measuring T>1 value-add.
    p.add_argument("--lcb-limit", type=int, default=0,
                   help="If >0, also evaluate this many LCB problems "
                        "(function_call subset, post-cutoff date) on base + each "
                        "wrapper T. 0 disables. Reasonable smoke value: 10.")
    p.add_argument("--lcb-difficulty", type=str, default="medium",
                   choices=["easy", "medium", "hard"],
                   help="LCB difficulty filter (smoke default: medium).")
    p.add_argument("--lcb-min-date", type=str, default="2024-10-01",
                   help="Skip LCB problems with contest_date earlier than this "
                        "(ISO YYYY-MM-DD). Defaults to past DS-Coder-V2 cutoff.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    base_path = Path(args.base)

    print(f"[smoke] base={base_path}  device={device}  dtype={dtype}")
    print(f"[smoke] limit={args.limit}  layer={args.recurrent_layer_idx}  T-values={args.T_values}")
    print(f"[smoke] gen_tokens={args.gen_tokens}  batch_size={args.batch_size}")

    print("[smoke] loading HumanEval...")
    problems = load_humaneval(args.limit)
    print(f"[smoke] loaded {len(problems)} problems")
    raw_prompts = [p["prompt"] for p in problems]

    print("[smoke] loading base (device_map='cuda', skips slow .to(cuda))...")
    import transformers as _tf
    _tf_major = int(_tf.__version__.split(".")[0])
    _dtype_kw = {"dtype": dtype} if _tf_major >= 5 else {"torch_dtype": dtype}
    print(f"[smoke] transformers={_tf.__version__} -> using {list(_dtype_kw)[0]}=...")
    _load_kwargs = dict(
        trust_remote_code=True,
        device_map="cuda",
        low_cpu_mem_usage=True,
        **_dtype_kw,
    )
    if args.quant != "none":
        from transformers import BitsAndBytesConfig
        _load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.quant,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[smoke] base quantized to {args.quant.upper()} (compute={dtype})")
    base = AutoModelForCausalLM.from_pretrained(str(base_path), **_load_kwargs)
    base.eval()
    tokenizer = _load_dscoder_tokenizer(str(base_path))
    print(f"[smoke] tokenizer class: {type(tokenizer).__name__}")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[smoke] base loaded: hidden_size={base.config.hidden_size} layers={base.config.num_hidden_layers}")

    print("[smoke] rendering prompts via chat template...")
    chat_prompts = build_chat_prompts(raw_prompts, tokenizer)
    if chat_prompts:
        print(f"[smoke]   sample chat prompt[0] head:\n--\n{chat_prompts[0][:300]}\n--")

    # Optional: LCB function_call subset alongside HumanEval.
    lcb_problems: list[dict] = []
    lcb_chat_prompts: list[str] = []
    if args.lcb_limit > 0:
        lcb_problems = load_lcb(
            limit=args.lcb_limit,
            difficulty=args.lcb_difficulty,
            min_date=args.lcb_min_date,
        )
        if lcb_problems:
            lcb_chat_prompts = build_lcb_chat_prompts(lcb_problems, tokenizer)
            print(f"[lcb]   sample LCB prompt[0] head:\n--\n{lcb_chat_prompts[0][:300]}\n--")

    results: list[RunResult] = []

    if not args.skip_base:
        if chat_prompts:
            print("[smoke] generating HumanEval with BASE...")
            t0 = time.time()
            base_completions = base_generate(
                base, tokenizer, chat_prompts, args.gen_tokens, device, args.batch_size
            )
            print(f"[smoke]   base HE generation: {time.time()-t0:.1f}s")
            r = run_eval("base_he", base_completions, problems)
            print(f"[smoke]   BASE HE pass@1 = {r.pass_at_1*100:.1f}%  ({r.n_pass}/{r.n_total})")
            results.append(r)
        if lcb_problems:
            print("[smoke] generating LCB with BASE...")
            t0 = time.time()
            base_lcb_completions = base_generate(
                base, tokenizer, lcb_chat_prompts, args.gen_tokens, device,
                args.batch_size,
            )
            print(f"[smoke]   base LCB generation: {time.time()-t0:.1f}s")
            r = run_eval_lcb("base_lcb", base_lcb_completions, lcb_problems)
            print(f"[smoke]   BASE LCB pass@1 = {r.pass_at_1*100:.1f}%  ({r.n_pass}/{r.n_total})")
            results.append(r)

    cfg = MythicRDTDeepseekV2Config(
        prelude_layers=args.prelude_layers,
        coda_layers=args.coda_layers,
        recurrent_layer_idx=args.recurrent_layer_idx,
        recurrent_block_start=args.recurrent_block_start,
        recurrent_block_end=args.recurrent_block_end,
        block_mode=args.block_mode,
        gate_init_bias=args.gate_init_bias,
        layerscale_init=args.layerscale_init,
        layerscale_clamp_max=args.layerscale_clamp_max,
        train_loop_iters=1,
        max_loop_iters=max(args.T_values),
        base_model_path=str(base_path),
    )
    print(f"[smoke] building wrapper (layer={cfg.recurrent_layer_idx})...")
    wrapper = MythicRDTDeepseekV2ForCausalLM(cfg, base=base).to(device)
    wrapper.eval()

    # Optional: load a Phase 1 fine-tune checkpoint. Without this the wrapper
    # runs with its init weights -- LoRA-B=0 + gate.bias=-3 -- which is the
    # plumbing-correctness mode, not the trained model. The checkpoint stores
    # ONLY trainable params, so we need to (a) rebuild the LoRA scaffolding
    # at the same targets/rank as training, then (b) load_state_dict. The
    # frozen base weights come from --base, same path as training.
    if args.checkpoint:
        from pathlib import Path as _Path
        from mythic_rdt.training import inject_depth_lora
        from mythic_rdt.training.trainer import (
            TRAINABLE_STATE_FN, _load_trainable_state,
        )
        ckpt_dir = _Path(args.checkpoint)
        # If user pointed at the parent run dir, find the latest checkpoint-N.
        if not (ckpt_dir / TRAINABLE_STATE_FN).exists():
            subs = sorted(ckpt_dir.glob("checkpoint-*"),
                          key=lambda p: int(p.name.split("-")[1]))
            if not subs:
                raise FileNotFoundError(
                    f"--checkpoint {ckpt_dir} has neither {TRAINABLE_STATE_FN} "
                    f"nor any checkpoint-N subdir."
                )
            ckpt_dir = subs[-1]
            print(f"[smoke] resolved latest sub-checkpoint: {ckpt_dir.name}")
        print(f"[smoke] injecting DepthLoRA (targets={args.lora_targets} rank={args.lora_rank}) ...")
        records = inject_depth_lora(
            wrapper,
            targets=args.lora_targets,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            lora_dtype=dtype,
        )
        for r in records:
            print(f"[smoke]   lora wired: {r.qualified_name} rank={r.rank} T={r.n_iters}")
        state = torch.load(
            ckpt_dir / TRAINABLE_STATE_FN, map_location="cpu", weights_only=True,
        )
        loaded, missing, unexpected = _load_trainable_state(wrapper, state)
        print(f"[smoke] loaded {loaded} trainable tensors  "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[smoke]   missing (first 3): {missing[:3]}")
        if unexpected:
            print(f"[smoke]   unexpected (first 3): {unexpected[:3]}")
        wrapper.eval()

    for T in args.T_values:
        if chat_prompts:
            print(f"[smoke] generating HumanEval with WRAPPER T={T}...")
            t0 = time.time()
            wrapper_completions = wrapper_generate(
                wrapper, tokenizer, chat_prompts, T, args.gen_tokens, device,
                args.batch_size, force_bypass=args.force_bypass,
                no_kv_cache=args.no_kv_cache,
            )
            print(f"[smoke]   wrapper T={T} HE generation: {time.time()-t0:.1f}s")
            r = run_eval(f"wrapper_T{T}_he", wrapper_completions, problems)
            print(f"[smoke]   WRAPPER T={T} HE pass@1 = {r.pass_at_1*100:.1f}%  "
                  f"({r.n_pass}/{r.n_total})")
            results.append(r)
        if lcb_problems:
            print(f"[smoke] generating LCB with WRAPPER T={T}...")
            t0 = time.time()
            wrapper_lcb_completions = wrapper_generate(
                wrapper, tokenizer, lcb_chat_prompts, T, args.gen_tokens, device,
                args.batch_size, force_bypass=args.force_bypass,
                no_kv_cache=args.no_kv_cache,
            )
            print(f"[smoke]   wrapper T={T} LCB generation: {time.time()-t0:.1f}s")
            r = run_eval_lcb(f"wrapper_T{T}_lcb", wrapper_lcb_completions, lcb_problems)
            print(f"[smoke]   WRAPPER T={T} LCB pass@1 = {r.pass_at_1*100:.1f}%  "
                  f"({r.n_pass}/{r.n_total})")
            results.append(r)

    print("\n[smoke] === summary ===")
    print(f"{'name':<14} {'pass@1':>8} {'n_pass/n_total':>16} {'sec':>8}")
    for r in results:
        print(f"{r.name:<14} {r.pass_at_1*100:>7.1f}%  {r.n_pass:>5}/{r.n_total:<7} {r.elapsed_sec:>8.1f}")

    print("\n[smoke] === Phase 1 gate (per MASTER_PLAN.md §5) ===")
    # Accept either old naming ("base"/"wrapper_T1") or new ("base_he"/"wrapper_T1_he")
    # so historical JSON loads still work.
    def _by(*names):
        for n in names:
            for r in results:
                if r.name == n:
                    return r
        return None
    base_r = _by("base_he", "base")
    w_t1 = _by("wrapper_T1_he", "wrapper_T1")
    if base_r and w_t1:
        gap = (w_t1.pass_at_1 - base_r.pass_at_1) * 100
        verdict = "PASS" if abs(gap) <= 5 else "FAIL"
        print(f"  HumanEval wrapper T=1 vs base: gap = {gap:+.1f} pp  -> {verdict} (|gap| {'<=' if verdict=='PASS' else '>'} 5 pp)")
    else:
        print("  (need both base_he and wrapper_T1_he to render gate)")
    base_lcb = _by("base_lcb")
    if base_lcb:
        # T>1 value-add tracking on the harder LCB subset.
        for T in args.T_values:
            wT = _by(f"wrapper_T{T}_lcb")
            if wT:
                gap = (wT.pass_at_1 - base_lcb.pass_at_1) * 100
                print(f"  LCB wrapper T={T} vs base: gap = {gap:+.1f} pp  "
                      f"({wT.n_pass}/{wT.n_total} vs {base_lcb.n_pass}/{base_lcb.n_total})")

    if args.output_json:
        out = [asdict(r) for r in results]
        with open(args.output_json, "w") as f:
            json.dump({"args": vars(args), "results": out}, f, indent=2)
        print(f"\n[smoke] wrote -> {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
