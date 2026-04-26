---
name: eval-gemma4
description: Runs lm_eval against a Gemma 4 / Mythic-Gemma4 GGUF served by llama-server with all mandatory flags (--use_cache, --log_samples, tokenizer pinned to original 128e dir, --reasoning-format deepseek --reasoning-budget 8192). Use when user says 'eval', 'run GPQA', 'benchmark', 'lm_eval', 'check score', 'evaluate model'. Patterns after ../scripts/eval_gpqa_v3.sh. Always sanity-checks samples_*.jsonl post-run for empty / markdown-fence / <5-char junk. Do NOT use without --use_cache, do NOT use for training, fine-tuning, or quantization tasks.
paths:
  - scripts/eval_*.sh
  - scripts/pod_*_eval*.sh
---
# eval-gemma4

Run lm_eval against a Gemma 4 / Mythic-Gemma4 GGUF served by llama-server, with every mandatory safety flag from the parent project's hard-won rules.

## Critical

These rules are non-negotiable. Skipping any of them has cost the project hours of compute in the past.

1. **NEVER omit `--use_cache <path>`.** Without it, any crash (PEG parser, OOM, llama-server segfault, network blip) restarts the eval from question 0. Convention: `--use_cache <workdir>/<bench>_cache/<model_name>` — one prefix per model.
2. **NEVER omit `--log_samples`.** The post-run sanity check needs `samples_<task>_*.jsonl` to inspect. A pass@1 of 0.0 may mean the scorer crashed at `exec()` because of markdown fences, not that the model is bad.
3. **NEVER omit `--reasoning-format deepseek --reasoning-budget 8192`** from `llama-server`. Without `--reasoning-budget`, Gemma 4 emits malformed channel tokens (`<|channel>thought` missing closing) and crashes lm_eval at the first chemistry-heavy question. Wasted 1+ hour learning this.
4. **Tokenizer MUST be the original 128e dir** (`gemma-4-26B-A4B-it`), NOT a pruned variant or the model under test. If the project uses a 98e/109e base, point `tokenizer=` at `../google/gemma-4-26B-A4B-it`.
5. **Use the `lightseek` conda env** (`conda activate lightseek`). It has transformers 5.5.0 / Gemma 4 tokenizer support. NEVER use the `vllm` env.
6. **NEVER write eval outputs, caches, or logs to `/tmp`.** It's tmpfs (64 GB RAM). Outputs → project folder or `../google/`.
7. **Background `llama-server` correctly.** Use `disown` so it survives parent-shell exit. NEVER pipe its stdout to `head` or `tail` — they SIGPIPE the server after a few lines.
8. **Always sanity-check sample files after the run.** Count: total / empty / markdown-fence (` ``` `) / <5-char junk. If any anomaly, STOP and investigate before reporting the score.

## Instructions

### Step 1 — Confirm scope and pick a benchmark

Ask the user (max one round) if not stated:
- Which benchmark? Default = `gpqa_diamond_cot_zeroshot` (198 questions, ~6–10 h on 3090).
- Quick test or full run? Quick = `--limit 10` or `--limit 20`.
- Which GGUF / model name? It becomes the cache prefix and results folder.

Reference score to beat: Gemma 4 26B-A4B-it on GPQA Diamond = **75.25%**. Mythic-Gemma4 at T=1 must be within 2% of base.

Verify before proceeding: you have an absolute path to a quantized GGUF file, a model name (kebab-case), and a benchmark task name.

### Step 2 — Set up workdir paths

Use this layout (matches `../scripts/eval_gpqa_v3.sh`):

```bash
BASE=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
GGUF=<absolute path to GGUF file>
NAME=<model-name-kebab>          # e.g. mythic-gemma4-T8-Q4_K_M
BENCH=gpqa_diamond                # short tag, used for cache + results subdir
WORKDIR=$BASE/Mythic-Gemma4/eval_runs/$NAME
CACHE_DIR=$WORKDIR/${BENCH}_cache
RESULTS=$WORKDIR/results
mkdir -p "$CACHE_DIR" "$RESULTS"
```

Verify before proceeding: `$GGUF` exists (`ls -lh "$GGUF"`), `$WORKDIR` is on persistent disk (NOT `/tmp`).

### Step 3 — Start `llama-server` in the background

Exact invocation — every flag is mandatory:

```bash
nohup /opt/llama.cpp/build/bin/llama-server \
    -m "$GGUF" \
    --port 8099 \
    -c 32768 \
    -t 12 \
    -ngl 99 \
    --no-warmup \
    --reasoning-format deepseek \
    --reasoning-budget 8192 \
    > "$WORKDIR/llama-server.log" 2>&1 &
disown
LLAMA_PID=$!
echo "llama-server PID=$LLAMA_PID, log=$WORKDIR/llama-server.log"
```

Do NOT pipe stdout to `head`/`tail`. Do NOT forget `disown`. Do NOT change ports without checking nothing else is on 8099.

Wait for readiness (poll, don't sleep blindly):

```bash
until curl -sf http://localhost:8099/health >/dev/null; do sleep 2; done
echo "llama-server ready"
```

Verify before proceeding: `curl -s http://localhost:8099/v1/models` returns JSON, server log shows `model loaded` with no `error` lines.

### Step 4 — Activate the env and run lm_eval

```bash
conda activate lightseek

lm_eval \
    --model local-chat-completions \
    --model_args "model=$NAME,base_url=http://localhost:8099/v1/chat/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=$BASE/google/gemma-4-26B-A4B-it,max_gen_toks=16384" \
    --tasks gpqa_diamond_cot_zeroshot \
    --apply_chat_template \
    --batch_size 1 \
    --use_cache "$CACHE_DIR/${NAME}" \
    --log_samples \
    --output_path "$RESULTS/${BENCH}_${NAME}" \
    2>&1 | tee "$WORKDIR/lm_eval.log"
```

Notes:
- `tokenizer=` MUST be the original 128e dir, never a pruned/quantized variant.
- For chat-mode reasoning models on HumanEval, switch to `--model local-completions` and point `base_url` at the raw `/v1/completions` endpoint instead. Chat mode wraps answers in markdown fences and `pass@1` reports 0.0 (bug-015).
- For long runs (>30 min), check progress every 10 minutes: `tail -n 5 "$WORKDIR/lm_eval.log"` — confirm the request counter is moving and partial samples look real.

Verify before proceeding: command exited 0, results folder contains `results_*.json` and `samples_*.jsonl`.

### Step 5 — Mandatory sanity check on sample files

Never trust the headline score until samples are inspected. Run:

```bash
SAMPLES=$(ls "$RESULTS/${BENCH}_${NAME}"/samples_*.jsonl 2>/dev/null | head -1)
echo "Samples file: $SAMPLES"

python - <<'PY'
import json, os, sys
p = os.environ['SAMPLES']
total = empty = fenced = junk = 0
for line in open(p):
    s = json.loads(line)
    g = (s.get('resps') or s.get('filtered_resps') or [['']])[0]
    g = g[0] if isinstance(g, list) else g
    total += 1
    if not g or not g.strip(): empty += 1
    if '```' in (g or ''): fenced += 1
    if g and len(g.strip()) < 5: junk += 1
print(f'total={total} empty={empty} fenced={fenced} junk_lt5={junk}')
PY
```

Then read the score from `results_*.json`:

```bash
jq '.results' "$RESULTS/${BENCH}_${NAME}"/results_*.json
```

Report BOTH numbers side-by-side: `pass@1 = X` AND `total=N empty=E fenced=F junk=J`. If `empty`, `fenced`, or `junk` are non-zero, STOP. The score is suspect — investigate before reporting.

### Step 6 — Tear down `llama-server`

```bash
kill "$LLAMA_PID" 2>/dev/null
wait "$LLAMA_PID" 2>/dev/null
```

Do NOT `kill -9` first — let it flush. Confirm port 8099 is free: `ss -tlnp | grep 8099` should return nothing.

### Step 7 — Update OpenWolf memory

Append a one-liner to `../.wolf/memory.md`:
```
| HH:MM | eval $NAME on $BENCH | $RESULTS/... | pass@1=X total=N empty=E fenced=F | ~tokens |
```
If anything anomalous turned up (empty samples, fenced outputs, server crash), log it to `../.wolf/buglog.json` per the OpenWolf protocol.

## Examples

### Example 1 — Quick GPQA smoke test on a new quant

**User:** "Run a 20-question GPQA on the new Q4_K_M I just built."

**Actions:**
1. Set `NAME=mythic-T8-Q4_K_M`, `BENCH=gpqa_diamond`, build paths under `eval_runs/mythic-T8-Q4_K_M/`.
2. Start `llama-server` with the mandatory `--reasoning-format deepseek --reasoning-budget 8192`, `disown`, poll `/health`.
3. Run `lm_eval ... --tasks gpqa_diamond_cot_zeroshot --limit 20 --use_cache .../gpqa_diamond_cache/mythic-T8-Q4_K_M --log_samples --output_path .../results/...`.
4. Inspect sample files: `total=20 empty=0 fenced=0 junk_lt5=0` → score is trustworthy.
5. Report: `pass@1 = 0.70 (14/20)` plus the sample stats.
6. Kill `llama-server`, log to `../.wolf/memory.md`.

**Result:** Resumable cache lives at `.../gpqa_diamond_cache/mythic-T8-Q4_K_M`. Re-running with `--limit 198` will skip the 20 already done.

### Example 2 — Full GPQA, server crashes mid-run

**User:** "Run full GPQA on mythic-T16."

**Actions:** Same setup; mid-run llama-server segfaults at q137. Because `--use_cache` was passed, restart `llama-server`, re-run identical `lm_eval` command. Cache replays q0–q136 from SQLite; eval resumes at q137. No lost work.

## Common Issues

- **`pass@1 = 0.0` but the model seems fine** → Open the sample file and check `fenced` count. If non-zero, the scorer is failing at `exec(prompt+generation)` because of ` ``` ` markdown fences. Switch to `--model local-completions` and change the `base_url` from the chat endpoint to the raw text-completion endpoint on the same port. bug-015.

- **`llama-server` crashes around q5–q15 (chemistry)** → `--reasoning-budget` was missing. Stop, restart with `--reasoning-format deepseek --reasoning-budget 8192`, re-run lm_eval; cache resumes.

- **`Address already in use` on port 8099** → A prior `llama-server` is still running. `ss -tlnp | grep 8099` to find PID, `kill <pid>`, retry. Do NOT change port without updating `base_url=` to match.

- **`tokenizer not found` / `Gemma4Tokenizer unknown`** → Wrong env. `conda activate lightseek` (transformers 5.5.0). The `vllm` env has older transformers and will fail.

- **Eval restarts from q0 after a crash** → `--use_cache` was missing or pointed at a different path between runs. Always use `$CACHE_DIR/$NAME` exactly. The cache is a SQLite file under that prefix.

- **`results_*.json` written but `samples_*.jsonl` missing** → `--log_samples` was forgotten. Re-run is required — score cannot be trusted without sample inspection.

- **Server log mentions `expert tensors fall back to F16`** → GGUF was built with `intermediate_size` not divisible by 32 for Q4_K/Q8_0. Eval still works, file is just larger than expected. Note in the report; don't block the eval.

- **Tokenizer mismatch warnings (vocab size)** → `tokenizer=` is pointed at a pruned variant. Repoint to `$BASE/google/gemma-4-26B-A4B-it` (original 128e). Pruned variants have the same vocab but the harness sometimes flags shape mismatches that affect formatting.

- **`/tmp` filling up during long eval** → Cache or output path leaked into `/tmp`. STOP immediately (risk of system collapse — tmpfs is RAM-backed). Move outputs to `$WORKDIR` on persistent disk and re-run.
