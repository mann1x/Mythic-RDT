---
paths:
  - scripts/eval_*.sh
  - experiments/**/eval*.sh
---

# lm_eval mandatory rules (every invocation, no exceptions)

Reference working script: `../scripts/eval_gpqa_v3.sh` in parent project.

## Required flags

- `--use_cache <workdir>/<bench>_cache/<model_name>` — SQLite cache makes runs resumable. Without it, any death (PEG parser, OOM, HTTP 500, llama-server crash, network blip) restarts from 0.
- `--log_samples` — required so post-run sanity check has data to inspect.
- `--apply_chat_template` for chat-mode models (Gemma 4 reasoning).
- `--batch_size 1` for llama-server backend (`num_concurrent=1` in model_args).

## llama-server args (Gemma 4)

```bash
/opt/llama.cpp/build/bin/llama-server -m <gguf> --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
    --reasoning-format deepseek --reasoning-budget 8192
```

`--reasoning-budget 8192` is **mandatory**. Without it, malformed `<|channel>thought` tokens crash lm_eval mid-run on first chemistry question. Use `disown` on background; never pipe through `head`/`tail` (SIGPIPE).

## Tokenizer

Must be the **original 128e model dir** (`gemma-4-26B-A4B-it`), NOT the pruned variant. Set via `tokenizer=...` in `--model_args`.

## Post-run sanity

After run, inspect `samples_<task>_*.jsonl`:
- Count: total, empty, markdown-fence (` ``` `), <5-char junk.
- If `pass@1 == 0.0`, suspect scorer crash (HumanEval+chat mode wraps in fences). Print counts next to score. STOP if anomaly.

## Long runs (>30 min)

Verify request counter advances every 10 min; spot-check partial samples look real.
