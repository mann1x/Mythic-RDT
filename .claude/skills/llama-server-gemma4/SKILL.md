---
name: llama-server-gemma4
description: Launches llama-server for a Gemma 4 GGUF with the mandatory eval flags (--reasoning-format deepseek --reasoning-budget 8192, port 8099, ctx 32768, ngl 99). Use when user says 'start llama-server', 'serve the gguf', 'launch eval server', 'spin up Gemma 4', or 'serve <model>.gguf'. Backgrounds with disown, avoids head/tail SIGPIPE traps, and verifies /v1/models responds before returning. Do NOT use for: vllm serving, transformers/HF model loading, non-Gemma-4 GGUFs (different reasoning format), or production inference (this is an eval server config).
paths:
  - scripts/eval_*.sh
  - scripts/*.sh
---
# llama-server for Gemma 4 GGUF

## Critical

1. **`--reasoning-budget 8192` is MANDATORY.** Without it, llama-server emits malformed channel tokens (`<|channel>thought` missing closing) and crashes lm_eval mid-run at the first chemistry-heavy question. Wasted 1+ hour learning this — it is non-negotiable.
2. **`--reasoning-format deepseek` is MANDATORY** for Gemma 4. Pairs with `--reasoning-budget`; both must be present together.
3. **Never pipe llama-server output through `head` or `tail`.** They will SIGPIPE the server after a few lines and silently kill it. Use `tee` to a logfile or redirect to a file with `>`.
4. **Always background with `disown`** when launching from an interactive shell or Bash tool, otherwise the server dies when the parent shell exits.
5. **Never write the logfile to `/tmp`.** `/tmp` is tmpfs (64 GB RAM-backed). Write logs to the project folder or `../google/<model>/`.
6. The llama-server binary lives under the project's standard llama.cpp build tree. Do not invoke any other build.

## Instructions

1. **Locate the GGUF.** Confirm the file exists and note its absolute path. Typical locations:
   - `../google/<model-name>/` (quants from the parent project)
   - the project's `quants/` directory
   - Any path the user supplied verbatim. Use the user's exact path.
   Verify before proceeding:
   ```bash
   ls -lh <gguf-path>
   ```

2. **Pick a logfile path.** Convention: alongside the GGUF or in the project folder, with a name that includes the model tag. Never `/tmp`. Verify the parent dir exists.

3. **Check port 8099 is free.**
   ```bash
   ss -ltnp | grep :8099 || echo free
   ```
   If occupied, either kill the existing server (`pkill -f 'llama-server.*8099'`, then re-check) or pick another port and tell the user. Do NOT silently kill someone else's server — confirm with the user first if a different `llama-server` is already there.

4. **Launch with the canonical command.** Use this exact invocation, substituting `<gguf>` and `<log>`:
   ```bash
   nohup /opt/llama.cpp/build/bin/llama-server \
       -m <gguf> \
       --port 8099 \
       -c 32768 \
       -t 12 \
       -ngl 99 \
       --no-warmup \
       --reasoning-format deepseek \
       --reasoning-budget 8192 \
       > <log> 2>&1 &
   disown
   echo "PID=$!"
   ```
   Notes:
   - `nohup` + `disown` together survive shell exit and SIGHUP.
   - Redirect stdout+stderr to the logfile with `> <log> 2>&1`. Do NOT pipe to `head`/`tail`.
   - Keep flag order stable so the command is greppable in process lists.

5. **Wait for readiness.** Loop on the HTTP endpoint, NOT on log greps (logs lag). Cap at ~120 s for a cold load of a large GGUF:
   ```bash
   for i in $(seq 1 60); do
     curl -fsS http://localhost:8099/v1/models >/dev/null 2>&1 && { echo ready; break; }
     sleep 2
   done
   ```
   Verify `/v1/models` returns 200 before declaring success. If it never comes up, `tail -n 100 <log>` (reading a file, not piping the live server stream) and report the error.

6. **Smoke-test reasoning.** Send one short chat completion to confirm `--reasoning-budget` is actually active:
   ```bash
   curl -s http://localhost:8099/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"gemma-4","messages":[{"role":"user","content":"2+2=?"}],"max_tokens":64}' \
     | head -c 500
   ```
   The response must include a normal `content` field with no unclosed `<|channel>` tokens. If you see truncated channel markers, the reasoning flags are wrong — kill the server and re-launch.

7. **Report to the user**: PID, port, logfile path, GGUF path, and the exact lm_eval `base_url` (`http://localhost:8099/v1/chat/completions`).

8. **Hand off to lm_eval.** Pair this server with the canonical lm_eval invocation from `../CLAUDE.md` / `scripts/eval_gpqa_v3.sh`: `--model local-chat-completions`, `base_url=http://localhost:8099/v1/chat/completions`, `num_concurrent=1`, `tokenizer=<original-128e-or-base-model-dir>`, plus the **mandatory** `--use_cache` and `--log_samples`.

## Examples

**User says:** "start llama-server for the 109e Q4_K_M"

**Actions:**
1. `ls -lh ../google/gemma-4-A4B-109e-Q4_K_M.gguf` → confirm exists.
2. `ss -ltnp | grep :8099 || echo free` → free.
3. Launch:
   ```bash
   nohup /opt/llama.cpp/build/bin/llama-server \
       -m ../google/gemma-4-A4B-109e-Q4_K_M.gguf \
       --port 8099 -c 32768 -t 12 -ngl 99 --no-warmup \
       --reasoning-format deepseek --reasoning-budget 8192 \
       > ../google/llama-server-109e.log 2>&1 &
   disown
   echo "PID=$!"
   ```
4. Poll `curl -fsS http://localhost:8099/v1/models` until 200.
5. Curl smoke-test → clean response.

**Result:** Server up on PID 12345, log at `../google/llama-server-109e.log`. Ready for `lm_eval --model local-chat-completions --model_args base_url=http://localhost:8099/v1/chat/completions,...`.

## Common Issues

- **lm_eval crashes after a few questions with `KeyError` or malformed JSON, server still running:** `--reasoning-budget` was omitted or `--reasoning-format` is not `deepseek`. Kill server, re-launch with both flags. This is bug-005 / bug-009 from the parent project.
- **Server dies seconds after launch, log shows nothing past model load:** you piped output through `head` or `tail`. SIGPIPE killed it. Re-launch redirecting to a file with `> <log> 2>&1`.
- **Server dies the moment the Bash tool call ends:** missing `disown` (or you used `&` without `nohup` in some shells). Re-launch with `nohup ... &` then `disown`.
- **`bind: address already in use` on port 8099:** previous llama-server still running. `pkill -f 'llama-server.*8099'`, wait 2 s, retry. If it's a different user's process, pick another port and inform the user — do not kill blindly.
- **`/v1/models` returns 200 but generations are empty / `finish_reason: length` immediately:** the chat template isn't being applied. Confirm `--reasoning-format deepseek` is set; check `lm_eval` is passing `--apply_chat_template`.
- **Quants are way larger than expected and slow:** expert `intermediate_size` not divisible by 32 → tensors fell back to F16 during quantization. This is a quantization-side bug, not a server-side bug — re-quantize with a compatible expert width.
- **OOM on load with `-ngl 99` on a 3090:** GGUF is too large for 24 GB at full offload. Either lower `-ngl` (e.g. `-ngl 40`) or use a smaller quant (Q4_K_M instead of Q6_K). Tell the user before changing `-ngl` — it changes eval semantics.
- **`/tmp` filled up:** the log was written to `/tmp`. Move it to persistent disk and re-launch. Per the parent CLAUDE.md, `/tmp` is tmpfs and risks system collapse.
