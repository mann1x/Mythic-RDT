---
name: fetch-base-model
description: Downloads the canonical Mythic-Gemma4 base model `ManniX-ITA/gemma-4-A4B-98e-v3-it` from HuggingFace into `base/gemma-4-A4B-98e-v3-it/` and verifies SHA256 against the audit table in `BASE_MODEL_ANALYSIS.md`. Use when user says 'fetch base', 'download base model', 'set up base', 'phase 0 setup', 'bootstrap base', or starting a fresh project checkout. Refuses to substitute `../google/gemma-4-A4B-98e-hybrid/` (bytes differ). Do NOT use for fine-tuned checkpoints, GGUF downloads, or arbitrary HF model fetches — only the canonical Mythic-Gemma4 base.
paths:
  - base/**
  - scripts/convert_*.py
  - BASE_MODEL_ANALYSIS.md
---
# Fetch Base Model

## Critical

1. **The base model MUST be `ManniX-ITA/gemma-4-A4B-98e-v3-it` downloaded fresh from HuggingFace into the project's persistent base directory.** Do NOT symlink, copy, or substitute `../google/gemma-4-A4B-98e-hybrid/` — it is NOT bit-identical to the published artifact. SHA256s differ. See `BASE_MODEL_ANALYSIS.md`.
2. **Never download into `/tmp`.** /tmp is tmpfs (64 GB RAM-backed). The model is ~50 GB safetensors — it will eat RAM and disappear on reboot. Always download into the project's persistent directory.
3. **Use the `lightseek` conda env** (`conda activate lightseek`) for any transformers/HF interaction. NEVER touch the `vllm` env (older transformers, breaks Gemma 4 tokenizer).
4. **Verify SHA256 of every weight shard and tokenizer file against the table in `BASE_MODEL_ANALYSIS.md` BEFORE marking the fetch complete.** A silent mismatch invalidates every downstream eval and conversion.
5. **HF auth is pre-configured for user `ManniX-ITA`.** Do not re-login or rotate tokens. If `huggingface-cli whoami` fails, stop and ask the user.

## Instructions

### Step 1 — Read the canonical SHA256 table

Read `BASE_MODEL_ANALYSIS.md` and locate the section listing the expected SHA256 hashes for `ManniX-ITA/gemma-4-A4B-98e-v3-it`. Extract the filename → hash mapping into a dict you will use in Step 4.

```bash
grep -A 100 'SHA256' BASE_MODEL_ANALYSIS.md
```

*Verify before proceeding:* You have a non-empty mapping `{filename: sha256}` covering at minimum every weight shard, the index manifest, the tokenizer files, and the model config. If `BASE_MODEL_ANALYSIS.md` does not contain hashes, STOP and tell the user the audit table is missing — do not invent hashes.

### Step 2 — Confirm we are in the project root and target dir does not already hold a different artifact

From the project root (`Mythic-Gemma4/`):

```bash
pwd  # must end with /Mythic-Gemma4
ls -la 2>/dev/null
```

If the target download directory already exists and contains files, **do NOT overwrite**. Skip directly to Step 4 (verify-only mode). If verification in Step 4 passes, the fetch is already done — report and exit.

If the user has manually placed `../google/gemma-4-A4B-98e-hybrid/` and asks you to symlink it: REFUSE. Quote `BASE_MODEL_ANALYSIS.md`: "bytes differ." Offer to download fresh instead.

*Verify before proceeding:* You are in the `Mythic-Gemma4/` project root and the target dir is either absent OR present-and-being-verified.

### Step 3 — Download via `huggingface-cli`

Activate env, then download. Use `huggingface-cli download` (not `git lfs clone` — the LFS pointers are slower and waste a copy):

```bash
conda activate lightseek
huggingface-cli whoami  # must print ManniX-ITA
huggingface-cli download ManniX-ITA/gemma-4-A4B-98e-v3-it \
    --local-dir-use-symlinks False \
    --resume-download
```

`--local-dir-use-symlinks False` forces real files (not HF cache symlinks) so SHA256 verification works on the files themselves and the artifact survives a `~/.cache` purge. `--resume-download` makes the command idempotent across retries.

Run this in a long-running shell (not a tool call that may time out at 2 min). For a foreground bash call, set `timeout: 3600000` (1h). For background, use `run_in_background: true` and poll.

*Verify before proceeding:* Exit code 0. The download dir contains the model config, tokenizer files, the weight index manifest, and the full set of weight shards (count must match the index manifest's `weight_map` shard count). Total directory size is in the tens of GB.

### Step 4 — Verify SHA256 of every file

Uses the hash mapping from Step 1.

```bash
for f in *.safetensors *.json *.model; do
    actual=$(sha256sum "$f" | awk '{print $1}')
    expected=<lookup-from-step-1>
    if [ "$actual" != "$expected" ]; then
        echo "MISMATCH: $f"
        echo "  expected: $expected"
        echo "  actual:   $actual"
    else
        echo "OK: $f"
    fi
done
```

Prefer scripting this from Python with the dict from Step 1 rather than hardcoding bash. Print a final tally line: `N OK / M MISMATCH`.

*Verify before proceeding:* Every file is `OK`. **A single MISMATCH is a hard stop** — delete the offending file, re-run Step 3 (`--resume-download` will re-fetch only that file), and verify again. If a file mismatches twice in a row, STOP and ask the user; this likely means the HF revision changed and `BASE_MODEL_ANALYSIS.md` is out of date.

### Step 5 — Smoke-test load

Confirm the artifact is structurally loadable by transformers (config + tokenizer only — do not move 50 GB of weights to GPU just to verify):

```bash
conda activate lightseek
python -c "
from transformers import AutoConfig, AutoTokenizer
p = '.'
cfg = AutoConfig.from_pretrained(p)
tok = AutoTokenizer.from_pretrained(p)
assert cfg.model_type.startswith('gemma'), cfg.model_type
assert cfg.num_hidden_layers > 0
print('OK', cfg.model_type, 'layers=', cfg.num_hidden_layers, 'vocab=', tok.vocab_size)
"
```

*Verify before proceeding:* Output starts with `OK gemma`. If `AutoConfig` fails with `KeyError: 'gemma4'` or similar, you are in the wrong env — re-activate `lightseek` and confirm `transformers.__version__` is 5.5.0 or newer.

### Step 6 — Record the successful fetch

Follow the OpenWolf protocol from `../.wolf/OPENWOLF.md`:

1. Append one line to `../.wolf/memory.md` in the format `| HH:MM | fetched base | <download dir> | sha256-verified | ~tokens |`.
2. Update `../.wolf/anatomy.md` with an entry for the download dir (one description line + size estimate).
3. Do NOT commit the weights — `.gitignore` should already exclude the download dir (verify with `git check-ignore`).

Report to the user: directory path, total size, shard count, and that every SHA256 matched the audit table.

## Examples

**User says:** "fetch the base model, fresh checkout"

**Actions taken:**
1. Read `BASE_MODEL_ANALYSIS.md` → extract 23 expected SHA256 entries.
2. `pwd` confirms `/srv/.../Mythic-Gemma4`. Target dir does not exist.
3. `conda activate lightseek && huggingface-cli download ManniX-ITA/gemma-4-A4B-98e-v3-it --local-dir-use-symlinks False --resume-download` → 28 min, exit 0.
4. Python script verifies every file → `23 OK / 0 MISMATCH`.
5. Smoke test prints `OK gemma4 layers=42 vocab=...`.
6. Append to `../.wolf/memory.md` and `../.wolf/anatomy.md`.

**Result:** Download dir ready, 49 GB, audit-clean. Reported to user with shard count and total size.

---

**User says:** "can we just symlink the hybrid one in google/?"

**Actions taken:** Refuse. Quote `BASE_MODEL_ANALYSIS.md`: `../google/gemma-4-A4B-98e-hybrid/` is NOT bit-identical to `ManniX-ITA/gemma-4-A4B-98e-v3-it`. Offer to download the canonical artifact instead.

**Result:** No symlink created; user agrees to fresh download or explicitly overrides.

## Common Issues

- **`huggingface-cli whoami` returns `Not logged in`** → STOP, ask user. Do not run `huggingface-cli login` autonomously; tokens are managed by the user.
- **`KeyError: 'gemma4'` from `AutoConfig.from_pretrained`** → wrong env. Run `python -c "import transformers; print(transformers.__version__)"`. If <5.5.0 you are in `vllm` or system python — `conda activate lightseek` and retry.
- **One shard mismatches SHA256 once** → `rm <bad-file>` and re-run the Step 3 download command (`--resume-download` re-fetches only the missing file). Re-verify.
- **Same shard mismatches twice in a row** → likely the HF revision was updated upstream and `BASE_MODEL_ANALYSIS.md` is stale. STOP and tell the user; do not edit the audit table to make verification pass.
- **`No space left on device` mid-download** → check destination disk: `df -h .`. If `/` is full but the project disk has room, you accidentally let HF cache fall back to `~/.cache/huggingface`. Re-run Step 3 ensuring `--local-dir-use-symlinks False` and `HF_HOME=$(pwd)/.hf-cache` if needed; also confirm `pwd` is on the persistent dev disk, not `/tmp`.
- **Download times out at 2 min in a Bash tool call** → use `timeout: 3600000` (1h) or `run_in_background: true`. The model is tens of GB; default 2-min Bash timeout is not enough.
- **Download dir shows up as untracked in `git status`** → verify `.gitignore` covers it. If it does not, STOP — committing 50 GB of weights breaks the repo. Add the dir to `.gitignore` before doing anything else.
