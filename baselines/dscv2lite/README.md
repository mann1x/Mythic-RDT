# DSC-V2-Lite-Instruct baselines (Mythic-RDT)

Pinned eval JSONs for the Mythic-Coder pre-recurrence floor. Mirrored from
the CTD repo `baselines/dscv2lite/` so Mythic-RDT can validate wrapper
deltas without depending on cross-repo file lookups.

| File | HE pass@1 | MBPP pass@1 | Notes |
|---|---:|---:|---|
| `BASE_FA2_CHAT_v2_HE_MBPP.json` | **73.8%** (121/164) | **64.6%** (244/378) | DSC-V2-Lite-Instruct base, no adapter; canonical for v2 |
| `M115_DSCV2LITE_M37c_HE_MBPP.json` | 71.3% (117/164) | 66.1% (250/378) | M37c recipe (funcsig + --code-only-mask, lr=5e-5, r=16, 2 epochs) — net wash vs base |

## Eval recipe

- Base: `/workspace/mythic-rdt/base/DeepSeek-Coder-V2-Lite-Instruct` (with
  vectorized MoE routing patch)
- Stack: `attn_implementation="flash_attention_2"` + `model.merge_and_unload()`
  for adapters + `tokenizer.apply_chat_template` user-message wrap
- `06_eval_batched.py` flags: `--quant bf16 --chat-template --batch-size 8 --max-new-tokens 1024 --exec-timeout 30 --he-limit 164 --mbpp-limit 378`

## Why these are committed in BOTH repos

CTD owns the eval script + adapter results (`baselines/dscv2lite/`). Mythic-
RDT mirrors them so the wrapper's success-criteria targets stay verifiable
even if a future maintainer only checks out Mythic-RDT.

## Baseline drift vs the prior doc claim

`BASE_DEEPSEEK_CODER_V2_LITE.md` (commit 88164cc, 2026-05-07) reported
75.6 / 60.6, measured on a now-destroyed RTX 4090 vast.ai pod. Re-eval on
the current RTX 6000 Ada pod (same script + flags) lands 73.8 / 64.6 —
within ~2 pp on HE and ~4 pp on MBPP. The MBPP gap is the larger drift,
plausibly from a between-pod tokenizer/cache/seed nuance; future reruns
should anchor to **73.8 / 64.6 measured here**, not the old 75.6 / 60.6
doc figure. Targets in MASTER_PLAN.md / BASE_DEEPSEEK_CODER_V2_LITE.md
should be re-anchored accordingly:

- T=8 HumanEval pass@1 ≥ base + 5 pp → ≥ **78.8 %** (was 80.6)
- T=8 MBPP pass@1 ≥ base + 3 pp → ≥ **67.6 %** (was 63.6)

The relative target (+5pp / +3pp) is unchanged.
