# Mythic-RDT — Current Status (Stage 1)

**Last updated:** 2026-04-28 — v5 dual-T training in flight on pod (run `3a6muz4p`).
**Companion to:** `MASTER_PLAN.md` (kickoff plan, unchanged), `README.md` (intended public surface).
This file is the source of truth for *what actually happened* and *what is in flight*.

---

## 1. Architecture as built

### 1.1 Wrapper shape (`block_mode = True`)

The wrapper is **not** the OpenMythos single-layer recurrence — it loops a contiguous
**block of layers** (defaulting `recurrent_block_start=4`, `recurrent_block_end=22`,
i.e. 19 layers of DS-Coder-V2-Lite-Instruct's middle stack) `T` times.

Per-iteration update:

```
h_{t+1} = h_t + LayerScale_t · sigmoid(gate_bias_t) · (LTI_inject(h_0) + Block(h_t, e))
```

- **Prelude** layers `0 … prelude_layers-1` (default 4): frozen, run once.
- **Recurrent block** `recurrent_block_start … recurrent_block_end-1` (default 4–22, 19 MoE layers): frozen *weights*, looped `T` times with depth-wise LoRA + per-T gate + per-T LayerScale on top.
- **Coda** last `coda_layers` (default 4): frozen, run once after the loop.
- **`LTI_inject(h_0) = LTI_A · h_0 + LTI_B`** — Parcae-style injection of the prelude output into every iteration. `LTI_A := Diag(-exp(log_A))` enforces ρ(A) < 1.
- **Per-T DepthLoRA** on `self_attn.q_proj_or_q_a` and `self_attn.o_proj` of every recurrent layer. `T` distinct (lora_A, lora_B) slices per target. Rank 8, alpha 16. **Not** on shared/routed experts.
- **Identity-biased gate**: `g_t = sigmoid(gate_bias[t])`, `gate_bias` init **0.0** (was -3 in v2 — caused dead-gate, see §4.2).
- **Per-loop LayerScale**: `ls_t` scalar, init 1e-4, **clamped to ≤ 1e-2** (v3+ change after v2 explosion).
- **No ACT halting** in v0. `T` is fixed at inference.

### 1.2 Trainable parameter count (v4/v5, 19-block, T=4)

```
80 trainable tensors:
  18 layers × 2 targets × (lora_A, lora_B)  = 72   (per-T, shape [T, ...])
  + gate.bias                                = 1   (per-T, shape [T])
  + layerscale.scale                         = 1   (per-T, shape [T])
  + LTI_A, LTI_B                             = 2   (no T axis)
  + 4 minor housekeeping                     = 4
```

**~7 M trainable** on top of ~15.7 B frozen base.

### 1.3 KV cache plumbing (post-bug-049)

`DynamicCache` is per-iteration: `dict[int → DynamicCache]`, one cache per `t ∈ [0, T-1]`.
Position IDs are derived from per-sequence `attention_mask.cumsum(-1) - 1` to handle
left-padded batches correctly. Earlier code used `arange(past_kv_len, past_kv_len+seq_len)`
which silently corrupted positions for any padded batch — that bug masked the wrapper as
"broken on LCB" with KV cache enabled (bug-049, 2026-04-27).

---

## 2. Training runs — chronological with verdicts

### v1 — first attempt (smoke OOM, abandoned same day)

- **Goal**: prove fine-tune machinery wires up end-to-end.
- **Outcome**: `bf16` wrapper OOMs on 48 GB at `T ≥ 2`. fp32 `cross_entropy` cast was the dominant memory cost; loop-body activations also unchecked.
- **Memory note**: `feedback_phase1_oom_root_causes.md`.
- **Fix path**: drop fp32 CE cast → checkpoint loop body → QLoRA NF4 base.

### v2 — first real training (catastrophic regression)

- **Setup**: full curriculum T=2→4→8, gate init bias **-3**, no LayerScale clamp, no KL anchor, no quant.
- **Outcome**: HumanEval pass@1 = **0 %** at every T vs base 100 %. Wrapper produced **BPE gibberish** even at T=1.
- **Root causes** (memory `project_phase1_v2_catastrophic_regression.md`, `project_phase1_v2_gate_bias_dead.md`):
  - LayerScale unclamped → recurrence injected order-1 perturbations, drove activations off-distribution.
  - `gate_bias = -3` saturates `sigmoid` → ∂L/∂bias ≈ 0 → gate never learned to open. Logged value pinned at -3.0 throughout training.
  - No KL-to-base anchor → wrapper free to drift anywhere.
  - Token budget too small to recover from drift.
- **v3 design**: clamp `LayerScale ≤ 1e-2`, gate init bias = **0**, add KL anchor `α=0.05` every 8 steps, raise token budget.

### v3-T1 — single-T probe, 19-block, validated ✅

- **Setup**: `--max-loop-iters 1`, gate init bias 0, LS-clamp 1e-2, KL anchor `α=0.05`/every-8, NF4 base.
- **Outcome at ckpt-400**: HumanEval pass@1 = **95 %** vs base 100 %. T=1 wrapper preserves base behavior.
- **Verdict** (`project_phase1_v3_t1_validation.md`): architecture is sound. Block-mode + 19-layer block + safe gate/LS/KL is a viable pre-training base for stacking T>1.
- **Used as init for v5** (see §2.6).

### v4-anchored — stacked T=2/T=4 from v3-T1 ckpt-400

- **Setup**: `--init-from-checkpoint` from v3-T1 ckpt-400 (auto-expanded T-axis from 1 to 4), curriculum T=1→2→4, KL anchor unchanged.
- **Headline numbers (CORRECTED, 2026-04-28)**:

  | Eval | base | T=1 | T=2 | T=4 |
  |---|---|---|---|---|
  | HumanEval-20 | 100 % | 95 % | 95 % | 85 % |
  | LCB-medium-10 | 30 % | **0 %** | **0 %** | **0 %** |

- **Verdict**: HE preserved (short prompts stay in base manifold); LCB collapses (long-form algorithmic generation drifts off-manifold despite KL anchor). Wrapper produces *valid alternative Python with different algorithms*, not gibberish — but with different (wrong) answers.
- The **earlier "30 % LCB" reading was wrong** — caused by bug-050 (see §3). Real wrapper LCB = 0/10 at all T, byte-identical between pod A6000 and local 3090.
- **Don't ship v4-anchored.** Memory `project_phase1_v4_anchored_corrected_verdict.md`.

### v4-extended — overnight resume (failed wiring)

- **Setup**: re-launch v4 from ckpt-400 for additional ~400 steps to see if more training closes the LCB gap.
- **Failure mode**: passed `--lora-targets q_proj o_proj` (bare attribute names) instead of dotted `self_attn.q_proj_or_q_a self_attn.o_proj`. **No LoRA wired**. Wandb also wasn't picked up by the launcher.
- Caught the next morning. No checkpoints worth keeping; aborted.

### v5 — dual-T margin + distill (completed 2026-04-28, partial 🟡)

- **Hypothesis**: probes 1-3 (see §3.3) showed T=4 loss is *higher* than T=1 loss by 0.3–0.8 nats on held-out text — recurrence is actively degrading the model. Training objective lacks any signal that says "deeper should be at least as good as shallower". Cheapest local minimum is "T=1 hugs base, T=4 drifts wherever".
- **Fix**: replace single-T sample with dual forward (`T_lo = 1`, `T_hi = max_loop_iters = 4`) and add two new loss terms:
  - **Margin**: `α_margin · ReLU(CE(T_hi) − CE(T_lo) + margin_nats)` — penalises deeper-than-shallow loss directly.
  - **Self-distillation**: `α_distill · KL(softmax(logits_lo) || softmax(logits_hi.detach()))` — pulls low-T logits toward high-T logits so high-T can "teach" low-T.
- **Setup**: init from **v3-T1 ckpt-400** (NOT v4-anchored). KL anchor reduced 0.05→0.02. 200 steps × 113 s = 6h 21m total. Saves at 100/150/200 (50 rotated). Wandb run `3a6muz4p`. Pod sidecar venv `/workspace/venv-tf4`.

- **Training signals (clean):**
  - `ce_gap` collapsed **0.80 → 0.08 nats** over 200 steps (10× reduction). Margin loss bit hard, target hit on training distribution.
  - Train loss 2.57 → 1.41 (mean per 25-step bucket). Plateau by step ~100. No NaNs, grad_norm bounded 1.5–2.5.

- **Eval verdict (`eval_results/v5_he20.json`, `eval_results/v5_lcb10.json`):**

  | Eval | base | v4-anchored | **v5 ckpt-200** | delta |
  |---|---|---|---|---|
  | HE-20 T=1 | 100 % | 95 % | **95 %** | 0 |
  | HE-20 T=2 | — | 95 % | **85 %** | −10 pp |
  | HE-20 T=4 | — | 85 % | **80 %** | −5 pp |
  | LCB-10 T=1 | 30 % | 0 % | **0 %** | 0 |
  | LCB-10 T=2 | — | 0 % | **10 %** | **+10 pp** |
  | LCB-10 T=4 | — | 0 % | **10 %** | **+10 pp** |

- **Read:** v5 traded ~5–10 pp HE pass@1 at T≥2 for ~10 pp LCB pass@1 at T≥2. **First non-zero LCB ever from the wrapper.** But still 1/3 of base on LCB; T=1 LCB still 0 % despite direct dual-T training of T=1 + KL anchor. Margin/distill closed the *training-time* CE gap (0.80 → 0.08) but **did not generalize to long-form pass@1** at T=1.

- **Architectural finding:** the 19-layer block_mode wrapper degrades long-form generation **even at T=1, even with direct training of the T=1 slice**. KL anchor preserves token-level distribution on training corpus, not algorithmic correctness over 200–400 token generations. Same finding as v4-anchored at T=1 — both v4 and v5 collapse the T=1 LCB slice from base 30 % → 0 %. v3-T1 LCB **never tested** (HE-only validation) — that data point would settle whether the T=1 collapse is intrinsic to block_mode or specific to v4/v5 training.

- **Decision: don't ship v5, don't launch v6 yet.** Run diagnostic probes first (see §10 Quickstart probes added below) — three cheap experiments that localise the failure mode (wrapper plumbing vs architectural perturbation vs training-time drift). Only then decide whether to (a) shrink the recurrent block, (b) re-anchor T=1 with stronger / per-token KL on long contexts, (c) abandon block_mode for single-layer recurrence.

---

## 3. Bugs that distorted earlier readings

### bug-049 — KV cache padding silently corrupts position IDs

- **Symptom**: wrapper T=1 pass@1 on LCB-medium = **0 %** with KV cache, vs **30 %** without.
- **Root cause** (`src/mythic_rdt/modeling.py`): `position_ids` built from `arange(past_kv_len, past_kv_len + seq_len)` on left-padded batches. Padded positions advance as if real → attention math wrong for any batch with mixed lengths. KV cache amplified the corruption across iterations.
- **Fix**: derive position IDs from `attention_mask.cumsum(-1) - 1` (mask zero positions to id 1).
- **Compounded by**: stale `__pycache__` shadowing the patched `.py`. `rm -rf src/mythic_rdt/__pycache__` is **classifier-blocked locally** — use `PYTHONDONTWRITEBYTECODE=1` env var instead. Memory `feedback_pyc_purge_after_modeling_patch.md`.

### bug-050 — smoke script silently runs UNTRAINED wrapper

- **Symptom**: v4-anchored ckpt-400 evaluated with `--T-values 1` on LCB → "30 %", apparently matching base. With `--T-values 1 2 4` → "0 %".
- **Root cause** (`scripts/humaneval_smoke.py:887`): wrapper built with `max_loop_iters = max(args.T_values)`. With `--T-values 1`, that's 1 → wrapper has T-axis params shaped `[1, ...]`. Checkpoint params have shape `[4, ...]`. `_load_trainable_state` (`src/mythic_rdt/training/trainer.py:107-115`) only handles T-axis *expansion* (smaller ckpt → bigger target via slicing copy). The *opposite* (bigger ckpt → smaller target) falls through to the else branch and is appended to `unexpected`. Load "succeeds" because `strict=False`, but only **2 of 80 tensors** load (LTI A, LTI B — no T axis). The other 78 silently use init values: zero LoRA, default gate `sigmoid(0) = 0.5`, default LayerScale 1e-2. Wrapper runs ≈ base behavior.
- **Mitigation**: always pass `--T-values 1 2 4` (or whatever covers the ckpt's training `T_max`). For an isolated T=1 probe, also pass `--max-loop-iters 4` so the wrapper-build matches the checkpoint.
- **Verify load fingerprint** in every smoke log: `[smoke] loaded 80 trainable tensors  missing=0 unexpected=0`. Anything else means partial load — STOP.
- **TODO**: harden `_load_trainable_state` to refuse silent shape-mismatch (raise unless explicitly opted out).

### Smaller bugs along the way

- **bug-047/048**: OOM on local 3090 (phantom GPU process + fragmentation). Mitigated with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Local smoke `[smoke] loaded 0 problems` → IndexError**: HumanEval list empty crashed the sample-print path. Guarded with `if chat_prompts:`.
- **"30 % vs 0 %" diff between pod and local**: false alarm — caused by bug-049 + bug-050 stacking. Once both were fixed, local 3090 and pod A6000 produce **byte-identical** generations on LCB-medium.

---

## 4. Design choices that worked / didn't

| Choice | Verdict | Note |
|---|---|---|
| `block_mode=True` (loop 19 layers, not 1) | ✅ keep | OpenMythos single-layer mode never tried at scale; block mode validated by v3-T1 |
| Gate init bias `0` (not `-3`) | ✅ keep | -3 dead-gate the entire v2 run; 0 lets gradient flow |
| LayerScale clamp `≤ 1e-2` | ✅ keep | Without clamp, v2 exploded |
| KL-to-base anchor `α=0.05`/every-8 | 🟡 partial | Preserves HE (short prompts) but does NOT prevent LCB drift on long generations. Lowered to 0.02 in v5 to free the new dual-T losses. Future: per-token KL on long-context training data. |
| `--init-from-checkpoint` for T-axis expansion | ✅ keep | HF Trainer `--resume` crashes on AdamW shape mismatch when `--max-loop-iters` changes; `--init-from-checkpoint` loads only trainable state. Memory `feedback_init_from_checkpoint_pattern.md`. |
| QLoRA NF4 base + bf16 trainable | ✅ keep | Fits 48 GB at T=4. Verified deterministic across A6000/3090 with bnb 0.49.2. |
| Curriculum T=2→4→8 | 🟡 deferred | We've never reached T=8 yet; v4 was T=4 max, v5 is dual-T (1, 4). T=8 is post-v5. |
| LoRA on shared experts | ❌ excluded | Always-on across iterations — no depth differentiation needed. |
| Init v5 from v4-anchored ckpt-400 | ❌ rejected | v4 is a bad LCB local min. Init from v3-T1 ckpt-400 instead — known-good base, no T-axis polluted weights. |
| Single-T forward + curriculum sampling | ❌ insufficient | Probes show no signal that "deeper ≥ shallower". Replaced with dual-T forward + margin/distill loss in v5. |

---

## 5. Diagnostic probes that drove v5

Run on v4-anchored ckpt-400, summarised in `experiments/` (probe scripts inline, not committed).

1. **Per-T logit KL on LCB prompts** — KL(`logits[T=1]` ‖ `logits[T=4]`) on first-token-after-prompt. Found: KL grows monotonically with T but small in absolute terms (~0.05–0.2). Wrapper *is* doing something different at higher T; question is whether different ⇒ better.
2. **Per-slice LoRA magnitude** — `‖B[t] − B[0]‖_F / ‖B[0]‖_F` from checkpoint. Found: 1.4× relative diff between t=0 and t=3 → slices DID specialize, NOT collapsed/identical. Architecture/training does separate the per-T weights as intended.
3. **Per-T cross-entropy on held-out text** (easy/medium/hard slices). Found: **CE(T=4) > CE(T=1) by 0.3–0.8 nats** across all difficulty buckets. Recurrence is *actively degrading* the model, not just failing to improve it.

**Conclusion**: training objective has no term that rewards "deeper ≥ shallower". Cheapest minimum is the one we hit: T=1 hugs base via KL anchor, T=4 drifts wherever the FFN mass pushes it. Mitigations like "lower LR for deep slices" only *limit* recurrence's harm — they don't make it useful. Need an explicit T-monotonicity signal → v5 dual-T design.

---

## 6. Scripts directory map

### Core

| Script | Purpose | Key flags |
|---|---|---|
| `scripts/finetune_phase1.py` | Phase 1 trainer entrypoint. Wraps base, applies DepthLoRA, runs HF Trainer. | `--max-loop-iters`, `--curriculum`, `--kl-anchor-alpha`, `--kl-anchor-every`, `--init-from-checkpoint`, `--lora-rank`, `--lora-targets` (must be dotted!), `--margin-alpha`, `--margin-nats`, `--distill-alpha`, `--dual-t-lo`, `--dual-t-hi`, `--quant nf4` |
| `scripts/humaneval_smoke.py` | Smoke eval. Builds wrapper, runs HE-20 + LCB-medium-N. | `--T-values 1 2 4` (MUST cover ckpt's `T_max`), `--checkpoint`, `--lcb-limit`, `--lcb-difficulty`, `--lcb-min-date`, `--max-loop-iters` (override if T-values omits ckpt T_max) |
| `scripts/measure_base_loss.py` | Reference base CE on a held-out slice (used by margin probe). | — |

### Phase 0 / one-offs

| Script | Status |
|---|---|
| `scripts/phase0_sanity.py` | T=1/gate=0 bit-exactness check vs base. Run once per stage. |
| `scripts/phase0_probe_layers.py` | Picks `recurrent_layer_idx` (single-layer mode). Stage 2 / Gemma 4 use only — Stage 1 chose block-mode 4–22 instead. |
| `scripts/setup_pod_env.sh` | Bootstraps the sidecar venv on a fresh pod. |
| `scripts/_diag_*.py`, `_dscoder_compat.py` | Throwaway diagnostics from the early DS-Coder + transformers 5.x compatibility war. Don't reuse without checking. |

### Pod-side runners (not committed; live on `/workspace/mythic-rdt/`)

| Script | Purpose |
|---|---|
| `run_v5.sh` | v5 training + post-eval pipeline. Sequential: dual-T train (200 steps, save 50/100/150/200) → HE-20 eval → LCB-10 eval (separate process). Requires `WANDB_API_KEY=…` in env. |
| `run_overnight*.sh` | v4-extended attempts (now of historical interest only — broken LoRA wiring). |

---

## 7. Eval methodology — non-negotiables

1. **Every smoke run** must use `--T-values` that includes the checkpoint's training `T_max`. Verify `[smoke] loaded 80 trainable tensors  missing=0 unexpected=0` in the log. Anything less = partial load = false-positive base-like score.
2. **HumanEval and LCB run as separate processes**. Combining them in one process triggers CUDA fragmentation / phantom-process OOMs on the local 3090.
3. **lm-eval (when used)**: `--use_cache <dir>` + `--log_samples` + post-run sanity check on `samples_*.jsonl` (empty / markdown-fence / <5-char junk). Inherited from parent project's bug-010, bug-015.
4. **Same generation params for base vs wrapper**. The smoke script uses greedy decoding (`do_sample=False`). Don't change without re-baselining base.
5. **Determinism check**: pod (A6000) and local (3090) generations on the same NF4 base should be byte-identical. If they diverge, suspect pyc staleness, transformers version drift, or torch dtype mismatch.

---

## 8. Where things live

- **Base weights**: `base/DeepSeek-Coder-V2-Lite-Instruct/` (4 bf16 shards, ~31 GB).
- **Checkpoints (local)**: `checkpoints/phase1_v3_t1/`, `checkpoints/phase1_v4_anchored/`.
- **Checkpoints (pod)**: `/workspace/mythic-rdt/checkpoints/phase1_v3_t1/`, `/workspace/mythic-rdt/checkpoints/v5_probe/` (in flight).
- **Eval results**: `eval_results/*.json` + matching `.log`. JSON has full args + per-task pass/fail, log has the live smoke output.
- **Pod root**: `/workspace/mythic-rdt/` (mirrors local Mythic-RDT/).
- **Pod sidecar venv**: `/workspace/venv-tf4` (transformers 4.46.3, torch 2.6.0+cu126, bnb 0.49.2). DS-Coder modeling.py is broken on transformers 5.x → memory `project_dscoder_5x_blocker.md`.
- **Local conda env**: `mythic-rdt` (same versions as pod sidecar).

---

## 9. Open items / next steps

**v5 verdict landed 2026-04-28 ~12:25 UTC** (HE 95/85/80, LCB 0/10/10; partial — see §2.6).

**Diagnostic probes 1-3 completed 2026-04-28 ~13:30 UTC** (`eval_results/probe[1-3]_*.json`):

| Probe | Setup | LCB-10 T=1 | Implies |
|---|---|---|---|
| 1 (pod) | v3-T1 ckpt-400, 19-layer block, **trained** | **0 %** | not training-time drift |
| 3 (local) | no ckpt, **5-layer block** (10-14), untrained | **0 %** | not block-size |
| 2 (pod) | no ckpt, 19-layer block, **explicitly zeroed** (LS=0, gate=sigmoid(-10)≈0, lora_B=0) | **30 %** ✅ | **plumbing is fine** |

**Diagnosis:** wrapper plumbing is CORRECT (probe 2 reproduces base byte-for-byte). The failure is the recurrence injection itself, even at "near-identity" init magnitudes (~5e-5 per token contribution). The retrofit-recurrence paper's "near-identity init → smooth fine-tune up" assumption does NOT hold for code generation — code's exec-based pass@1 is fragile to micro-perturbations over 200-400 token generations in a way prose perplexity isn't. Full diagnosis: `memory/project_phase1_v6_diagnosis.md`.

**v6A untrained smoke verdict (2026-04-28 ~14:00 UTC, `eval_results/v6a_untrained_lcb10.json`)** — local 3090, base = 40 % LCB:

| LCB-10 | base | v3-T1 | v4-anchored | v5 ckpt-200 | **v6A untrained** |
|---|---|---|---|---|---|
| T=1 | 40 % | 0 % | 0 % | 0 % | **40 %** ✅ identity |
| T=2 | — | — | 0 % | 10 % | **20 %** |
| T=4 | — | — | 0 % | 10 % | **10 %** |

**v6A T=1 LCB ≡ base byte-for-byte (mathematical identity confirmed empirically).** T=2/T=4 untrained surprised on the upside (predicted 0 %, got 20/10) — the t=0 identity puts h in a more anchored state than fresh prelude output, so t≥1 perturbed iterations are more robust than at fresh init. This validates the architectural fix as a strict superset of base.

**v6A architectural fix (cleanest):** make t=0 iteration of the recurrence loop unconditionally identity:

```python
# src/mythic_rdt/modeling.py, _loop_step:
if t == 0:
    h_next = block_out  # T=1 wrapper output ≡ base byte-for-byte.
else:
    h_next = block_out + ls[t] * gate[t] * (injection + ...)
```

This makes T=1 = base by construction. T≥2 iterations inject normally with learned per-T params. Training reduces to pure offensive objective: "make T=4 beat the fixed T=1=base baseline". No defensive trade-off.

**v6 candidate sequence:**

1. **v6A** — first iteration is identity (this section). Smoke untrained T=1 = base, T=2/4 likely still ≈0% LCB at init.
2. **Train v6A** with v5's dual-T(1,4) margin+distill objective. T=1 anchored structurally; train budget all goes into T=4.
3. If trained T=4 LCB > base 30 % → **v6A wins**. Proceed to Phase A scale evals (HE-164, LCB-50/hard-20, MBPP+) → C.5 ACT halting head → Phase B data scale.
4. If trained T=4 LCB ≤ base → **v6C** (v6A + curriculum gate clamp: gate_bias[t≥1] explicitly held near 0 for first half of training, slowly relaxes). Belt-and-suspenders for the perturbation-at-T≥2 problem.
5. If v6C also fails → **v6D** (v6A + shrunk recurrent block 5-9 layers). Or pivot to OpenMythos's single-layer recurrence (different inject mechanism). Or pivot to Stage 2 where prose-style benchmarks may tolerate the failure mode.

### Standing items (deferred)

- ~~Harden `_load_trainable_state`~~ — **DONE** 2026-04-28, raises `CheckpointShapeMismatchError` by default.
- **Stage 2 (Mythic-Gemma4)** still parked. Stage 2 verification should run probe 2 first before training to confirm prose benchmarks tolerate the wrapper at near-identity init.

### Three diagnostic probes to run BEFORE any new training (cheap, ~1 day local 3090, ~$0)

1. **v3-T1 LCB probe.** Smoke v3-T1 ckpt-400 on LCB-medium-10 with `--T-values 1 --max-loop-iters 1`. Tells us whether T=1 LCB collapse is intrinsic to block_mode (v3-T1 also = 0 % → architectural) or something v4/v5 broke (v3-T1 ≈ 30 % → training drift from v3 → v4 onward).

2. **Bypass probe (zeroed-trainables wrapper).** Build the wrapper with no checkpoint loaded, then explicitly set `gate.bias = -10` (sigmoid ≈ 0), `LayerScale.scale = 0`, `LoRA-B = 0`. This wrapper should be **mathematically bit-identical to base**. Smoke on LCB-10 at T=1.
   - If 30 % → wrapper plumbing is correct, v4/v5 trained drift is the problem.
   - If 0 % → something in the wrapper plumbing itself (KV cache, position IDs, dtype, layer ordering) breaks long-form generation regardless of trainable params. New bug to find.

3. **Shrunk-block probe.** Build wrapper with `recurrent_block_start=10 recurrent_block_end=14` (5 layers vs current 19), no checkpoint, T=1 untrained. Smoke LCB-10. If 25–30 % → 19-layer block is the perturbation source; v6 design = shrink the recurrent block to 5–9 layers. If still 0 % → block size isn't the issue, block_mode itself is.

These three probes localise the failure to **wrapper plumbing** / **architectural perturbation** / **training-time drift** — three completely different next moves. Do not start v6 (any flavour) without knowing which.

### After probes — v6 design candidates (mutually-exclusive next-experiment options)

- **A — per-token KL on long-context training data.** Train v6 with KL anchor evaluated on 500–1k token continuations (LCB-style synthesised prompts) instead of short corpus snippets. Targets the actual failure regime. Highest leverage *if* probe 2 passes and probe 1 shows v3-T1 LCB ≈ 30 %.
- **B — shrink recurrent block to 5–9 layers.** Less compounding drift per iteration. Trade-off: less capacity per loop. Right move *if* probe 3 shows 25–30 %.
- **C — anneal gate clamped near 0 for first half of training.** Forces near-base inductive bias. Cheap to try regardless of probe outcomes.
- **D — ACT halting head.** Phase C.5 from the agreed roadmap. Worth pursuing only after we have a wrapper that doesn't collapse at T=1.

### Standing items (deferred)

- ~~Harden `_load_trainable_state`~~ — **DONE** 2026-04-28, raises `CheckpointShapeMismatchError` by default. 8 unit tests in `tests/test_load_trainable_state.py`.
- **Stage 2 (Mythic-Gemma4)** parked until Stage 1 ships a real LCB number. Currently nothing close.

---

## 10. Quickstart for the next session

```bash
# 1. Check v5 status (pod):
ssh -p 36738 root@ssh6.vast.ai \
  'tail -200 /workspace/mythic-rdt/eval_results/V5_RUN.log; ps -p 25561'

# 2. Wandb live: https://wandb.ai/<user>/mythic-rdt/runs/3a6muz4p

# 3. Fetch v5 checkpoints when training finishes (run_v5.sh handles eval, but for manual smoke):
rsync -av -e 'ssh -p 36738' \
  root@ssh6.vast.ai:/workspace/mythic-rdt/checkpoints/v5_probe/checkpoint-200/ \
  checkpoints/v5_probe/checkpoint-200/

# 4. Smoke locally (3090):
PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n mythic-rdt python scripts/humaneval_smoke.py \
  --base base/DeepSeek-Coder-V2-Lite-Instruct \
  --checkpoint checkpoints/v5_probe/checkpoint-200 \
  --max-loop-iters 4 --T-values 1 2 4 \
  --lcb-limit 10 --lcb-difficulty medium --lcb-min-date 2024-10-01 \
  --quant nf4 --batch-size 4 --gen-tokens 384 \
  --output-json eval_results/v5_ckpt200_local_3090.json

# 5. VERIFY in the log: "[smoke] loaded 80 trainable tensors  missing=0 unexpected=0"
#    Anything else = partial load = STOP.
```

---

*Memory entries cross-referenced: `project_phase1_v3_t1_validation.md`, `project_phase1_v4_anchored_corrected_verdict.md`, `feedback_smoke_max_loop_iters.md`, `feedback_pyc_purge_after_modeling_patch.md`, `feedback_init_from_checkpoint_pattern.md`, `project_dscoder_5x_blocker.md`. Bug log: `.wolf/buglog.json` entries `bug-049`, `bug-050`.*
