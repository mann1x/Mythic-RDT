# Mythic-RDT — Current Status (Stage 1)

**Last updated:** 2026-04-30 — **v6K** (v6H baseline + focal-weighted CE on T=4) training in flight on vast.ai pod 35822024 (wandb `et2eektc`), ETA finish ~17:00–22:30 UTC. v6H = controlled baseline (no harm, no win). v6I REJECTED (kl_anchor proved load-bearing).
**Companion to:** `MASTER_PLAN.md` (kickoff plan, unchanged), `README.md` (intended public surface).
This file is the source of truth for *what actually happened* and *what is in flight*.

**TL;DR scoreboard (LCB-30 medium, base = 30% = 9/30 problems):**

| Run | T=1 | T=2 | T=4 | Verdict |
|---|---|---|---|---|
| v3-T1 | 0% | — | — | T=1 LCB collapse intrinsic to block_mode |
| v4-anchored | 0% | 0% | 0% | wrapper produces alt-but-wrong code |
| v5 dual-T | 0% | 10% | 10% | first non-zero LCB ever; T=1 still 0% |
| v6A trained | 26.7% | 6.7% | 3.3% | drift: T>1 worse than T=1; recipe REJECTED |
| **v6H** | **30%** | **20%** | **26.7%** | **first to NOT catastrophically degrade T=4** |
| v6I (LCB-10) | 30% | 20% | 10% | catastrophic; kl_anchor was load-bearing; REJECTED |
| **v6K** | TBD | TBD | TBD | **active — target T=4 ≥ 33%** |

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

- **Decision (then): don't ship v5, don't launch v6 yet.** Run diagnostic probes first.
- **What actually happened next:** probes 1-3 ran 2026-04-28; localized the failure to recurrence injection at near-identity init magnitudes. v6 series began. See §2.7+.

### v6A — first-iter-identity architecture, dual-T trained, REJECTED 🔴

- **Setup:** make t=0 iteration of the recurrence loop unconditionally identity (`h_next = block_out`, no LTI/gate/LayerScale/LoRA contribution). T=1 wrapper output ≡ base byte-for-byte by construction. Then dual-T(1,4) margin+distill train as in v5, 200 steps.
- **v6A untrained smoke** (local 3090, base=40% LCB-10): T=1=40% (=base, byte-identical), T=2=20%, T=4=10%. Identity invariant held empirically.
- **v6A trained ckpt-200 LCB-30** (after v6E inference fix, see §2.8): T=1=26.7% (=base −1), T=2=6.7%, T=4=3.3%. **Trained recurrence is harmful and compounds across iterations.** Failures at T=4 dominated by syntax errors. Drift root cause: the `block_mode` formula `h_next = block_out + small·injection` REPLACES h with the 19-layer block output every iteration → ||h|| grew 5.74× over 4 iters (probe `_probe_recurrence_drift.py`). Coda received OOD activations.
- **v6A weights unsalvageable.** No post-loop fix (B/C/blend) recovered T=4 above 0% — drift is BOTH magnitude AND direction (cos→0 on hypersphere). Memory `project_v6a_post_fix_verdict.md`, `project_recurrence_root_cause_block_mode.md`, `project_inference_fix_test_v6a.md`.

### v6E inference path bug (caught mid-v6A debugging)

- **Symptom:** all "v6E base-identity at T=1" claims via `model.generate()` were FALSE.
- **Root cause** (`src/mythic_rdt/modeling.py`): `MythicRDTDeepseekV2ForCausalLM.forward` has TWO branches — `use_cache=False` (training) calls `_loop_step` (which had the v6E `first_iter_identity` logic), and `use_cache=True` (HF generate path) inlines the loop body in `forward` (which had ZERO `first_iter_identity` logic). HF `GenerationMixin.generate()` always passes `use_cache=True`. Every wrapper.generate() call ran with LoRA[0] active + recurrence add at t=0, even though training treated t=0 as identity.
- **Fix:** mirror `_loop_step` semantics in the inline use_cache loop (`set_loop_t(-1)` + `h = block_out` at t=0).
- **Cost:** ~6 hours of probes localizing it. New cerebrum Do-Not-Repeat: any change to recurrence loop semantics MUST be implemented in BOTH branches. Memory `project_v6e_inference_path_bug.md`, commit `be8a0d4`.

### v6H — Fix-A `block_mode_residual` baseline ✅ CONTROLLED

- **Setup:** new architectural fix for the block_mode drift problem — replace `h_next = block_out + small·injection` with `h_next = h + ls·g·(block_out − h) + small·injection`. The h-residual bounds ||h|| growth structurally. Recipe: `block_mode_residual=True`, `first_iter_identity=True`, prelude=4 / coda=4, recurrent block 4..22 (19 layers), `max_loop_iters=4`, LoRA rank 8, `layerscale_init=0.05` / `clamp_max=0.5`, `gate_init_bias=0.0`, `kl_anchor_alpha=0.5/every-2`, `margin_alpha=0.10/nats=0.02`, dual-T(1,4), 400 steps × ~66s/it = ~7h22m. Wandb `ueors8i6`.
- **Eval (pod RTX 6000 Ada, NF4 base):**

  | Eval | base | T=1 | T=2 | T=4 |
  |---|---|---|---|---|
  | HE-20 | 100% | 100% | 100% | 100% |
  | LCB-30 medium | 30% | 30% | 20% | 26.7% |

- **Verdict:** **first v6 to NOT catastrophically degrade T=4** (vs v6A LCB-30 T=4=13.3%). HE-20 perfect across all T = recurrence does ZERO harm on easier algorithmic problems. LCB-30 T=1 byte-exact base parity (first_iter_identity invariant + KL anchor preserved LoRA-B[0] from drifting). T=4 still −1 problem vs base (controlled, near-base, no harm but no win either).
- **Why no T>base improvement:** wandb-visible loss tug-of-war. `kl_anchor (α=0.5)` pulls T=4 → base@T=1 distribution; `margin (α=0.10)` pushes T=4 NLL ≤ T=1 NLL − 0.02. Optimizer converges to "T=4 ≈ base in distribution + tiny CE win + no useful generation refinement". Memory `project_v6h_final_verdict.md`.

### v6I — drop kl_anchor, REJECTED 🔴 (kl_anchor was load-bearing)

- **Hypothesis:** the v6H tug-of-war was the *cause* of "controlled but flat". Drop `kl_anchor` (α=0.5 → 0), bump `margin_alpha` 0.10 → 0.15, tighten `layerscale_clamp_max` 0.5 → 0.25. Same Fix-A architecture as v6H. Wandb `r328zfvp`.
- **Pod-side training metrics looked HEALTHIER than v6H:** ce_gap matured to −0.075 (vs v6H −0.06), margin_loss collapsed to ~0 sustained, no NaNs.
- **Actual generation result (pandorum 5080 16GB, LCB-10, base=40%/4):**

  | Ckpt | T=1 | T=2 | T=4 |
  |---|---|---|---|
  | ckpt-200 | 30.0% | 20.0% | 10.0% |
  | ckpt-300 | 30.0% | 20.0% | 10.0% |

- **Identical 30/20/10 across two consecutive checkpoints — stable catastrophic.** Killed at step 350/400 (~$4 burn before kill). The optimizer found a deeper CE win by drifting T=4 distribution OFF the manifold — loss reward was real but didn't transfer to autoregressive generation.
- **The actual lesson:** `kl_anchor` was load-bearing, not a brake. It is the bound that keeps T=4's distribution close enough to base that the wrapper's coda can still parse it. Margin alone provides no such constraint; Fix-A `block_mode_residual` + tighter LayerScale clamp 0.25 is NOT sufficient safety without distributional pressure.
- **New cerebrum Do-Not-Repeat (2026-04-30):** before dropping a regularizer because two losses "look opposing" on wandb, run a sacrificial short A/B (≤200 steps + interim eval). Don't commit a 7-13h run to a hypothesis pure mechanism analysis can't refute. Loss-on-training ≠ generation quality at T>1; deeper ce_gap with a regularizer removed is a RED FLAG. Memory `project_v6i_rejected.md`.

### v6K — v6H baseline + focal-weighted CE on T=4 (in flight 🟡)

- **Setup:** restore v6H's full recipe (kl_anchor 0.5 + margin 0.10 + clamp 0.5) and add ONE delta — focal-weighted CE on T=4 only:

  ```
  focal_w_per_token = (1 − p_T1_correct) ^ gamma   # gamma = 1.0
  ce_hi_focal = sum(focal_w * ce_per_tok) / sum(focal_w)
  ```

  `ce_lo (T=1)` stays unweighted so T=1 still trains on all tokens uniformly. Concentrates T=4 gradient on tokens where T=1 is uncertain — exactly the tokens where recurrence has *room* to add value, vs being averaged out across confident tokens.
- **Run:** `phase1_v6k_focal_anchored` on pod 35822024 (RTX 6000 Ada, NF4), 400 steps × ~74s/it = ~8h ETA. Launched 2026-04-30 09:23 UTC, ETA finish ~17:30–22:30 UTC. Wandb `et2eektc`.
- **Hypothesis:** v6H proved architecture works (no harm) but T=4 self-cancels back to base because the dual-T loss applied uniformly across all tokens averages out the recurrence's potential contribution on hard tokens with its harm on easy ones. Focal weighting concentrates the gradient where it counts.
- **Risks** (per cerebrum 2026-04-29 design discussion):
  1. "Hard" ≠ "improvable" — high-entropy tokens may be genuinely uncertain (random variable names). Mitigation: ce_lo unweighted preserves T=1 capability; gentle gamma=1.0 (not 2.0).
  2. kl_anchor + margin tug-of-war returns. Mitigation: same as v6H — it works (T=4 = base parity, no harm). Focal adds info without adding a third opposing force.
- **Decision rules at finish:**
  - T=4 LCB-30 ≥ 33% (= +1 problem on LCB-30) → focal CE is the productive direction; iterate on gamma or add EMA in v6L.
  - T=4 ≈ 27% (= v6H) → focal didn't help; need a fundamentally different signal (e.g., teacher distillation from fp16 self).
  - T=4 << 27% → focal damages even with anchor present; abandon focal direction.
- Memory `project_v6k_design.md`. Run script `scripts/pod_runner/run_v6k.sh`.

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

### Active investigation: v6K (in flight)

- **Run:** `phase1_v6k_focal_anchored` on pod 35822024, wandb `et2eektc`. ETA finish 2026-04-30 ~17:00–22:30 UTC.
- **Decision rules** at v6K finish (LCB-30 medium, base=30%):
  - **T=4 ≥ 33% (≥+1 vs base)** → focal CE is productive; iterate (v6L = focal + EMA, or sweep gamma 0.5/1.5/2.0).
  - **T=4 ≈ 27% (≈ v6H)** → focal didn't help; pivot to teacher distillation from fp16 self (same tokenizer, removes NF4 noise).
  - **T=4 << 27%** → focal damages even with anchor; abandon focal direction; reconsider per-token KL on long-context training data.
- **Pre-eval gate:** verify `[smoke] loaded 80 trainable tensors  missing=0 unexpected=0` (bug-050 guard). Run pandorum-side LCB-30 first (free GPU), then HE-20 if v6K has any LCB delta worth confirming.

### Settled items (decisions locked, no re-litigation)

- **Architecture:** Fix-A `block_mode_residual=True` + `first_iter_identity=True`. Validated by v6H. The pre-Fix-A `h_next = block_out + small·injection` formula is permanently rejected (drift root cause, see §2.7).
- **kl_anchor IS load-bearing.** Empirically refuted the 2026-04-29 "tug-of-war = drop one of the two" framing via v6I. Cerebrum Do-Not-Repeat updated. Future loss recipes ADD information on top of v6H, not REMOVE safety.
- **first_iter_identity is INIT-only invariant.** After ANY training step that updates LoRA-B[0], T=1 ≠ base byte-exact. Always re-eval base on the same problem set in the same session.
- **modeling.py forward has TWO branches** (`use_cache=True` vs `False`). Recurrence-loop changes MUST go in BOTH. New cerebrum Do-Not-Repeat. Long-term: consolidate into one helper.

### Standing items (deferred)

- **Stage 2 (Mythic-Gemma4)** parked until Stage 1 ships a real LCB win (T=4 > base by ≥1 problem on LCB-30, sustained). Nothing close yet.
- **C.5 ACT halting head** deferred until at least one wrapper recipe shows T>1 productive on LCB.
- **v6L candidate (if v6K wins):** focal CE + EMA smoothing on focal_w (reduce gradient noise from per-batch focal weighting). Cheap follow-on.
- **Distillation candidate (if v6K flat):** fp16 self-distillation (unquantized DS-Coder-V2-Lite as teacher, same tokenizer, no NF4 noise). DS-V3.x/V4 ruled out — different tokenizer, no token-level KL alignment.

### Infrastructure (live)

- **Pandorum 5080 16GB stack** (Windows + WSL2 + RTX 5080 Blackwell sm_120) is validated for ckpt eval (NF4 base + bnb 0.49.2 + sidecar venv). Frees the local 3090 for ollama/dev work.
- **Pod backup hardening:** `scripts/sync_pod_to_solidpc.sh` mirrors v6K artifacts (checkpoints, run logs, wandb dir) to solidPC every 25 min via two redundant cron paths (system crontab + Claude session cron). Survives pod death.
- **CIFS mounts** on pandorum WSL: `/mnt/backup_models` → `\\solidpc\backup_models`, `/shared/dev` → `\\solidpc\dev`. Sticks across reboots via fstab + scheduled portproxy refresh.

---

## 10. Quickstart for the next session

```bash
# 1. Check v6K status (pod):
ssh -p 22024 root@ssh6.vast.ai \
  'tail -200 /workspace/mythic-rdt/eval_results/V6K_RUN.log; ps -p $(pgrep -f finetune_phase1)'

# 2. Wandb live: https://wandb.ai/mannix/mythic-rdt/runs/et2eektc

# 3. v6K artifacts auto-sync to solidPC every 25 min (system crontab + scripts/sync_pod_to_solidpc.sh).
#    Force-sync:
bash scripts/sync_pod_to_solidpc.sh
ls checkpoints/phase1_v6k_focal_anchored/

# 4. When v6K finishes — eval on pandorum 5080 (free GPU, NF4 base via CIFS):
ssh wsl 'cd /mnt/backup_models/Mythic-RDT && \
  source /home/claude_test/miniconda3/etc/profile.d/conda.sh && conda activate mythic-rdt && \
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_TOKEN=<token> HF_HUB_OFFLINE=1 \
  python -u scripts/humaneval_smoke.py \
    --base /mnt/backup_models/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v6k_focal_anchored \
    --first-iter-identity \
    --T-values 1 2 4 --max-loop-iters 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --gate-init-bias 0.0 --layerscale-clamp-max 0.5 \
    --quant nf4 --batch-size 2 --gen-tokens 384 \
    --lcb-limit 30 --lcb-difficulty medium --lcb-min-date 2024-10-01 \
    --output-json /home/claude_test/v6k_lcb30_pandorum.json'

# 5. VERIFY in the log: "[smoke] loaded 80 trainable tensors  missing=0 unexpected=0"
#    Anything else = partial load = STOP (bug-050).

# 6. Fallback eval on local 3090 (only if pandorum unavailable AND ollama not loaded):
nvidia-smi  # check free; if GPU >2GB used by non-ours, USE PANDORUM, do not force-unload ollama
```

---

*Memory entries cross-referenced: `project_v6h_final_verdict.md`, `project_v6i_rejected.md`, `project_v6k_design.md`, `project_v6a_post_fix_verdict.md`, `project_v6e_inference_path_bug.md`, `project_recurrence_root_cause_block_mode.md`, `project_phase1_v3_t1_validation.md`, `project_phase1_v4_anchored_corrected_verdict.md`, `feedback_smoke_max_loop_iters.md`, `feedback_pyc_purge_after_modeling_patch.md`, `feedback_init_from_checkpoint_pattern.md`, `project_dscoder_5x_blocker.md`. Bug log: `.wolf/buglog.json` entries `bug-049`, `bug-050`, `bug-054`.*
