# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Last updated: 2026-05-10

## User Preferences

### [2026-04-26] solidPC repo is source of truth — mirror pod edits back

**Rule:** After any pod-side edit, immediately `scp` it back to solidPC at the same relative path. Better: edit on solidPC first, push up.

**Why:** Pods are spot-interruptible and ephemeral. Pod-only edits evaporate.

**How to apply:** No pod-only edits except throwaway debug. Mirror configs, scripts, and `base/.../modeling_*.py` patches.

### [2026-04-26] Fine-tune scripts must be pause/resume-safe

**Rule:** Phase 2+ scripts handle SIGINT/SIGTERM, auto-checkpoint every 500-1000 steps, support `--resume <path>` restoring model + optimizer + scheduler + RNG + step + dataset position.

**Why:** Vast.ai pods get killed mid-run; non-resumable scripts waste hours.

**How to apply:** Prefer `transformers.Trainer` (built-in checkpointing). Save to persistent disk, never /tmp.

## Key Learnings

- **Project:** Mythic-RDT

### [2026-04-26] DS-Coder-V2-Lite + transformers 5.x compat

Base ships `modeling_deepseek.py` for tf 4.x. tf 5.6 fails with `ImportError: is_torch_fx_available`. Patched line 56 with try/except fallback. **Side effect:** HF caches dynamic-code modules under `$HF_HOME/modules/transformers_modules/<MODEL>/` — after patching `base/`, `rm -rf` that dir or the patch is invisible.

### [2026-04-26] Recurrence cell dtype must match base dtype

When wrapping bf16 base, cast `RecurrenceCell` (default fp32): `self.recurrence.to(dtype=base_dtype)` after construction. fp32 reintroduced selectively around RMSNorm inside loop body per stability rule; LTI/gate/LayerScale params match base dtype.

### [2026-04-26] Phase 0 bit-exact gate semantics

`force_gate_zero=True, T=1` bypasses recurrence (`h <- block_out`) → bit-exact with `embed → layer[0] → layer[recurrent_idx] → layer[-1] → norm → lm_head`. Verified for `recurrent_layer_idx ∈ {10,13,16}` on bf16 CPU; max_abs_diff = 0.0. Default-gate T=1 intentionally diffs vs reference (sigmoid(-3)·1e-4 ≈ 5e-6 near-identity in `h`, coda sees `h_prelude`).

### [2026-04-29] kl_anchor + margin loss = incoherent tug-of-war

Dual-T(1,4) with both `kl-anchor-alpha 0.5` AND `margin-alpha 0.1` creates opposing objectives: kl_anchor pulls T=4 → base@T=1; margin pushes T=4 NLL below T=1. Optimizer alternates, never satisfies both. Effect: small CE win at T=4 but distribution tracks base; LCB T=4 starts harmful, slowly recovers.

**Lesson v6I+:** Pick ONE — distillation-only, margin-only with reverse anchor, or teacher distillation. Also: ramp T=4 curriculum smoothly (0%@step80 → 50%@step200) to avoid phase-boundary loss bumps.

## Do-Not-Repeat

### [2026-04-26] After patching base/.../modeling_*.py, clear HF dynamic-module cache

HF copies file into `$HF_HOME/modules/transformers_modules/<MODEL>/` at first load and never refreshes. If patch import still fails, `rm -rf` that dir.

### [2026-04-29] NEVER claim "T=1 = base by first_iter_identity" after training

`first_iter_identity=True` makes recurrence ADD zero at t=0 — does NOT make BLOCK forward = base. Recurrent block carries DepthLoRA on q_proj/o_proj. At INIT, LoRA-B[0]=0 → T=1=base byte-exact. After ANY training step, T=1 ≠ base.

Bitten twice: (1) v6A ckpt-200 LCB-30 T=1=10% vs base 26.7%; (2) v6H ckpt-100 assumed T=1=base.

**How to apply:** Always re-eval base on the SAME problem set in the same session as any trained-checkpoint T=1 eval. Invariant only holds at INIT.

## Decision Log

### [2026-04-26] V0 vendoring: keep trust_remote_code against base/

Hold local patched `base/DeepSeek-Coder-V2-Lite-Instruct/modeling_deepseek.py`; no second copy in `src/mythic_rdt/` for v0. **Revisit before Phase 4** — at publish, vendor clean copy under `src/mythic_rdt/deepseek_v2/`.

### [2026-04-26] Stage 1 recurrent_layer_idx = 10 (user override of rec=13)

Probe across layers 10/13/16 × T=1/4/8 all PASS gate. T=8 gibberish: 10=7%, 13=9%, 16=7% (σ≈2.6pp). Analyst recommended 13; user picked **layer 10** (lowest T=1 gibberish 6%). Locked as Stage 1 default. 13/16 are fallbacks.

### [2026-04-26] vast.ai pod provisioning notes

- `vastai/pytorch:2.6.0-cuda-12.6-py311-22.04-ipv6` doesn't exist. Use `cuda-12.4.1-auto` / `cuda-12.6.3-auto` or versioned `2.11.0-cu128-cuda-12.9-mini-py311-2026-04-15`.
- `vastai update instance <id> --image <new>` swaps images without recreate; failed pulls retry.
- Vast Python at `/venv/main/bin/activate`. No conda. Python 3.12, torch ≥2.10.
- HF CLI is now `hf` (not `huggingface-cli`).
- 200GB stopped disk = ~$8/day; 100GB = ~$4/day. Pick smallest fit.
- DS-Coder bf16 = 5291 tensors; `.to(cuda)` is ~5min Python-bound. Use `device_map="cuda"` in `from_pretrained`.