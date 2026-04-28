# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-04-26

## User Preferences

<!-- How the user likes things done. Code style, tools, patterns, communication. -->

### [2026-04-26] Always keep solidPC repo synced with pod-side script changes

**Rule:** Whenever I edit a script ON the vast.ai pod, immediately mirror the change back to solidPC `/srv/dev-disk-by-uuid-.../backup_models/Mythic-RDT/`. solidPC is the source of truth; the pod is ephemeral.

**Why:** Pod is destroyable / spot-interruptible. Edits made only on the pod evaporate when it dies. Lost work, lost reproducibility.

**How to apply:** After every pod-side edit (script, config, base/.../modeling_*.py patch), `scp` it back to solidPC at the same relative path. Better yet: edit on solidPC first, then `scp` up. Avoid pod-only edits unless they are throwaway debug.

### [2026-04-26] Fine-tuning scripts MUST be pause/resume-safe

**Rule:** Every fine-tune script (Phase 2+) must:
- Handle `SIGINT` (Ctrl+C) and `SIGTERM` cleanly via signal handlers.
- On signal: flush + save model state, optimizer state, scheduler state, RNG state, current step, dataset position.
- Auto-checkpoint every N steps (configurable, default ~500-1000 steps for long curriculums).
- Support `--resume <path>` flag that restores state and continues from the saved step.

**Why:** vast.ai pods can be spot-interrupted with seconds notice. Long curriculum runs (Phase 2 = 5 B tokens) will be killed mid-run multiple times. A non-resumable script wastes hours of compute per kill.

**How to apply:** Use `transformers.Trainer` if possible (has built-in checkpointing), or hand-roll with `torch.save({"model": ..., "optimizer": ..., "scheduler": ..., "step": ..., "rng": ...}, path)` and `torch.load` matching `--resume`. Save to persistent disk, NOT to /tmp.

## Key Learnings

- **Project:** Mythic-RDT

### [2026-04-26] DS-Coder-V2-Lite-Instruct + transformers 5.x compat

The base ships `modeling_deepseek.py` written for transformers 4.x. Loading it via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` on transformers 5.6 fails with `ImportError: is_torch_fx_available`. Patched `base/DeepSeek-Coder-V2-Lite-Instruct/modeling_deepseek.py` line 56 with a try/except fallback. All other 5.x imports (`Cache`, `DynamicCache`, `_prepare_4d_causal_attention_mask`, `is_flash_attn_2_available`, `ALL_LAYERNORM_LAYERS`, `is_torch_greater_or_equal_than_1_13`) survive 5.6 unchanged.

**Side effect**: HF caches dynamic-code modules under `/srv/dev-disk-by-uuid-.../huggingface/modules/transformers_modules/<MODEL>/`. After patching `base/`, `rm -rf` that cache dir or the patch is invisible.

### [2026-04-26] Recurrence cell dtype must match base dtype

When wrapping a bf16 base, `RecurrenceCell` (default fp32) must be cast: `self.recurrence.to(dtype=base_dtype)` after construction. Phase-1+ work will reintroduce fp32 selectively around RMSNorm inside the loop body (per MASTER_PLAN.md stability rule), but the LTI Linear / gate / LayerScale params themselves should match the base dtype to avoid matmul rejections.

### [2026-04-26] Phase 0 bit-exact gate semantics

`MythicRDTDeepseekV2ForCausalLM(force_gate_zero=True, T=1)` is implemented as a *plumbing-correctness* mode that bypasses the recurrence cell entirely (`h <- block_out`). At T=1 it is bit-exact with `embed -> layer[0] -> layer[recurrent_idx] -> layer[-1] -> norm -> lm_head`. Verified for `recurrent_layer_idx in {10, 13, 16}` on bfloat16 CPU; max_abs_diff = 0.0 exactly.

A *separate* secondary check (gate=default, T=1) intentionally produces large diffs vs the reference because at gate sigmoid(-3)*1e-4 ≈ 5e-6 the loop is near-identity in `h` (i.e. it passes the prelude output through almost unchanged), so the coda sees `h_prelude` instead of `block_out`. That is the retrofit-recurrence design: at init the loop is the identity and learns to deviate during fine-tune. Do not interpret the secondary diff as a wrapper bug.

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

### [2026-04-26] After patching base/.../modeling_*.py, clear HF dynamic-module cache

Editing `base/<MODEL>/modeling_*.py` is not enough -- HF copies the file into `~/.cache/huggingface/modules/transformers_modules/<MODEL>/` (or `$HF_HOME/modules/...`) at first load and never refreshes it. If you patch the source and the import still fails the same way, `rm -rf $HF_HOME/modules/transformers_modules/<MODEL>` and re-run.

## Decision Log

<!-- Significant technical decisions with rationale. Why X was chosen over Y. -->

### [2026-04-26] V0 vendoring strategy: keep trust_remote_code against base/

Per MASTER_PLAN.md §5 phase 0 step "decide vendoring strategy" and §7 open question 7. **Decision: keep trust_remote_code=True** against the local `base/DeepSeek-Coder-V2-Lite-Instruct/` for v0. We hold a local copy of `modeling_deepseek.py` already (and have patched it for transformers 5.x); no need to vendor a second copy into `src/mythic_rdt/`. **Revisit before Phase 4 (custom-code package release)** -- at publish time we should vendor a clean patched copy under `src/mythic_rdt/deepseek_v2/` so the released model doesn't need users to keep our base/ patch around.

### [2026-04-26] Stage 1 recurrent_layer_idx = 10 (user override of analyst rec=13)

`experiments/01_phase0_probe` ran 100-prompt probe (50 HumanEval + 50 wikitext-2) on vast.ai A6000 across layers 10/13/16 × T=1/4/8. All three PASS the MASTER_PLAN.md §5 gate (PPL ratio = 1.000, gibberish ≤ 11%). Per-layer T=8 gibberish: layer 10 = 7%, layer 13 = 9%, layer 16 = 7%. Std-error at n=100 ≈ 2.6 pp so 7 vs 9 is roughly 1σ.

Analyst recommended layer 13 (architectural middle, default). User overrode to **layer 10** -- closer to input, lowest T=1 gibberish (6%), ties for lowest T=8 (7%). Locked into `MythicRDTDeepseekV2Config(recurrent_layer_idx=10)` as the Stage 1 default. Layers 13 and 16 stay as known-good fallbacks if Phase 1 fine-tune underperforms.

### [2026-04-26] vast.ai pod provisioning learnings

- `vastai/pytorch:2.6.0-cuda-12.6-py311-22.04-ipv6` does NOT exist in Docker Hub. Real tag pattern is `2.X.Y-cuRRR-cuda-VV.W-mini-pyXX-DATE` or the auto variants `cuda-VV.W.Z-auto`. Use `cuda-12.4.1-auto` or `cuda-12.6.3-auto` to match host driver, or pick a versioned tag like `2.11.0-cu128-cuda-12.9-mini-py311-2026-04-15`.
- `vastai update instance <id> --image <new>` lets you swap images without destroying/recreating. Failed-pull instances retry on image swap.
- Vast image's Python is at `/venv/main/bin/activate` (sourced manually); no `conda` is on PATH. Python 3.12, torch >= 2.10 ships with the auto images.
- HF CLI in `huggingface_hub >= 1.x` is `hf` not `huggingface-cli`. Old name not aliased.
- Storage cost while STOPPED on a 200 GB disk = ~$0.33/hr ≈ $8/day. Storage on a 100 GB disk would be ~$4/day. For "stop and reuse later" workflows pick the smallest disk that fits the working set.
- DS-Coder bf16 base is 5291 separate weight tensors. `nn.Module._apply` walks them in pure Python, so post-load `.to(cuda)` is ~5 minutes (not GPU-bound). Use `device_map="cuda"` in `from_pretrained` to avoid the second copy entirely.
