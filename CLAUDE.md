# Mythic-Gemma4 — Working Notes

Project goal: build a **Recurrent-Depth Transformer (RDT)** that wraps Gemma 4 26B-A4B (and possibly other Gemma 4 releases) using the OpenMythos architectural blueprint, recovering quality via fine-tuning only — no from-scratch training.

The final model **must be loadable from `transformers`** via the custom-code path (`trust_remote_code=True`), like DeepSeek-V2 / Qwen3-MoE were before upstreaming.

**Read `MASTER_PLAN.md` before starting any task.** It contains the feasibility verdict, port/no-port matrix, phased roadmap, and open risks.

## Source repos and references

- OpenMythos (MIT, kyegomez): https://github.com/kyegomez/OpenMythos — RDT reconstruction, blueprint for prelude / recurrent block / coda / LTI injection / depth-LoRA / MoE.
- Article: https://www.marktechpost.com/2026/04/19/meet-openmythos-...
- Retrofit-Recurrence paper (arXiv 2511.07384): "Teaching Pretrained LMs to Think Deeper with Retrofitted Recurrence" — the *exact* technique we'll use to convert Gemma 4 into an RDT via curriculum fine-tune, not from scratch.
- Gemma 4 26B-A4B base: `google/gemma-4-26B-A4B-it` (in this server: `../google/gemma-4-26B-A4B-it`)
- Prior project working notes: `../CLAUDE.md` (parent backup_models — Gemma 4 Surgery Project)

## Critical rules (carry over from parent project)

1. **Never write large files to /tmp.** /tmp is tmpfs (64 GB RAM). All weights, intermediates, scores → persistent disk under this project folder or `../google/`.
2. **lm-eval is mandatory** — always pass `--use_cache <path>` and `--log_samples`, always sanity-check `samples_*.jsonl` after the run (empty / markdown-fence / <5-char junk). See `../CLAUDE.md` for the full rules.
3. **llama-server for Gemma 4 eval needs `--reasoning-format deepseek --reasoning-budget 8192`.** Without `--reasoning-budget`, server crashes mid-eval on chemistry questions.
4. **imatrix.dat must be preserved** — every quant gets its imatrix saved next to it and uploaded to HF. Pre-destroy checklist in parent CLAUDE.md.
5. **Use lightseek conda env** (`conda activate lightseek`) for transformers 5.5.0 / Gemma 4 tokenizer. NEVER touch the `vllm` env.
6. **GGUF gotcha**: expert `intermediate_size` must be divisible by 32 for Q4_K/Q8_0, otherwise quants fall back to F16.

## Project-specific conventions

- Code under `src/mythic_gemma4/` — Python package, importable.
- HF custom-code modeling files at the repo root once they're stable: `configuration_mythic_gemma4.py`, `modeling_mythic_gemma4.py` (so they ship with the HF checkpoint).
- Conversion scripts under `scripts/convert_*.py` — reproducible, take base model path + output path, no hardcoded paths.
- Feasibility / sanity probes under `experiments/<NN>_<name>/` — numbered so order is obvious. Always include a `README.md` with hypothesis + result.
- Evals via `scripts/eval_*.sh` patterned after `../scripts/eval_gpqa_v3.sh`.
- Reference scores to beat: Gemma 4 26B-A4B-it on GPQA Diamond = 75.25%.
- **Base model lives at `base/gemma-4-A4B-98e-v3-it/`** (downloaded from HF `ManniX-ITA/gemma-4-A4B-98e-v3-it`). Do NOT use the local `../google/gemma-4-A4B-98e-hybrid/` intermediate — it's not bit-identical to the published artifact. See `BASE_MODEL_ANALYSIS.md` for SHA256s.

## Porting to transformers — non-negotiable shape

The model class subclasses or composes `Gemma4ForCausalLM`. The recurrence is implemented inside `forward()` of the new top-level class. Custom config (`MythicGemma4Config`) extends `Gemma4Config` with the RDT fields (`prelude_layers`, `coda_layers`, `recurrent_layer_idx`, `max_loop_iters`, `train_loop_iters`, `lti_init`, `depth_lora_rank`, `halting_strategy`, ...). Loaders use `AutoConfig.register("mythic_gemma4", ...)` and `AutoModelForCausalLM.register(...)`. End-user load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "ManniX-ITA/Mythic-Gemma4-26B-A4B", trust_remote_code=True
)
```

Anything that breaks this shape is a bug.

## What WON'T be ported from OpenMythos (and why)

- **MLA (Multi-Latent Attention)** — Gemma 4 uses GQA; converting GQA→MLA needs a learned KV compression that we can't do without a meaningful pretrain budget. Keep GQA. Note in the plan, revisit only if a small fine-tune can absorb the conversion.
- **From-scratch training scripts (`training/3b_fine_web_edu.py` etc.)** — irrelevant; we only fine-tune.
- **Some MoE knobs** — we inherit Gemma 4's expert layout (128 experts, top-8). Don't try to re-layout experts.
- **ACT halting head** — first release uses **fixed `n_loops=T`**. ACT is a phase-2 polish; learning a halting head from a fine-tune budget is non-trivial.

## What WILL be added (needs fine-tuning to learn)

- LTI injection params A, B (init: A = `Diag(-exp(log_A))` with small log_A, B = small near-zero) — Parcae stability.
- Depth-wise LoRA adapters on Q/K/V/O and MoE router (rank 8–16, T distinct sets).
- Per-iteration LayerScale (init very small, e.g., 1e-4) — protect Gemma's representations during early curriculum.
- Optional: depth-aware router LoRA only (cheap; let the experts' weights stay frozen).

## Stability mechanisms (mandatory, from retrofit-recurrence + LayerScale literature)

- **Identity-biased gating**: `h_{t+1} = h_t + g · [LTI_inject + RecurrentBlock(h_t, e)]` with `g` initialized near 0 (negative bias on the gate logit). Prevents drift in early training.
- **Spectral radius constraint** on A (Parcae): `A := Diag(-exp(log_A))` parameterization guarantees ρ(A) < 1.
- **Curriculum**: train at T=2, then T=4, then T=8, then target T (16). Mixed-T sampling within each phase.
- **Dropout / stochastic depth on the loop**: optionally drop a recurrence step occasionally to teach robustness.

## Eval methodology

Same as parent project. GPQA Diamond + HumanEval + MMLU subset. Always with `--use_cache` and `--log_samples`. Always with `--reasoning-format deepseek --reasoning-budget 8192` on llama-server. Sanity check sample files. Compare:
- Mythic-Gemma4 at T=1 vs Gemma 4 base (must be ≥ within 2%).
- Mythic-Gemma4 at T=4, T=8, T=16 vs Gemma 4 base (target: T=8 ≥ base, T=16 > base on hard tasks).

## OpenWolf

This project lives under the parent OpenWolf-managed `backup_models/`. Follow `../.wolf/OPENWOLF.md` rules.

## Auto-memory

User-level memory is at `/root/.claude/projects/-srv-dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5-backup-models/memory/`. Add Mythic-Gemma4 entries there as the project develops.
