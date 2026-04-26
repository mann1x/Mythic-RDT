# Mythic-Gemma4 — Working Notes

Build a **Recurrent-Depth Transformer (RDT)** wrapping Gemma 4 26B-A4B (MoE, 98 experts/layer, top-8) using OpenMythos blueprint + retrofit-recurrence curriculum (arXiv 2511.07384). Fine-tune only — no from-scratch training. Final model loadable via `transformers` `trust_remote_code=True`.

**Read `MASTER_PLAN.md` before any task.** See `BASE_MODEL_ANALYSIS.md` for base model decision (98e-v3 vs 128e fallback).

## Key files & references

- `MASTER_PLAN.md` — feasibility verdict, port matrix, phased roadmap, risks, open questions.
- `BASE_MODEL_ANALYSIS.md` — SHA256 audit, candidate comparison; **base = `ManniX-ITA/gemma-4-A4B-98e-v3-it`**.
- `../CLAUDE.md` — parent backup_models project (Gemma 4 Surgery) inherited rules.
- `../scripts/expert_neuron_v4.json` — per-layer contribution data; pick `recurrent_layer_idx` from this.
- OpenMythos (MIT): https://github.com/kyegomez/OpenMythos
- Retrofit-Recurrence: arXiv 2511.07384

## Critical rules (inherited from parent)

1. **Never write large files to `/tmp`** (tmpfs, 64 GB RAM). Weights and intermediates → project folder or `../google/`.
2. **lm-eval mandatory flags** — always `--use_cache <path>` + `--log_samples`. Sanity-check `samples_*.jsonl` after every run (empty / markdown-fence / <5-char junk). See `../CLAUDE.md` for full ruleset.
3. **llama-server for Gemma 4** requires `--reasoning-format deepseek --reasoning-budget 8192`. Without it, server crashes on first chemistry question.
4. **imatrix files must be archived** next to every quant + uploaded to HF. Pre-destroy checklist in parent `CLAUDE.md`.
5. **Use `lightseek` conda env** (`conda activate lightseek`) for transformers 5.5.0 / Gemma 4 tokenizer. NEVER touch `vllm` env.
6. **GGUF gotcha**: expert `intermediate_size` must be divisible by 32 for Q4_K/Q8_0; otherwise tensors fall back to F16.

## Layout & conventions

- HF custom-code modeling files at repo root once stable.
- Conversion: `scripts/convert_*.py` — reproducible, args = base path + output path, no hardcoded paths.
- Numbered probe directories under `experiments/` — each with a hypothesis-and-result note.
- Evals: `scripts/eval_*.sh` patterned after `../scripts/eval_gpqa_v3.sh`.
- Reference score to beat: Gemma 4 26B-A4B-it on GPQA Diamond = **75.25%**.

## HF custom-code shape (non-negotiable)

Top-level subclasses or composes `Gemma4ForCausalLM`. Recurrence inside `forward()` of new class. `MythicGemma4Config` extends `Gemma4Config` adding: `prelude_layers`, `coda_layers`, `recurrent_layer_idx`, `max_loop_iters`, `train_loop_iters`, `lti_init`, `depth_lora_rank`, `halting_strategy`. Loaders register via `AutoConfig.register("mythic_gemma4", ...)` and `AutoModelForCausalLM.register(...)`. End-user load:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "ManniX-ITA/Mythic-Gemma4-26B-A4B", trust_remote_code=True
)
```

Anything that breaks this shape is a bug.

## Won't port from OpenMythos

- **MLA** — Gemma 4 uses GQA; conversion needs pretrain budget we don't have. Keep GQA.
- **From-scratch training scripts** — irrelevant.
- **MoE re-layout** — inherit Gemma 4's 98 experts × top-8.
- **ACT halting head** — v0 uses fixed `n_loops=T`; ACT is phase-6 polish.

## Will add (fine-tune learnable)

- LTI params A, B (init `A = Diag(-exp(log_A))`, `B ≈ 0` + tiny noise) — Parcae stability.
- Depth-wise LoRA on Q/K/V/O + MoE router (rank 8–16, T distinct sets).
- Per-iteration LayerScale (init `1e-4`).
- Identity-biased gating: `h_{t+1} = h_t + ls·g·(LTI_inject + RecurrentBlock(h_t, e))`, `g` init via sigmoid bias = -3.

## Stability mechanisms (mandatory)

- Spectral radius constraint: `A := Diag(-exp(log_A))` guarantees ρ(A) < 1.
- Curriculum: T=2 → T=4 → T=8 → T=16, mixed-T sampling per phase.
- fp32 RMSNorm in recurrence path (parent project bf16 NaN bug, layers 11-29).
- Optional stochastic depth on the loop.

## Eval methodology

GPQA Diamond + HumanEval + MMLU subset. Always `--use_cache` + `--log_samples`. Always `--reasoning-format deepseek --reasoning-budget 8192` on llama-server. Sanity-check sample files. Compare:
- T=1 vs Gemma 4 base (must be ≥ within 2%).
- T=4, T=8, T=16 vs base (target: T=8 ≥ base, T=16 > base on hard tasks).

## Common commands

Run the canonical GPQA Diamond eval:

```bash
conda activate lightseek
bash ../scripts/eval_gpqa_v3.sh
```

Check Caliber pre-commit hook is wired:

```bash
grep -q caliber .git/hooks/pre-commit && echo "hook-active" || echo "no-hook"
```

## Phase 0 sanity gate (before phase 1)

At T=1 with `gate=0`, output **must be bit-exact** with running Gemma 4's chosen middle layer once (fp32). T=4/8/16 untrained: no NaN/inf, no mode collapse on 100 FineWeb-Edu prompts. Decision: if T=8 untrained drops PPL > 50% or > 20% gibberish, revisit middle-layer choice + gate init before phase 1.

## Personal context

- User does NOT have an Anthropic API key. Caliber must use `claude-cli` provider (seat-based), never API.
- Treat Caliber `init`/`regenerate` as token-expensive; confirm timing if rate limits tight.

## OpenWolf

Lives under parent OpenWolf-managed `backup_models/`. Follow `../.wolf/OPENWOLF.md`. Check `../.wolf/anatomy.md` before reading files; check `../.wolf/cerebrum.md` before generating code; log bugs to `../.wolf/buglog.json`.

## Auto-memory

User memory at `/root/.claude/projects/-srv-dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5-backup-models-Mythic-Gemma4/memory/`. Add Mythic-Gemma4 entries as project develops.

## Caliber

Pre-commit hook auto-syncs agent configs (`.claude/settings.json`, `.claude/hooks/caliber-*.sh` already wired). Check with `grep -q caliber .git/hooks/pre-commit`. If missing, run `/setup-caliber`.

<!-- caliber:managed:pre-commit -->
## Before Committing

**IMPORTANT:** Before every git commit, you MUST ensure Caliber syncs agent configs with the latest code changes.

First, check if the pre-commit hook is already installed:
```bash
grep -q "caliber" .git/hooks/pre-commit 2>/dev/null && echo "hook-active" || echo "no-hook"
```

- If **hook-active**: the hook handles sync automatically — just commit normally. Tell the user: "Caliber will sync your agent configs automatically via the pre-commit hook."
- If **no-hook**: run Caliber manually before committing:
  1. Tell the user: "Caliber: Syncing agent configs with your latest changes..."
  2. Run: `/usr/local/bin/caliber refresh && git add CLAUDE.md .claude/ .cursor/ .cursorrules .github/copilot-instructions.md .github/instructions/ AGENTS.md CALIBER_LEARNINGS.md .agents/ .opencode/ 2>/dev/null`
  3. After it completes, briefly tell the user what Caliber updated. Then proceed with the commit.

**Valid `/usr/local/bin/caliber refresh` options:** `--quiet` (suppress output) and `--dry-run` (preview without writing). Do not pass any other flags — options like `--auto-approve`, `--debug`, or `--force` do not exist and will cause errors.

**`/usr/local/bin/caliber config`** takes no flags — it runs an interactive provider setup. Do not pass `--provider`, `--api-key`, or `--endpoint`.

If `/usr/local/bin/caliber` is not found, tell the user: "This project uses Caliber for agent config sync. Run /setup-caliber to get set up."
<!-- /caliber:managed:pre-commit -->

<!-- caliber:managed:learnings -->
## Session Learnings

Read `CALIBER_LEARNINGS.md` for patterns and anti-patterns learned from previous sessions.
These are auto-extracted from real tool usage — treat them as project-specific rules.
<!-- /caliber:managed:learnings -->

<!-- caliber:managed:model-config -->
## Model Configuration

Recommended default: `claude-sonnet-4-6` with high effort (stronger reasoning; higher cost and latency than smaller models).
Smaller/faster models trade quality for speed and cost — pick what fits the task.
Pin your choice (`/model` in Claude Code, or `CALIBER_MODEL` when using Caliber with an API provider) so upstream default changes do not silently change behavior.

<!-- /caliber:managed:model-config -->

<!-- caliber:managed:sync -->
## Context Sync

This project uses [Caliber](https://github.com/caliber-ai-org/ai-setup) to keep AI agent configs in sync across Claude Code, Cursor, Copilot, and Codex.
Configs update automatically before each commit via `/usr/local/bin/caliber refresh`.
If the pre-commit hook is not set up, run `/setup-caliber` to configure everything automatically.
<!-- /caliber:managed:sync -->
