# Mythic-RDT — Working Notes

Build a **Recurrent-Depth Transformer (RDT)** wrapping an existing MoE base via OpenMythos blueprint + retrofit-recurrence curriculum (arXiv 2511.07384). Fine-tune only — no from-scratch training. Final model loadable via `transformers` `trust_remote_code=True`.

**Two-stage release strategy** (see `BASE_MODEL_ANALYSIS.md`):

- **Stage 1: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`** — coding-specialized DeepSeekMoE, native MLA + shared experts. Headline = HumanEval pass@1, base ~81 %. Target T=8 ≥ 86 %. Architecturally pure OpenMythos port. **Start here.** Spec: `BASE_DEEPSEEK_CODER_V2_LITE.md`. Published as `ManniX-ITA/Mythic-RDT-Coder-V2-Lite` (a.k.a. **Mythic-Coder**).
- **Stage 2: `ManniX-ITA/gemma-4-A4B-98e-v3-it`** — general-reasoning, gated on Stage 1 success. Headline = GPQA Diamond. Spec: `BASE_GEMMA4_98E_V3.md`.

**Read `MASTER_PLAN.md` before any task.**

## Key files & references

- `MASTER_PLAN.md` — feasibility verdict, port matrix, two-stage roadmap, risks, open questions.
- `BASE_MODEL_ANALYSIS.md` — Stage 1 / Stage 2 decision record + alternatives considered.
- `BASE_DEEPSEEK_CODER_V2_LITE.md` — Stage 1 architecture (MLA, shared experts, recurrent layer choice, memory/throughput).
- `BASE_GEMMA4_98E_V3.md` — Stage 2 architecture, mismatches with OpenMythos, SHA audit.
- `../CLAUDE.md` — parent backup_models project (Gemma 4 Surgery) inherited rules.
- `../scripts/expert_neuron_v4.json` — per-layer contribution data; pick `recurrent_layer_idx` from this for Stage 2.
- OpenMythos (MIT): https://github.com/kyegomez/OpenMythos
- Retrofit-Recurrence: arXiv 2511.07384
- DeepSeek-V2 paper: arXiv 2405.04434
- DeepSeek-Coder-V2 paper: https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/paper.pdf

## Critical rules (inherited from parent)

1. **Never write large files to `/tmp`** (tmpfs, 64 GB RAM). Weights, intermediates → project folder or `../google/`.
2. **lm-eval mandatory flags** — always `--use_cache <path>` + `--log_samples`. Sanity-check `samples_*.jsonl` after every run (empty / markdown-fence / <5-char junk). Distinct cache paths per stage: `<wd>/<bench>_cache/dscoderv2lite` (Stage 1), `<wd>/<bench>_cache/gemma4_98e` (Stage 2).
3. **HumanEval gotcha (Stage 1)**: NEVER eval HumanEval via `local-chat-completions` — chat mode wraps generations in ```` ```python ```` fences and the `exec(prompt+gen)` scorer fails to 0 % even on correct code (parent project bug-015). Use `local-completions` + `/v1/completions` (raw text). Sanity-check `samples_humaneval_*.jsonl` for fences before trusting any score.
4. **llama-server flags differ by stage**:
   - **Stage 1 (DS-Coder)**: standard chat-completions or completions, no `--reasoning-format` flag.
   - **Stage 2 (Gemma 4)**: requires `--reasoning-format deepseek --reasoning-budget 8192`. Without it, server crashes on first chemistry question.
5. **imatrix files must be archived** next to every quant + uploaded to HF. Pre-destroy checklist in parent `CLAUDE.md`. (Note: GGUF quantization is out of scope for Mythic-RDT v0 — recurrence runs in PyTorch only.)
6. **Use `lightseek` conda env** (`conda activate lightseek`) for transformers 5.5.0 / Gemma 4 tokenizer. NEVER touch `vllm` env. DS-Coder uses `trust_remote_code=True` — its own modeling/tokenizer code ships in the HF repo.
7. **GGUF gotcha** (Stage 2 only, deferred): expert `intermediate_size` must be divisible by 32 for Q4_K/Q8_0; otherwise tensors fall back to F16.

## Layout & conventions

- HF custom-code modeling files at repo root once stable.
- Conversion: `scripts/convert_*.py` — reproducible, args = base path + output path, no hardcoded paths. Two scripts: `convert_dscoder_to_mythic.py` (Stage 1) and `convert_gemma4_to_mythic.py` (Stage 2).
- Numbered probe directories under `experiments/` — each with a hypothesis-and-result note.
- Evals: `scripts/eval_*.sh` patterned after `../scripts/eval_gpqa_v3.sh`. Stage 1 strips the reasoning-format flag; Stage 2 keeps it.
- Reference scores to beat:
  - **Stage 1 (Mythic-Coder)**: DS-Coder-V2-Lite-Instruct HumanEval pass@1 ≈ 81 %; target T=8 ≥ 86 %.
  - **Stage 2 (Mythic-Gemma4)**: Gemma 4 26B-A4B-it GPQA Diamond = **75.25 %**; target T=8 ≥ 78 %.

## HF custom-code shape (non-negotiable)

Top-level project class is `MythicRDT` (`MythicRDTConfig` / `MythicRDTForCausalLM`). Stage-specific subclasses:

- `MythicRDTDeepseekV2Config` / `MythicRDTDeepseekV2ForCausalLM` — wraps DS-Coder-V2-Lite-Instruct (Stage 1).
- `MythicRDTGemma4Config` / `MythicRDTGemma4ForCausalLM` — wraps Gemma 4 26B-A4B (Stage 2).

Shared fields on `MythicRDTConfig`: `prelude_layers`, `coda_layers`, `recurrent_layer_idx`, `max_loop_iters`, `train_loop_iters`, `lti_init`, `depth_lora_rank`, `halting_strategy`. Loaders register via `AutoConfig.register("mythic_rdt_<base>", ...)` and `AutoModelForCausalLM.register(...)`.

End-user load:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "ManniX-ITA/Mythic-RDT-Coder-V2-Lite",  # or Mythic-RDT-Gemma4-26B-A4B-98e
    trust_remote_code=True,
)
out = model.generate(input_ids, max_new_tokens=128, n_loops=8)
```

Anything that breaks this shape is a bug.

## Won't port from OpenMythos (depends on stage)

| OpenMythos component | Stage 1 (DS-Coder-V2-Lite) | Stage 2 (Gemma 4 98e v3) |
|---|---|---|
| MLA (Multi-Latent Attention) | ✅ **native** | ❌ skip — keep GQA |
| DeepSeekMoE shared experts | ✅ **native** | ❌ skip — Gemma has no shared experts |
| ACT halting head | ❌ deferred to phase 7 — fixed `n_loops=T` for v0 | ❌ same |
| From-scratch training | ❌ never | ❌ never |
| MoE re-layout | ❌ inherit base topology | ❌ inherit base topology |

## Will add (fine-tune learnable, both stages)

- LTI params A, B (init `A = Diag(-exp(log_A))`, `B ≈ 0` + tiny noise) — Parcae stability.
- Depth-wise LoRA on Q/K/V/O + MoE router (rank 8–16, T distinct sets).
- Per-iteration LayerScale (init `1e-4`).
- Identity-biased gating: `h_{t+1} = h_t + ls·g·(LTI_inject + RecurrentBlock(h_t, e))`, `g` init via sigmoid bias = -3.

**Don't add LoRA to DS-Coder's shared experts** (Stage 1) — they're always-on across loop iterations and don't need depth differentiation.

## Stability mechanisms (mandatory, both stages)

- Spectral radius constraint: `A := Diag(-exp(log_A))` guarantees ρ(A) < 1.
- Curriculum: T=2 → T=4 → T=8 → T=16, mixed-T sampling per phase.
- fp32 RMSNorm in recurrence path (parent project bf16 NaN bug, Gemma 4 layers 11–29). Apply to both stages defensively.
- Optional stochastic depth on the loop.
- DS-Coder-specific (Stage 1): keep aux load-balance loss α₁ = 0.001 (or raise to 0.005 if expert utilization collapses).

## Eval methodology

Stage 1 (Mythic-Coder) — primary suite:
- HumanEval / HumanEval+ (pass@1) — primary headline benchmark
- MBPP+ (pass@1)
- LiveCodeBench
- GSM8K (math sanity)
- MMLU subset (general drift check)
- T=1 vs DS-Coder base (within 1 pp on HumanEval/MMLU). T=8 vs base (HumanEval +5 pp).
- **HumanEval MUST use `local-completions` + `/v1/completions`** (rule 3 above).

Stage 2 (Mythic-Gemma4) — primary suite:
- GPQA Diamond — primary headline
- HumanEval (with chat-completions OK for Gemma 4)
- MMLU subset
- llama-server requires `--reasoning-format deepseek --reasoning-budget 8192`.

Always `--use_cache` + `--log_samples`. Sanity-check sample files for empty / fence / junk.

## Common commands

Stage 1 fetch:

```bash
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --local-dir base/DeepSeek-Coder-V2-Lite-Instruct \
    --local-dir-use-symlinks False
```

Stage 2 fetch + verify (against SHA256 table in `BASE_GEMMA4_98E_V3.md`):

```bash
huggingface-cli download ManniX-ITA/gemma-4-A4B-98e-v3-it \
    --local-dir base/gemma-4-A4B-98e-v3-it \
    --local-dir-use-symlinks False
cd base/gemma-4-A4B-98e-v3-it
sha256sum model-*.safetensors
```

Run the canonical GPQA Diamond eval (Stage 2):

```bash
conda activate lightseek
bash ../scripts/eval_gpqa_v3.sh
```

Stage 1 eval (no `--reasoning-format` flag — see `scripts/eval_humaneval_dscoder.sh` once written; uses `local-completions`).

Check Caliber pre-commit hook is wired:

```bash
grep -q caliber .git/hooks/pre-commit && echo "hook-active" || echo "no-hook"
```

## Phase 0 sanity gate (before phase 1, EACH stage)

At T=1 with `gate=0`, output **must be bit-exact** with running the base's chosen middle layer once (fp32). T=4/8/16 untrained: no NaN/inf, no mode collapse on 100 prompts (mix of HumanEval-style code + FineWeb-Edu prose for Stage 1; FineWeb-Edu prose for Stage 2). Decision: if T=8 untrained drops PPL > 50 % or > 20 % gibberish, revisit middle-layer choice + gate init before phase 1.

## Personal context

- User does NOT have an Anthropic API key. Caliber must use `claude-cli` provider (seat-based), never API.
- Treat Caliber `init`/`regenerate` as token-expensive; confirm timing if rate limits tight.

## OpenWolf

Lives under parent OpenWolf-managed `backup_models/`. Follow `../.wolf/OPENWOLF.md`. Check `../.wolf/anatomy.md` before reading files; check `../.wolf/cerebrum.md` before generating code; log bugs to `../.wolf/buglog.json`.

## Auto-memory

User memory at `/root/.claude/projects/-srv-dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5-backup-models-Mythic-RDT/memory/`. Add Mythic-RDT entries as project develops.

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
