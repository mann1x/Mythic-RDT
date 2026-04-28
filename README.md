# Mythic-RDT

**Recurrent-Depth Transformer wrapping MoE bases (DeepSeek-Coder-V2-Lite-Instruct, Gemma 4 26B-A4B 98e v3)** via the OpenMythos blueprint and a retrofit-recurrence fine-tune curriculum (arXiv 2511.07384).

Status: **pre-alpha, Phase 1 fine-tune iterating.** No public release yet. Stage 1 wrapper is built and evaluated; v3-T1 validated (95 % HumanEval-20), v4 stacked but collapses on LiveCodeBench-medium, v5 dual-T training is in flight (2026-04-28). **Read [`STATUS.md`](STATUS.md) for the current architecture, training history, bugs, and next steps** — `MASTER_PLAN.md` (below) is the kickoff plan and does not reflect post-v3 state.

## What this is

A pretrained-frozen MoE base + a small set of new modules (LTI injection, identity-biased gating, per-loop LayerScale, depth-LoRA) wrapped in a `T`-iteration loop around one of the base's middle transformer blocks. At inference time the user picks `n_loops`:

- `n_loops = 1` ≈ baseline cost and quality of the underlying base.
- `n_loops = 4–8` ≈ stronger reasoning / deeper deliberation, ~4–8× compute, **no extra storage**.
- `n_loops = 16` ≈ deepest mode, depth-extrapolated beyond training.

Two-stage release plan:

1. **Stage 1 — Mythic-Coder** (`ManniX-ITA/Mythic-RDT-Coder-V2-Lite`): wraps `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`. **Headline = HumanEval pass@1**, base ≈ 81 %, target ≥ 86 % at T=8.
2. **Stage 2 — Mythic-Gemma4** (`ManniX-ITA/Mythic-RDT-Gemma4-26B-A4B-98e`): wraps `ManniX-ITA/gemma-4-A4B-98e-v3-it`. Gated on Stage 1 success. Headline = GPQA Diamond.

See `MASTER_PLAN.md` for the full roadmap, `BASE_MODEL_ANALYSIS.md` for the base-model decision record, and `BASE_DEEPSEEK_CODER_V2_LITE.md` / `BASE_GEMMA4_98E_V3.md` for per-stage architecture specs.

## Quickstart (development environment)

The project lives in its own conda environment, **`mythic-rdt`** — never use the `base` env or other shared envs.

### Option A: conda env (recommended)

```bash
# Create the env and install pinned deps
conda env create -n mythic-rdt -f environment.yml
conda activate mythic-rdt

# Install the project package in editable mode
pip install -e .
```

### Option B: vanilla venv with pinned deps

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# For fine-tune extras:
pip install -r requirements-train.txt
```

### Sanity check

```bash
pytest tests/ -q
# Expected: 19 passed
```

## Repo layout

```
Mythic-RDT/
├── README.md                          # this file
├── STATUS.md                          # current state (arch, training history v1-v5, bugs, scripts)
├── CLAUDE.md                          # working notes for Claude Code sessions
├── MASTER_PLAN.md                     # phased roadmap (kickoff intent — see STATUS.md for current state)
├── BASE_MODEL_ANALYSIS.md             # Stage 1 / Stage 2 decision
├── BASE_DEEPSEEK_CODER_V2_LITE.md     # Stage 1 architecture spec
├── BASE_GEMMA4_98E_V3.md              # Stage 2 architecture spec
├── pyproject.toml                     # package metadata + loose deps
├── requirements.txt                   # pinned runtime deps (Stage 1 inference + dev)
├── requirements-train.txt             # extra pinned deps for fine-tune (peft, trl, wandb, datasets)
├── environment.yml                    # conda env recipe
├── src/
│   └── mythic_rdt/
│       ├── __init__.py
│       └── recurrence.py              # LTI, IdentityBiasedGate, PerLoopLayerScale, DepthLoRA, RecurrenceCell
├── tests/
│   └── test_recurrence.py             # 19 unit tests; no base-model dependency
├── scripts/                           # convert / eval / phase-0 sanity scripts (TBD)
├── experiments/                       # numbered probes (TBD)
└── base/                              # base-model weights (gitignored)
    └── DeepSeek-Coder-V2-Lite-Instruct/   # Stage 1 base, fetched from HF
```

## Fetch the Stage 1 base

```bash
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --local-dir base/DeepSeek-Coder-V2-Lite-Instruct \
    --local-dir-use-symlinks False
```

Total ~31 GB (4 bf16 safetensors shards + custom code via `trust_remote_code=True`).

## Hardware expectations

| Stage | Phase 0 sanity | Phase 1 surgery | Phase 2 fine-tune (5 B tokens) |
|---|---|---|---|
| **Stage 1 (Mythic-Coder)** | 1× 24 GB GPU (4-bit base) | 1× 48 GB GPU | 4× 3090 (~25 days) or 2× H100 (~11 days, ~$600) |
| **Stage 2 (Mythic-Gemma4)** | 1× 24 GB GPU (4-bit base) | 1× 48 GB GPU | 4× 3090 (~50 days) or 4× H100 (~11 days, ~$2000) |

GGUF / llama.cpp support is **out of scope for v0** — recurrence runs in PyTorch only.

## License

- This codebase: **Apache-2.0**.
- Stage 1 derivative weights: governed by the **DeepSeek Model Agreement** (commercial use allowed) + Apache-2.0 for the recurrence code.
- Stage 2 derivative weights: governed by the **Gemma Terms of Use** + Apache-2.0 for the recurrence code.
- OpenMythos blueprint: MIT (kyegomez/OpenMythos), attribution preserved.

## Citations

```bibtex
@misc{mythicrdt2026,
  title  = {Mythic-RDT: retrofit-recurrence on MoE bases},
  author = {ManniX-ITA},
  year   = {2026},
  url    = {https://github.com/mann1x/Mythic-RDT}
}

@misc{deepseekv2,
  title  = {DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model},
  author = {DeepSeek-AI},
  year   = {2024},
  eprint = {2405.04434},
  archivePrefix = {arXiv}
}

@misc{deepseekcoderv2,
  title  = {DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence},
  author = {DeepSeek-AI},
  year   = {2024},
  url    = {https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/paper.pdf}
}

@misc{retrofitrecurrence2025,
  title  = {Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence},
  year   = {2025},
  eprint = {2511.07384},
  archivePrefix = {arXiv}
}

@misc{openmythos2026,
  title  = {OpenMythos: an open-source PyTorch reconstruction of Claude Mythos},
  author = {Kye Gomez},
  year   = {2026},
  url    = {https://github.com/kyegomez/OpenMythos}
}
```
