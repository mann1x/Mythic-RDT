#!/bin/bash
# Overnight: v4-extended training (ckpt-400 -> ckpt-800) + split-smoke eval.
# Single-user, single-GPU, sequential. No parallelism.
set -uo pipefail
export PYTHONUNBUFFERED=1

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/OVERNIGHT_RUN.log
exec >> "$LOG" 2>&1

echo
echo "================================================================"
echo "[overnight] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# Always purge __pycache__ before any wrapper-importing run (bug-049).
rm -rf src/mythic_rdt/__pycache__

# ------------------------------------------------------------------
# Phase 1: v4-extended training (init-from v4-anchored ckpt-400)
# Same curriculum as v4-anchored, fresh optimizer + cosine LR.
# 400 steps * ~30-40s = ~3-3.5h on RTX A6000.
# ------------------------------------------------------------------
echo "[overnight] PHASE 1: v4-extended training"
echo "[overnight] $(date -u +%H:%M:%S) launching..."

nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

python scripts/finetune_phase1.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --output-dir checkpoints/phase1_v4_extended \
    --init-from-checkpoint checkpoints/phase1_v4_anchored \
    --quant nf4 \
    --max-loop-iters 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --lora-rank 8 --lora-targets q_proj o_proj \
    --curriculum-style v4-anchored \
    --curriculum-warmup-steps 80 \
    --curriculum-phase2-start 160 \
    --curriculum-phase3-start 280 \
    --kl-anchor-alpha 0.05 --kl-anchor-every 8 \
    --max-steps 400 \
    --save-steps 100 \
    --checkpoint-loop \
    --wandb-project mythic-rdt \
    --wandb-run-name phase1_v4_extended

TRAIN_RC=$?
echo "[overnight] $(date -u +%H:%M:%S) training exit_code=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "[overnight] TRAINING FAILED — skipping eval."
    exit "$TRAIN_RC"
fi

# ------------------------------------------------------------------
# Phase 2a: HE-only smoke at v4-extended final checkpoint
# Split into HE-only + LCB-only to avoid the HE→LCB contamination bug
# observed in 2026-04-27 verdict (T=1 LCB went 0% only when HE ran first).
# ------------------------------------------------------------------
rm -rf src/mythic_rdt/__pycache__
echo "[overnight] PHASE 2a: HE-20 eval at v4-extended final ckpt"
echo "[overnight] $(date -u +%H:%M:%S) launching..."

python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v4_extended \
    --T-values 1 2 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --quant nf4 --limit 20 --gen-tokens 384 --batch-size 4 \
    --output-json eval_results/v4_extended_he20.json

# ------------------------------------------------------------------
# Phase 2b: LCB-only smoke at v4-extended final checkpoint
# Separate process to avoid the HE→LCB state contamination.
# ------------------------------------------------------------------
rm -rf src/mythic_rdt/__pycache__
echo "[overnight] PHASE 2b: LCB-10 eval at v4-extended final ckpt"
echo "[overnight] $(date -u +%H:%M:%S) launching..."

python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v4_extended \
    --T-values 1 2 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --quant nf4 --limit 0 --gen-tokens 384 --batch-size 4 \
    --lcb-limit 10 --lcb-difficulty medium --lcb-min-date 2024-10-01 \
    --output-json eval_results/v4_extended_lcb10.json

echo "[overnight] PHASE 3: done"
echo "[overnight] $(date -u +%H:%M:%S) ALL DONE"
echo "[overnight] checkpoints in: checkpoints/phase1_v4_extended/"
echo "[overnight] eval JSON: eval_results/v4_extended_he20.json + v4_extended_lcb10.json"
