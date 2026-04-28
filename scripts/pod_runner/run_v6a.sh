#!/bin/bash
# v6A: dual-T training with margin + distill, on top of architectural fix
# (first-iter-identity: t=0 iter is unconditionally identity, T=1 wrapper
# output ≡ base byte-for-byte).
#
# FRESH init (NOT init-from-checkpoint): v3-T1/v4/v5 ckpts have
# first_iter_identity=False baked into their LoRA-B / LTI / gate weights;
# loading them into a first_iter_identity=True wrapper would conflict.
#
# Same recipe as v5 (dual-T 1↔4, margin α=0.1 nats=0.02, distill α=0.05,
# KL-anchor α=0.02 every-8). 200 steps × ~115s ≈ 6.5h on A6000.
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/V6A_RUN.log
exec >> "$LOG" 2>&1

echo
echo "================================================================"
echo "[v6a] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[v6a] WANDB_API_KEY length=${#WANDB_API_KEY:-0}"
echo "================================================================"

echo "[v6a] PHASE 1: dual-T training with first-iter-identity (margin + distill)"
echo "[v6a] FRESH init (no --init-from-checkpoint), 200 steps, ~6.5h ETA"

python scripts/finetune_phase1.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --output-dir checkpoints/phase1_v6a_dual_t \
    --quant nf4 \
    --max-loop-iters 4 \
    --first-iter-identity \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --lora-rank 8 \
    --curriculum-style v4-anchored \
    --curriculum-warmup-steps 80 --curriculum-phase2-start 160 --curriculum-phase3-start 280 \
    --margin-alpha 0.1 --margin-nats 0.02 \
    --distill-alpha 0.05 \
    --kl-anchor-alpha 0.02 --kl-anchor-every 8 \
    --max-steps 200 --save-steps 50 \
    --checkpoint-loop \
    --wandb-project mythic-rdt \
    --wandb-run-name phase1_v6a_dual_t

TRAIN_RC=$?
echo "[v6a] $(date -u +%H:%M:%S) training exit_code=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "[v6a] TRAINING FAILED — skipping eval."
    exit "$TRAIN_RC"
fi

echo "[v6a] PHASE 2a: HE-20 eval at v6A final ckpt"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v6a_dual_t \
    --first-iter-identity \
    --T-values 1 2 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --quant nf4 --limit 20 --gen-tokens 384 --batch-size 4 \
    --output-json eval_results/v6a_he20.json

echo "[v6a] PHASE 2b: LCB-10 eval (separate process)"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v6a_dual_t \
    --first-iter-identity \
    --T-values 1 2 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --quant nf4 --limit 0 --gen-tokens 384 --batch-size 4 \
    --lcb-limit 10 --lcb-difficulty medium --lcb-min-date 2024-10-01 \
    --output-json eval_results/v6a_lcb10.json

echo "[v6a] $(date -u +%H:%M:%S) ALL DONE"
