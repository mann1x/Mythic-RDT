#!/bin/bash
# v5: dual-T training with margin + distill loss.
# init from v3-T1 ckpt-400 (validated T=1 wrapper at 95% HE).
# Slower than v4 (~120s/step → 200 steps ≈ 7h, 400 steps ≈ 13h).
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/V5_RUN.log
exec >> "$LOG" 2>&1

echo
echo "================================================================"
echo "[v5] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[v5] WANDB_API_KEY length=${#WANDB_API_KEY:-0}"
echo "================================================================"

echo "[v5] PHASE 1: dual-T training (margin + distill)"
echo "[v5] init-from v3-T1 ckpt-400, 200 steps, ~7h ETA"

python scripts/finetune_phase1.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --output-dir checkpoints/phase1_v5_margin_distill \
    --init-from-checkpoint checkpoints/phase1_v3_t1 \
    --quant nf4 \
    --max-loop-iters 4 \
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
    --wandb-run-name phase1_v5_margin_distill

TRAIN_RC=$?
echo "[v5] $(date -u +%H:%M:%S) training exit_code=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "[v5] TRAINING FAILED — skipping eval."
    exit "$TRAIN_RC"
fi

echo "[v5] PHASE 2a: HE-20 eval at v5 final ckpt"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v5_margin_distill \
    --T-values 1 2 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --quant nf4 --limit 20 --gen-tokens 384 --batch-size 4 \
    --output-json eval_results/v5_he20.json

echo "[v5] PHASE 2b: LCB-10 eval (separate process)"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v5_margin_distill \
    --T-values 1 2 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --quant nf4 --limit 0 --gen-tokens 384 --batch-size 4 \
    --lcb-limit 10 --lcb-difficulty medium --lcb-min-date 2024-10-01 \
    --output-json eval_results/v5_lcb10.json

echo "[v5] $(date -u +%H:%M:%S) ALL DONE"
