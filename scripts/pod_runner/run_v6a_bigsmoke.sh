#!/bin/bash
# v6A big-smoke: HE-164 + LCB-medium-30 at T=1/2/4 split-process.
# Tighter sampling noise than the auto-pipeline's HE-20 / LCB-10.
# Pre-condition: checkpoints/phase1_v6a_dual_t/checkpoint-200/ exists on pod.
# ETA ~50 min on A6000 with NF4 base.
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/V6A_BIGSMOKE.log
exec >> "$LOG" 2>&1

echo
echo "================================================================"
echo "[v6a-bigsmoke] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

echo "[v6a-bigsmoke] PHASE A: HE-full (164 problems) at T=1/2/4"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v6a_dual_t \
    --first-iter-identity \
    --T-values 1 2 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --quant nf4 --limit 0 --gen-tokens 384 --batch-size 4 \
    --output-json eval_results/v6a_he164.json

echo "[v6a-bigsmoke] PHASE B: LCB-medium-30 at T=1/2/4 (separate process)"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v6a_dual_t \
    --first-iter-identity \
    --T-values 1 2 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --quant nf4 --limit 0 --gen-tokens 384 --batch-size 4 \
    --lcb-limit 30 --lcb-difficulty medium --lcb-min-date 2024-10-01 \
    --output-json eval_results/v6a_lcb30.json

echo "[v6a-bigsmoke] $(date -u +%H:%M:%S) ALL DONE"
