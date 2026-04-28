#!/bin/bash
# v2: lora-targets dropped (use script default self_attn.q_proj_or_q_a + self_attn.o_proj),
# WANDB_API_KEY assumed in env (caller exports via SSH).
set -uo pipefail
export PYTHONUNBUFFERED=1

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/OVERNIGHT_RUN_V2.log
exec >> "$LOG" 2>&1

echo
echo "================================================================"
echo "[overnight v2] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[overnight v2] WANDB_API_KEY length=${#WANDB_API_KEY:-0}"
echo "================================================================"

rm -rf src/mythic_rdt/__pycache__

echo "[overnight v2] PHASE 1: v4-extended training (CORRECTED lora-targets)"
echo "[overnight v2] $(date -u +%H:%M:%S) launching..."

python scripts/finetune_phase1.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --output-dir checkpoints/phase1_v4_extended \
    --init-from-checkpoint checkpoints/phase1_v4_anchored \
    --quant nf4 \
    --max-loop-iters 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --lora-rank 8 \
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
echo "[overnight v2] $(date -u +%H:%M:%S) training exit_code=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "[overnight v2] TRAINING FAILED — skipping eval."
    exit "$TRAIN_RC"
fi

rm -rf src/mythic_rdt/__pycache__
echo "[overnight v2] PHASE 2a: HE-20 eval"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --checkpoint checkpoints/phase1_v4_extended \
    --T-values 1 2 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --gate-init-bias 0.0 --layerscale-clamp-max 1e-2 \
    --quant nf4 --limit 20 --gen-tokens 384 --batch-size 4 \
    --output-json eval_results/v4_extended_he20.json

rm -rf src/mythic_rdt/__pycache__
echo "[overnight v2] PHASE 2b: LCB-10 eval (separate process)"
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

echo "[overnight v2] $(date -u +%H:%M:%S) ALL DONE"
