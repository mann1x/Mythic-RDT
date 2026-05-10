#!/bin/bash
# v6T: v6S recipe + max_loop_iters=8 + 800 steps + T=8 curriculum.
#
# Goal: Test whether the v6R/v6S T=4 result generalizes to T=8.
#       MASTER_PLAN.md headline target = HE-164 T=8 ≥ 86%.
#
# Differs from v6S in:
#   --max-loop-iters 4 → 8
#   --max-steps 400 → 800
#   --save-steps 100 → 200
#   --curriculum-style v4-anchored → default (default_curriculum supports T=8)
#   --curriculum-warmup-steps 80 → 160
#   --curriculum-phase2-start 200 → 400
#   --curriculum-phase3-start 320 → 600
#   --dual-t-hi 4 → 8
#   --ac-cpu-offload kept ON (T=8 needs the headroom for sl=2048)
#
# Pre-req: v6S must complete cleanly (validates --ac-cpu-offload before
# the long T=8 run commits ~10h of pod time).
#
# Cost: ~10h on RTX 6000 Ada.
#
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/V6T_RUN.log
mkdir -p eval_results checkpoints
exec >> "$LOG" 2>&1

echo
echo "================================================================"
echo "[v6t] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[v6t] WANDB_API_KEY length=${#WANDB_API_KEY}"
echo "[v6t] python=$(python --version)  torch=$(python -c 'import torch; print(torch.__version__)')  GPU=$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "[v6t] teacher_cache: $(ls -lh teacher_cache/dscoder_v2_lite_bf16_top32_seed0.pt 2>&1 | tail -1)"
echo "================================================================"

echo "[v6t] PHASE 1: v6S recipe + T=8, 800 steps"

python scripts/finetune_phase1.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --output-dir checkpoints/phase1_v6t_t8 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --first-iter-identity \
    --prelude-layers 4 --coda-layers 4 \
    --max-loop-iters 8 \
    --gate-init-bias 0.0 \
    --layerscale-init 1e-4 --layerscale-clamp-max 0.5 \
    --lora-rank 8 \
    --curriculum-style default \
    --curriculum-warmup-steps 160 \
    --curriculum-phase2-start 400 --curriculum-phase3-start 600 \
    --dual-t-lo 1 --dual-t-hi 8 \
    --margin-alpha 0.10 --margin-nats 0.02 \
    --distill-alpha 0.0 \
    --kl-anchor-alpha 0.0 --kl-anchor-every 0 \
    --focal-gamma 1.0 \
    --teacher-distill-alpha 0.3 \
    --teacher-logits teacher_cache/dscoder_v2_lite_bf16_top32_seed0.pt \
    --teacher-distill-temperature 1.0 \
    --teacher-refinement-mask \
    --learning-rate 5e-5 \
    --max-steps 800 --save-steps 200 \
    --data-seed 0 \
    --checkpoint-loop \
    --attn-impl flash_attention_2 --moe-vec \
    --wandb-project mythic-rdt \
    --wandb-run-name phase1_v6t_t8

TRAIN_RC=$?
echo "[v6t] $(date -u +%H:%M:%S) training exit_code=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "[v6t] TRAINING FAILED — skipping eval."
    exit "$TRAIN_RC"
fi

echo "[v6t] PHASE 2a: HE-164 eval at T=1,2,4,8"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct-nf4 \
    --checkpoint checkpoints/phase1_v6t_t8 \
    --first-iter-identity \
    --T-values 1 2 4 8 \
    --max-loop-iters 8 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --gate-init-bias 0.0 --layerscale-clamp-max 0.5 \
    --quant none --limit 164 --gen-tokens 512 --batch-size 16 \
    --attn-impl flash_attention_2 --moe-vec \
    --output-json eval_results/v6t_he164.json

echo "[v6t] PHASE 2b: LCB-medium-full at T=1,2,4,8"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct-nf4 \
    --checkpoint checkpoints/phase1_v6t_t8 \
    --first-iter-identity \
    --T-values 1 2 4 8 \
    --max-loop-iters 8 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --gate-init-bias 0.0 --layerscale-clamp-max 0.5 \
    --quant none --limit 0 --gen-tokens 512 --batch-size 16 \
    --lcb-limit 100 --lcb-difficulty medium --lcb-min-date 2024-10-01 \
    --attn-impl flash_attention_2 --moe-vec \
    --output-json eval_results/v6t_lcb_medium_full.json

echo "[v6t] $(date -u +%H:%M:%S) ALL DONE"
