#!/bin/bash
# v6W: v6R recipe (proven productive recurrence dynamics) + 4 deltas from
# the 2026-05-10 council audit (csl-2026-05-10-1918-30a0):
#
# DELTAS vs v6R:
#   1. --lora-target-modules up_proj,gate_proj,down_proj  (FFN-on-all-experts;
#                                                          M124f winner target
#                                                          set, ~3.7× capacity
#                                                          vs attn-only rank 8
#                                                          even after MoE
#                                                          top-6/64 sparsity)
#   2. --lora-rank 24                                     (matches v6V capacity;
#                                                          fits within 48 GB
#                                                          with checkpoint-loop)
#   3. --lti-residual-scale 0.01                          (council #5 fix:
#                                                          LTI was DEAD in
#                                                          block_mode_residual
#                                                          across v6H/Q/R/V.
#                                                          0.01 is small enough
#                                                          to bound ||h|| growth
#                                                          while giving A_diag
#                                                          /B_proj a real
#                                                          gradient signal)
#   4. --max-steps 800 --save-steps 200                   (v6R was 400; with
#                                                          larger LoRA tree
#                                                          + LTI active we
#                                                          want more steps to
#                                                          let recurrence
#                                                          converge before
#                                                          curriculum forces
#                                                          T=4-dominant regime)
#
# UNCHANGED from v6R (the GOOD knobs):
#   - layerscale-init 0.05    (500× v6V's 1e-4 — actual recurrence signal)
#   - margin-alpha 0.10       (explicit T_hi > T_lo gradient pressure)
#   - teacher-refinement-mask (focused distill, not blanket anchor)
#   - curriculum 80/200/320   (slow ramp; gate stabilizes before T=4)
#   - same-vocab teacher cache (dscoder bf16, no QC-14B alignment noise)
#   - first_iter_identity, block_mode, block_mode_residual
#   - prelude=4, coda=4, block 4..22
#
# Decision rule on completion (LCB-medium-full n=55, base = 25.5%):
#   * T=4 ≥ 32.7% (≥ +4 problems vs base, ~1.5 SE) → REAL signal. Iterate (v6X).
#   * T=4 28-32% → marginal; consider longer training (v6X-extended).
#   * T=4 25-27% (= base) → fixes didn't compound; LTI dead bug wasn't the
#     bottleneck after all. Step back, try alternative (curriculum tweaks,
#     different LoRA placement).
#   * T=4 < 22% → FFN-LoRA + LTI together broke the prior. Roll back to v6R.
#
# Cost: ~10-14h on RTX 6000 Ada (longer training + larger LoRA tree).
#
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/V6W_RUN.log
mkdir -p eval_results checkpoints
exec >> "$LOG" 2>&1

echo
echo "================================================================"
echo "[v6w] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[v6w] WANDB_API_KEY length=${#WANDB_API_KEY}"
echo "[v6w] python=$(python --version)  CUDA=$(python -c 'import torch; print(torch.version.cuda)')  GPU=$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "[v6w] teacher_cache: $(ls -lh teacher_cache/dscoder_v2_lite_bf16_top32_seed0.pt 2>&1 | tail -1)"
echo "================================================================"

if [ ! -f teacher_cache/dscoder_v2_lite_bf16_top32_seed0.pt ]; then
    echo "[v6w] FATAL: same-vocab teacher cache missing"
    exit 2
fi

echo "[v6w] PHASE 1: v6R recipe + FFN+rank24 + LTI residual + 800 steps"

python scripts/finetune_phase1.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --output-dir checkpoints/phase1_v6w_ffn_rank24_lti \
    --quant nf4 \
    --max-loop-iters 4 \
    --first-iter-identity \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --gate-init-bias 0.0 \
    --layerscale-init 0.05 --layerscale-clamp-max 0.5 \
    --lti-residual-scale 0.01 \
    --lora-rank 24 \
    --lora-targets up_proj gate_proj down_proj \
    --curriculum-style v4-anchored \
    --curriculum-warmup-steps 80 \
    --curriculum-phase2-start 200 --curriculum-phase3-start 320 \
    --dual-t-lo 1 --dual-t-hi 4 \
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
    --wandb-run-name phase1_v6w_ffn_rank24_lti

TRAIN_RC=$?
echo "[v6w] $(date -u +%H:%M:%S) training exit_code=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "[v6w] TRAINING FAILED — skipping eval."
    exit "$TRAIN_RC"
fi

echo "[v6w] PHASE 2a: HE-164 (autodetect handles FFN+rank24)"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct-nf4 \
    --checkpoint checkpoints/phase1_v6w_ffn_rank24_lti \
    --first-iter-identity \
    --T-values 1 2 4 \
    --max-loop-iters 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --gate-init-bias 0.0 --layerscale-clamp-max 0.5 \
    --quant none --limit 164 --gen-tokens 512 --batch-size 16 \
    --attn-impl flash_attention_2 --moe-vec \
    --output-json eval_results/v6w_he164.json

echo "[v6w] PHASE 2b: LCB-medium-full"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct-nf4 \
    --checkpoint checkpoints/phase1_v6w_ffn_rank24_lti \
    --first-iter-identity \
    --T-values 1 2 4 \
    --max-loop-iters 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --gate-init-bias 0.0 --layerscale-clamp-max 0.5 \
    --quant none --limit 0 --gen-tokens 512 --batch-size 16 \
    --lcb-limit 100 --lcb-difficulty medium --lcb-min-date 2024-10-01 \
    --attn-impl flash_attention_2 --moe-vec \
    --output-json eval_results/v6w_lcb_medium_full.json

echo "[v6w] $(date -u +%H:%M:%S) ALL DONE"
