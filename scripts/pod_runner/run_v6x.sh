#!/bin/bash
# v6X — v6R verbatim + ONE delta: --lti-residual-scale 0.01.
#
# Council follow-up csl-2026-05-12-0311-4b68 verdict on v6W (which moved
# 4 knobs at once and lost -12.7pp at T=4 LCB):
#   * primary destroyer was FFN-on-all-experts (huge LoRA tree, dilution
#     + manifold drift in autoregressive gen even when teacher-forced
#     loss was healthy)
#   * eval bug compounded the appearance of failure: smoke didn't pass
#     lti_residual_scale to the wrapper config, silencing LTI at eval
#   * architectural bug compounded further: LTI add was OUTSIDE the gate
#     mix, so gate=0 didn't restore bit-identity to base
#
# Both bugs fixed in commit before this run (smoke autodetects from ckpt
# config; recurrence.py block_mode_residual now puts LTI inside mix).
#
# v6X recipe = v6R verbatim, ONE delta:
#   --lti-residual-scale 0.01    (only architectural addition vs v6R;
#                                 with the inside-gate fix, this can't
#                                 break the gate=0 identity invariant)
#
# UNCHANGED from v6R (the proven good knobs):
#   - layerscale-init 0.05       (recurrence is open, not 1e-4 dead)
#   - margin-alpha 0.10          (T_hi > T_lo gradient pressure)
#   - teacher-refinement-mask    (focused distill on disagreement)
#   - curriculum 80/200/320      (slow ramp; gate stabilizes before T=4)
#   - sv teacher (dscoder bf16)
#   - first_iter_identity, block_mode_residual
#   - prelude=4, coda=4, block 4..22
#   - LoRA attn-only (q, o)      (NOT FFN — proven destroyer in v6W)
#   - LoRA rank 8                (NOT 24 — overfit risk)
#   - 400 steps                  (NOT 800 — drift away from base)
#
# Decision rule on completion (LCB-medium-full n=55, base = 25.5%) — REVISED
# per council to require monotonicity AND HE floor:
#   * MUST: T=4 ≥ T=2 ≥ T=1                  (monotonicity — refinement, not noise)
#   * MUST: HE-164 ≥ 70.7%                   (no code-prior regression)
#   * THEN T=4 LCB ≥ 32.7% → real signal     (council win condition)
#   * T=4 LCB 25-31% with mono+HE OK → marginal, iterate
#   * T=4 LCB < 22% OR mono violated OR HE drop → REJECT, council follow-up
#
# Cost: ~7-13h on RTX 6000 Ada (matches v6R wall-time). Same teacher
# cache as v6R (teacher_cache/dscoder_v2_lite_bf16_top32_seed0.pt).
#
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/V6X_RUN.log
mkdir -p eval_results checkpoints
exec >> "$LOG" 2>&1

echo
echo "================================================================"
echo "[v6x] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[v6x] WANDB_API_KEY length=${#WANDB_API_KEY}"
echo "[v6x] python=$(python --version)  CUDA=$(python -c 'import torch; print(torch.version.cuda)')  GPU=$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "[v6x] teacher_cache: $(ls -lh teacher_cache/dscoder_v2_lite_bf16_top32_seed0.pt 2>&1 | tail -1)"
echo "================================================================"

if [ ! -f teacher_cache/dscoder_v2_lite_bf16_top32_seed0.pt ]; then
    echo "[v6x] FATAL: same-vocab teacher cache missing"
    exit 2
fi

echo "[v6x] PHASE 1: v6R recipe + lti-residual-scale 0.01 (inside-gate), 400 steps"

python scripts/finetune_phase1.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct \
    --output-dir checkpoints/phase1_v6x_lti_inside_gate \
    --quant nf4 \
    --max-loop-iters 4 \
    --first-iter-identity \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --gate-init-bias 0.0 \
    --layerscale-init 0.05 --layerscale-clamp-max 0.5 \
    --lti-residual-scale 0.01 \
    --lora-rank 8 \
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
    --max-steps 400 --save-steps 100 \
    --data-seed 0 \
    --checkpoint-loop \
    --attn-impl flash_attention_2 --moe-vec \
    --wandb-project mythic-rdt \
    --wandb-run-name phase1_v6x_lti_inside_gate

TRAIN_RC=$?
echo "[v6x] $(date -u +%H:%M:%S) training exit_code=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "[v6x] TRAINING FAILED — skipping eval."
    exit "$TRAIN_RC"
fi

echo "[v6x] PHASE 2a: HE-164 (autodetect lti_residual_scale from ckpt config)"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct-nf4 \
    --checkpoint checkpoints/phase1_v6x_lti_inside_gate \
    --first-iter-identity \
    --T-values 1 2 4 \
    --max-loop-iters 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --gate-init-bias 0.0 --layerscale-clamp-max 0.5 \
    --quant none --limit 164 --gen-tokens 512 --batch-size 16 \
    --attn-impl flash_attention_2 --moe-vec \
    --output-json eval_results/v6x_he164.json

echo "[v6x] PHASE 2b: LCB-medium-full"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct-nf4 \
    --checkpoint checkpoints/phase1_v6x_lti_inside_gate \
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
    --output-json eval_results/v6x_lcb_medium_full.json

echo "[v6x] $(date -u +%H:%M:%S) ALL DONE"
