#!/bin/bash
# v6V train: JOINT M124f-style SFT recipe + recurrence wrap + cross-vocab
# QC-14B teacher distill. Runs scripts/06_train_sft_recurrence.py which
# combines CTD's TeacherCompletionDataset + chat-template + code-only-mask
# with Mythic-RDT's recurrence wrapper + T-curriculum + CTD distill.

set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CTD_REPO=/workspace/cross-tokenizer-distill

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/V6V_RUN.log
mkdir -p eval_results checkpoints
exec >> "$LOG" 2>&1

CACHE=teacher_cache/qc14b_xv_to_dscv2lite_funcsig_chat_top32.pt
CORPUS=/workspace/cross-tokenizer-distill/experiments/validation/data/funcsig_prompts_qwen25c14b_codeonly_T07.jsonl
OUT=checkpoints/phase1_v6v_joint_xv_qc14b

echo
echo "================================================================"
echo "[v6v] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[v6v] WANDB_API_KEY length=${#WANDB_API_KEY}"
echo "[v6v] teacher_cache: $(ls -lh $CACHE 2>&1 | tail -1)"
echo "[v6v] corpus: $(ls -lh $CORPUS 2>&1 | tail -1)"
echo "================================================================"

if [ ! -f "$CACHE" ]; then
    echo "[v6v] FATAL: teacher cache missing — precompute didn't finish?"
    exit 2
fi
if [ ! -f "$CORPUS" ]; then
    echo "[v6v] FATAL: corpus missing at $CORPUS"
    exit 2
fi

echo "[v6v] PHASE 1: JOINT SFT (M124f recipe) + recurrence wrap + xv distill"

# IMPORTANT student-path notes:
#  - Use the LOCAL pre-quantized NF4 cache, not the HF id. Two reasons:
#    1) load_dscoder_tokenizer() resolves the path and looks for
#       tokenization_deepseek_fast.py inside it; HF ids don't resolve
#       to anything that contains the file → ModuleNotFoundError.
#    2) bf16 wrapper OOMs the recurrence loop at T>=2 on 48 GB
#       (memory feedback_phase1_oom_root_causes.md). NF4 student is
#       what M124f used and is what fits the joint trainer.
#  - The NF4 cache's config.json carries the embedded
#    BitsAndBytesConfig, so we pass --quant none and let transformers
#    auto-detect (same pattern as v6V precompute teacher loading).
DSC_NF4=base/DeepSeek-Coder-V2-Lite-Instruct-nf4

python scripts/06_train_sft_recurrence.py \
    --student "$DSC_NF4" \
    --quant none \
    --corpus "$CORPUS" \
    --output-dir "$OUT" \
    --max-prompt-len 384 --max-total-len 1024 \
    --lora-rank 24 --lora-target-modules up_proj,gate_proj,down_proj \
    --lr 5e-5 --epochs 2 --batch-size 1 --grad-accum 16 \
    --warmup-steps 8 --logging-steps 5 --seed 0 \
    --code-only-mask --chat-template \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual --first-iter-identity \
    --prelude-layers 4 --coda-layers 4 \
    --max-loop-iters 4 --gate-init-bias 0.0 \
    --layerscale-init 1e-4 --layerscale-clamp-max 0.5 \
    --checkpoint-loop \
    --curriculum-style v4-anchored \
    --curriculum-warmup-steps 15 \
    --curriculum-phase2-start 25 --curriculum-phase3-start 45 \
    --teacher-logits-xv "$CACHE" \
    --teacher-distill-alpha 0.3 --teacher-distill-temperature 1.0 \
    --attn-impl flash_attention_2 --moe-vec \
    --wandb --wandb-project mythic-rdt \
    --wandb-run-name phase1_v6v_joint_xv_qc14b

TRAIN_RC=$?
echo "[v6v] $(date -u +%H:%M:%S) training exit_code=$TRAIN_RC"
[ "$TRAIN_RC" -ne 0 ] && exit "$TRAIN_RC"

echo "[v6v] PHASE 2a: HE-164 eval"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct-nf4 \
    --checkpoint "$OUT" \
    --first-iter-identity \
    --T-values 1 2 4 \
    --max-loop-iters 4 \
    --prelude-layers 4 --coda-layers 4 \
    --recurrent-block-start 4 --recurrent-block-end 22 \
    --block-mode --block-mode-residual \
    --gate-init-bias 0.0 --layerscale-clamp-max 0.5 \
    --quant none --limit 164 --gen-tokens 512 --batch-size 16 \
    --attn-impl flash_attention_2 --moe-vec \
    --output-json eval_results/v6v_he164.json

echo "[v6v] PHASE 2b: LCB-medium-full"
python scripts/humaneval_smoke.py \
    --base base/DeepSeek-Coder-V2-Lite-Instruct-nf4 \
    --checkpoint "$OUT" \
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
    --output-json eval_results/v6v_lcb_medium_full.json

echo "[v6v] $(date -u +%H:%M:%S) ALL DONE"
