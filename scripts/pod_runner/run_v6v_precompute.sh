#!/bin/bash
# v6V precompute (SFT-aligned): build per-prompt cross-vocab QC-14B teacher
# cache aligned to the funcsig dataset's chat-template-wrapped sequences.
#
# Output: indices/values/alignment_mask of shape [N=474, max_total_len=1024, K=32]
# in DSC student-vocab indices. Drop-in for the joint trainer's
# --teacher-logits-xv flag.

set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CTD_REPO=/workspace/cross-tokenizer-distill

cd /workspace/mythic-rdt
source /workspace/venv-tf4/bin/activate

LOG=/workspace/mythic-rdt/eval_results/V6V_PRECOMPUTE.log
mkdir -p eval_results teacher_cache
exec >> "$LOG" 2>&1

OUT=teacher_cache/qc14b_xv_to_dscv2lite_funcsig_chat_top32.pt
CORPUS=/workspace/cross-tokenizer-distill/experiments/validation/data/funcsig_prompts_qwen25c14b_codeonly_T07.jsonl

echo
echo "================================================================"
echo "[v6v-pre] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[v6v-pre] corpus=$CORPUS"
echo "[v6v-pre] $(ls -lh $CORPUS 2>&1 | tail -1)"
echo "[v6v-pre] output=$OUT"
echo "================================================================"

if [ -f "$OUT" ]; then
    echo "[v6v-pre] cache exists — skipping precompute"
    ls -lh "$OUT"
    exit 0
fi

if [ ! -f "$CORPUS" ]; then
    echo "[v6v-pre] FATAL: corpus missing at $CORPUS"
    exit 2
fi

# IMPORTANT: use the pre-quantized NF4 cache that already lives on the pod
# (built 2026-05-07 for Phase 0 of v2 CTD plan). Saves a 28 GB redownload of
# bf16 weights AND avoids re-quantization. The dir's config.json carries the
# embedded BitsAndBytesConfig, so we pass --quant none and let transformers
# pick up the embedded quant settings.
QC14B_NF4=/workspace/cache_nf4/Qwen2.5-Coder-14B-Instruct-nf4
if [ ! -d "$QC14B_NF4" ]; then
    echo "[v6v-pre] FATAL: pre-quantized QC-14B NF4 cache missing at $QC14B_NF4"
    echo "[v6v-pre]        Rebuild via scripts/cache_nf4_qc14b.py or recreate from HF."
    exit 2
fi

python scripts/precompute_xv_sft_cache.py \
    --teacher "$QC14B_NF4" \
    --teacher-tokenizer "$QC14B_NF4" \
    --student-tokenizer base/DeepSeek-Coder-V2-Lite-Instruct \
    --corpus "$CORPUS" \
    --output "$OUT" \
    --top-k 32 \
    --max-prompt-len 384 --max-total-len 1024 \
    --chat-template \
    --multi-token distribute \
    --alignment student_offset \
    --quant none \
    --dtype bfloat16 \
    --shard-every 100

PRE_RC=$?
echo "[v6v-pre] $(date -u +%H:%M:%S) precompute exit_code=$PRE_RC"
[ "$PRE_RC" -ne 0 ] && exit "$PRE_RC"

echo "[v6v-pre] cache: $(ls -lh $OUT | tail -1)"
echo "[v6v-pre] DONE"
