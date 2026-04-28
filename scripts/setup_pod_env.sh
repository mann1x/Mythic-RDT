#!/usr/bin/env bash
# Set up the Mythic-RDT inference/eval/finetune environment on a vast.ai
# pod (or any Linux host with NVIDIA driver supporting CUDA 12.4+).
#
# Why the sidecar venv:
#   DS-Coder-V2-Lite-Instruct's modeling_deepseek.py is broken under
#   transformers 5.x — even with use_cache=False the prefill produces
#   gibberish (verified 2026-04-26). The base's tokenizer also requires
#   the custom DeepseekTokenizerFast (auto_map missing in
#   tokenizer_config.json), so AutoTokenizer silently degrades to slow
#   LlamaTokenizer that drops spaces and non-ASCII.
#
#   Pinning transformers to 4.46.x avoids both bugs. We keep this in a
#   sidecar venv so the pod's main /venv/main can stay on whatever the
#   image ships with (typically 5.x for newer PyTorch images).
#
# Usage on pod:
#   bash scripts/setup_pod_env.sh
#   source /workspace/venv-tf4/bin/activate
#   cd /workspace/mythic-rdt
#   python scripts/humaneval_smoke.py --limit 20 ...

set -euo pipefail

VENV=/workspace/venv-tf4
TORCH_VER=2.6.0
TORCH_CUDA=cu126
TRANSFORMERS_VER=4.46.3

if [ -d "$VENV" ]; then
    echo "[setup] venv already exists at $VENV — recreating to ensure clean state"
    rm -rf "$VENV"
fi

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[setup] installing torch==${TORCH_VER}+${TORCH_CUDA}"
pip install --quiet --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" "torch==${TORCH_VER}"

echo "[setup] installing transformers==${TRANSFORMERS_VER} + accompanying deps"
pip install --quiet \
    "transformers==${TRANSFORMERS_VER}" \
    accelerate \
    sentencepiece \
    huggingface_hub \
    pyarrow \
    safetensors \
    einops \
    datasets \
    wandb \
    bitsandbytes

echo "[setup] verification:"
python - <<'PY'
import torch, transformers
print(f"  torch:        {torch.__version__}  cuda_avail={torch.cuda.is_available()}")
print(f"  transformers: {transformers.__version__}")
if torch.cuda.is_available():
    print(f"  device 0:     {torch.cuda.get_device_name(0)}")
PY

echo "[setup] DONE. Activate with: source ${VENV}/bin/activate"
