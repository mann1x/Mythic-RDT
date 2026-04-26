# Setup — Mythic-RDT development environment

This project uses a **dedicated conda env named `mythic-rdt`**. Do not install into `base` or share with other projects (lightseek, vllm, llama-factory, etc.) — those are pinned for other purposes and will conflict.

## Conda env (canonical)

```bash
# 1. Create the env from the recipe
conda env create -n mythic-rdt -f environment.yml

# 2. Activate
conda activate mythic-rdt

# 3. Install the project in editable mode
cd /path/to/Mythic-RDT
pip install -e .

# 4. (Optional) install fine-tune extras
pip install -r requirements-train.txt

# 5. Sanity test
pytest tests/ -q
```

If `conda env create` fails because of channel resolution, the simpler path is to bootstrap with pip:

```bash
conda create -n mythic-rdt python=3.11 pip -y
conda activate mythic-rdt
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is the pinned dep set; `environment.yml` is the conda export of the same env (channels + a few conda-only packages).

## What's in the env

| Package | Pinned version | Why |
|---|---|---|
| python | 3.11.x | modern + well-supported by torch 2.11 / transformers 4.x |
| torch | 2.11.0+cu130 | latest stable with CUDA 13.0 |
| transformers | 4.x | for `AutoModelForCausalLM`, `trust_remote_code` loading of DeepSeek's modeling code |
| safetensors | 0.4.x | weight loading |
| accelerate | 1.x | DDP, multi-GPU placement |
| huggingface-hub | 0.x | `huggingface-cli download` |
| bitsandbytes | 0.49.x | 4-bit base loading (frozen base, no gradients on quantized weights) |
| pytest, pytest-cov | 8.x, 5.x | unit tests |
| datasets | 4.x | training data |

**Fine-tune-only extras** (in `requirements-train.txt`):
- peft 0.19.x — LoRA wrappers, frozen-base parameter management
- trl 1.2.x — SFT / RLHF helpers
- wandb 0.26.x — run tracking

## CUDA / driver expectations

The env ships torch with CUDA 13.0 wheels. solidpc has driver compatible with CUDA 12.x and 13.0. Verify on first activation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: 2.11.0+cu130 True NVIDIA GeForce RTX 3090
```

If `torch.cuda.is_available()` is False, the driver version doesn't support cu130 — fall back to:

```bash
pip install --force-reinstall torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu121
```

## Reproducibility

- `requirements.txt` is pinned (no version ranges) and was generated via `pip freeze --exclude-editable`.
- `environment.yml` is the conda export of the same env.
- Both files are committed and updated together when deps change. Procedure for updating:

  ```bash
  conda activate mythic-rdt
  pip install <new-package>
  pip freeze --exclude-editable > requirements.txt
  conda env export --no-builds | grep -v "^prefix:" > environment.yml
  git add requirements.txt environment.yml
  git commit -m "deps: ..."
  ```

## Offline / air-gapped install

```bash
# On a machine with internet:
pip download -r requirements.txt -d wheels/

# Transfer wheels/ to the air-gapped box, then:
pip install --no-index --find-links wheels/ -r requirements.txt
pip install -e .
```

## Removing the env

```bash
conda deactivate
conda env remove -n mythic-rdt
```

This will not affect any other project envs (`lightseek`, `vllm`, etc.) on solidpc.
