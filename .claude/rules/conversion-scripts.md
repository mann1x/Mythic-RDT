---
paths:
  - scripts/convert_*.py
---

# Conversion script conventions

## Signature

All `scripts/convert_*.py` MUST take base model path + output path as args. No hardcoded paths. Reproducible.

```python
parser.add_argument("base_model_path", help="e.g. base/gemma-4-A4B-98e-v3-it")
parser.add_argument("output_path", help="e.g. checkpoints/mythic-gemma4-init")
```

## Base model source

MUST consume from `base/gemma-4-A4B-98e-v3-it/` (canonical HF download). NEVER from `../google/gemma-4-A4B-98e-hybrid/` — bytes differ (see `BASE_MODEL_ANALYSIS.md` SHA256 table).

## Output requirements

- Use `safetensors` save. Shards must not exceed 5 GB each.
- Preserve tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`).
- Copy `expert_drop_metadata.json` from base if present.
- Write `auto_map` into `config.json` so `trust_remote_code=True` works.

## Init values for RDT additions

- `log_A ~ Uniform(0.01, 0.1)`, `B = zeros + tiny noise` (LTI).
- `gate_bias = -3` (sigmoid ≈ 0.047 at start).
- `layerscale = 1e-4`.
- Depth-LoRA: rank 8 for Q/K/V/O, rank 16 for router.

These defaults make T=1 ≈ identity — phase 0 sanity gate depends on this.

## GGUF caveat

If any conversion produces GGUF: expert `intermediate_size` must be divisible by 32 (Q4_K/Q8_0), else fallback to F16 inflates the file. Note: llama.cpp does NOT support recurrent depth — GGUF is out of scope for v0.
