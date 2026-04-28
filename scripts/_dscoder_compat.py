"""Cross-version compatibility helpers for DS-Coder-V2-Lite-Instruct.

Two known bugs the helpers work around:

1. **Tokenizer**: `AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)`
   silently falls back to slow `LlamaTokenizer` because `tokenizer_config.json`
   declares `tokenizer_class: LlamaTokenizerFast` but does NOT register an
   `auto_map` entry for the model's custom `DeepseekTokenizerFast`. The slow
   class then mis-reads the tokenizers-library `tokenizer.json` (no SentencePiece
   `tokenizer.model` in the repo) and produces lossy round-trips that drop
   spaces and silently drop non-ASCII (e.g. all CJK).
   Fix: load `DeepseekTokenizerFast` directly via the trust-remote-code path.

2. **`from_pretrained` dtype kwarg**: 5.x renamed `torch_dtype` to `dtype`.
   Cross-version code needs to pick the right one based on transformers
   major version.

See `memory/project_dscoder_5x_blocker.md` for full context.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def load_dscoder_tokenizer(base_path: str | Path):
    """Load DS-Coder-V2-Lite-Instruct's `DeepseekTokenizerFast` directly.

    Bypasses the broken AutoTokenizer fallback path. See module docstring.
    """
    base_path = str(Path(base_path).resolve())
    sys.path.insert(0, base_path)
    try:
        from tokenization_deepseek_fast import DeepseekTokenizerFast  # noqa: E402
    finally:
        sys.path.pop(0)
    return DeepseekTokenizerFast.from_pretrained(base_path)


def dtype_kwarg(dtype: torch.dtype) -> dict[str, Any]:
    """Return the right `from_pretrained` dtype kwarg for the installed transformers.

    transformers 5.x: `dtype=`
    transformers 4.x: `torch_dtype=`
    """
    import transformers as _tf
    major = int(_tf.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
