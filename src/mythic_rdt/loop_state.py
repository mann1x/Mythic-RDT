"""Tiny contextvar holding the current recurrence loop iteration t.

Set by `MythicRDTDeepseekV2ForCausalLM.forward()` inside the loop, read by
DepthLoRA-wrapped Linear layers nested inside the recurrent base layer.

Why a contextvar (vs threading kwargs through the layer): the recurrent base
layer is the standard `DeepseekV2DecoderLayer`. Its `forward(hidden_states,
attention_mask, position_ids, ...)` signature is fixed; we cannot pass `t`
through it without monkey-patching the upstream code (fragile + bad for
trust_remote_code reload). A contextvar is set once before the layer call and
read by any LoRA-wrapped Linear deep inside the layer's forward.

This module has zero dependencies so it can be imported from both modeling.py
and training/lora_inject.py without circular-import risk.
"""
from __future__ import annotations

import contextvars

# Default 0 = "first iteration"; layers reading this without a set() call (e.g.
# during the Phase 0 bit-exact probe) get t=0 which is a valid LoRA iteration.
_current_t: contextvars.ContextVar[int] = contextvars.ContextVar(
    "mythic_rdt_loop_t", default=0
)


def set_loop_t(t: int) -> None:
    """Record the current loop iteration. Called by the wrapper's recurrence loop."""
    _current_t.set(int(t))


def get_loop_t() -> int:
    """Return the current loop iteration. Called by DepthLoRA-wrapped Linears."""
    return _current_t.get()


__all__ = ["set_loop_t", "get_loop_t"]
