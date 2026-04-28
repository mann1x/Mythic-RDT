"""Inject `DepthLoRA` adapters into the recurrent layer's projections.

The recurrent base layer is referenced from inside `MythicRDTDeepseekV2ForCausalLM`
as `wrapper.base.model.layers[recurrent_layer_idx]`. We do NOT rewrite its
`forward` -- instead we replace specific child Linear modules in-place with a
wrapper that adds the depth-LoRA delta on top of the frozen base output. The
LoRA delta uses the loop iteration `t` from `loop_state.get_loop_t()` (set by
the wrapper's recurrence loop).

Why in-place wrapping vs subclass: DS-Coder ships its modeling code via
`trust_remote_code`, and the layer's forward references its own attributes
(`self.self_attn.o_proj(x)`). Replacing the leaf attribute keeps the layer's
code path unchanged and is robust to future modeling.py edits in the base.

Quantization compat (QLoRA): when the base is loaded with bitsandbytes 4-bit
quantization, `o_proj` is a `bnb.nn.Linear4bit` rather than `nn.Linear`. The
wrapper still works because it holds a reference to the base module and calls
it as a black box; only the LoRA pair (`lora_A`, `lora_B`) needs to live in a
matmul-friendly dtype. We default LoRA dtype to bf16 to match the rest of the
trainable params; gradient flows through the LoRA params only (the base linear
stays frozen, quantized or not).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from ..loop_state import get_loop_t
from ..recurrence import DepthLoRA


# Default LoRA targets: q_proj or q_a_proj at the input of attention,
# o_proj at the output. Both are the standard LoRA targets and both are
# always present on DS-Coder MLA layers (q_proj when q_lora_rank is None,
# q_a_proj when set; we pick the right one at injection time).
DEFAULT_TARGETS: tuple[str, ...] = (
    "self_attn.q_proj_or_q_a",   # resolved at runtime (see _resolve_target)
    "self_attn.o_proj",
)


@dataclass
class InjectionRecord:
    """One row per LoRA-wrapped child module, for save/load + diagnostics."""
    qualified_name: str
    in_features: int
    out_features: int
    rank: int
    n_iters: int
    base_dtype: torch.dtype


class LoRAInjectedLinear(nn.Module):
    """Wrap a frozen Linear (possibly bnb 4-bit) and add a depth-LoRA delta.

    Forward: `out = base(x) + DepthLoRA(x, t=get_loop_t())`. The base module
    is held by reference; we never train it. Only the LoRA pair is a parameter.
    """

    def __init__(
        self,
        base_linear: nn.Module,
        in_features: int,
        out_features: int,
        n_iters: int,
        rank: int = 8,
        alpha: float = 16.0,
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Freeze the base; for nn.Linear this is a no-op when the wrapper's
        # outer freeze() loop has already run, but explicit is safer.
        for p in base_linear.parameters(recurse=False):
            p.requires_grad_(False)
        self.base = base_linear  # not registered as a child? -- it IS, but we
                                  # don't want its params duplicated in our
                                  # named_parameters when checkpointing. The
                                  # solution is to filter on requires_grad in
                                  # the trainer's _save (only trainable state).
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora = DepthLoRA(
            in_features=in_features,
            out_features=out_features,
            n_iters=n_iters,
            rank=rank,
            alpha=alpha,
        )
        if lora_dtype is None:
            # Match the residual stream dtype (bf16 for our setup).
            lora_dtype = torch.bfloat16
        self.lora.to(dtype=lora_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        t = get_loop_t()
        # Defensive: if t is out of range (e.g. probe scripts run T=1 with
        # max_loop_iters=8 -> t in [0, 8)), DepthLoRA will raise. Skip the
        # delta in that case so probe scripts still work.
        if t < 0 or t >= self.lora.n_iters:
            return out
        # Compute delta in the LoRA's dtype, then cast to match base output.
        delta = self.lora(x.to(self.lora.lora_A.dtype), t=t)
        return out + delta.to(out.dtype)


def _resolve_attr_path(module: nn.Module, dotted_path: str) -> tuple[nn.Module, str, nn.Module]:
    """Return (parent, leaf_attr, leaf_module) for a dotted attribute path.

    Special token "q_proj_or_q_a" picks `q_a_proj` when present (q_lora_rank
    set in the base config) else `q_proj`. Either way the result is the
    Linear at the input of the query path.
    """
    parts = dotted_path.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    leaf_name = parts[-1]
    if leaf_name == "q_proj_or_q_a":
        if hasattr(parent, "q_a_proj") and getattr(parent, "q_a_proj") is not None:
            leaf_name = "q_a_proj"
        elif hasattr(parent, "q_proj") and getattr(parent, "q_proj") is not None:
            leaf_name = "q_proj"
        else:
            raise AttributeError(
                f"Neither q_a_proj nor q_proj found on {parent.__class__.__name__}"
            )
    leaf = getattr(parent, leaf_name)
    return parent, leaf_name, leaf


def inject_depth_lora(
    wrapper,
    targets: Iterable[str] = DEFAULT_TARGETS,
    rank: int = 8,
    alpha: float = 16.0,
    lora_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> list[InjectionRecord]:
    """Inject DepthLoRA wrappers into the recurrent block's specified Linears.

    Block mode (v3+): wraps the listed `targets` on EVERY layer in
    `cfg.block_layer_indices`. Single-layer mode (v0-v2): wraps targets on
    just `cfg.recurrent_layer_idx`. Same set of `targets` used per layer.
    Each (layer, target) pair gets its own DepthLoRA module with `n_iters
    = cfg.max_loop_iters` slices.

    Returns a list of `InjectionRecord` rows describing what was wired.
    """
    cfg = wrapper.config
    n_iters = cfg.max_loop_iters
    layer_indices = list(cfg.block_layer_indices)
    records: list[InjectionRecord] = []

    targets_list = list(targets)
    for layer_idx in layer_indices:
        rec_layer = wrapper.base.model.layers[layer_idx]
        for path in targets_list:
            try:
                parent, leaf_name, leaf = _resolve_attr_path(rec_layer, path)
            except AttributeError as exc:
                # Some MLA paths only exist in certain configs; skip silently.
                print(f"[lora] skip layer {layer_idx} {path}: {exc}")
                continue
            in_f = getattr(leaf, "in_features", None)
            out_f = getattr(leaf, "out_features", None)
            if in_f is None or out_f is None:
                raise TypeError(
                    f"{type(leaf).__name__} at layer {layer_idx} {path} lacks "
                    f"in_features/out_features; cannot inject LoRA."
                )
            wrapped = LoRAInjectedLinear(
                leaf,
                in_features=in_f,
                out_features=out_f,
                n_iters=n_iters,
                rank=rank,
                alpha=alpha,
                lora_dtype=lora_dtype,
            )
            try:
                target_device = next(leaf.parameters()).device
                wrapped.lora.to(device=target_device)
            except StopIteration:
                pass
            setattr(parent, leaf_name, wrapped)
            records.append(
                InjectionRecord(
                    qualified_name=f"layers.{layer_idx}.{path}->{leaf_name}",
                    in_features=in_f,
                    out_features=out_f,
                    rank=rank,
                    n_iters=n_iters,
                    base_dtype=getattr(leaf, "weight", torch.empty(0)).dtype,
                )
            )
    return records


def list_injected(wrapper) -> list[tuple[str, LoRAInjectedLinear]]:
    """Walk the wrapper and return (qualified_name, module) for every LoRAInjectedLinear."""
    out: list[tuple[str, LoRAInjectedLinear]] = []
    for name, mod in wrapper.named_modules():
        if isinstance(mod, LoRAInjectedLinear):
            out.append((name, mod))
    return out


__all__ = [
    "DEFAULT_TARGETS",
    "InjectionRecord",
    "LoRAInjectedLinear",
    "inject_depth_lora",
    "list_injected",
]
