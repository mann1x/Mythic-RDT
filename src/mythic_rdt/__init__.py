"""Mythic-RDT: Recurrent-Depth Transformer wrapping MoE bases.

See `MASTER_PLAN.md` for project structure and goals.
Stage 1 base: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct.
Stage 2 base: ManniX-ITA/gemma-4-A4B-98e-v3-it.
"""

__version__ = "0.1.0"

from mythic_rdt.configuration import (
    MythicRDTConfig,
    MythicRDTDeepseekV2Config,
)
from mythic_rdt.recurrence import (
    LTIInjection,
    IdentityBiasedGate,
    PerLoopLayerScale,
    DepthLoRA,
    RecurrenceCell,
)

__all__ = [
    "MythicRDTConfig",
    "MythicRDTDeepseekV2Config",
    "LTIInjection",
    "IdentityBiasedGate",
    "PerLoopLayerScale",
    "DepthLoRA",
    "RecurrenceCell",
]
