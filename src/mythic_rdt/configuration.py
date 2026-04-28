"""Mythic-RDT configuration classes.

Stage-specific subclasses of `transformers.PretrainedConfig` that hold
the recurrence hyperparameters (LTI, depth-LoRA, gate, LayerScale, loop
counts) on top of a base model's existing config.

The base model's own config is loaded separately at runtime via
`AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)`
and stored on `self.base_config_dict` so it survives `save_pretrained`
without us having to re-import the upstream config class.

Naming follows MASTER_PLAN.md and the project's HF custom-code shape:
- Top-level: `MythicRDTConfig` (currently empty container; reserved for
  cross-base shared fields if/when they emerge).
- Stage 1: `MythicRDTDeepseekV2Config` (base = DeepSeek-Coder-V2-Lite).
- Stage 2 (later): `MythicRDTGemma4Config` (base = Gemma 4 26B-A4B).
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import PretrainedConfig


class MythicRDTConfig(PretrainedConfig):
    """Empty cross-base parent. Stage subclasses set `model_type`."""

    model_type = "mythic_rdt"


class MythicRDTDeepseekV2Config(MythicRDTConfig):
    """Recurrent-Depth wrapper config for DeepSeek-Coder-V2-Lite-Instruct.

    Field naming is non-negotiable -- HF custom-code paths and the
    surgery / phase0 scripts read these by name. Defaults match
    MASTER_PLAN.md Stage 1 spec.

    Args:
        prelude_layers: number of base layers consumed verbatim before
            the recurrence loop (DS-Coder-V2-Lite layer 0 is the only
            dense FFN, so default = 1).
        coda_layers: number of base layers consumed verbatim after the
            recurrence loop. Default = 1.
        recurrent_layer_idx: index into the base model's `model.layers`
            list selecting the layer reused T times. Default = 13
            (middle of 27-layer DS-Coder-V2-Lite). Must be in
            (prelude_layers - 1, num_hidden_layers - coda_layers).
        train_loop_iters: T used during fine-tune (curriculum changes
            this between phases).
        max_loop_iters: T cap at inference; depth-extrapolatable.
        lti_log_a_init_low / high: uniform init bounds for `log_A` so
            A = -exp(log_A) starts small-magnitude (Parcae form).
        lti_b_init_std: std of LTI B-projection init (small ~ near zero).
        depth_lora_rank: rank for per-iteration LoRA on Q/K/V/O. Router
            LoRA uses 2x this (per master plan).
        depth_lora_alpha: LoRA scaling (alpha / rank).
        gate_init_bias: identity-biased gate init bias (sigmoid(-3) ~
            0.047). Combined with layerscale_init = 1e-4 gives an
            effective open factor ~ 5e-6 -- near-identity loop at init.
        layerscale_init: per-loop LayerScale init value.
        layerscale_per_channel: if True, [T, hidden] diagonal scale;
            else single scalar per iter.
        halting_strategy: 'fixed' (only option in v0). 'act' deferred.
        base_model_path: filesystem path or HF id of the base model.
            The wrapper loads it via AutoModelForCausalLM at __init__.
        base_config_dict: snapshot of the base model's config.to_dict()
            so we don't depend on the dynamic class at unpickle time.
        recurrence_norm_dtype: dtype string ('float32' | 'bfloat16') for
            RMSNorm inside the recurrence path. Default 'float32' for
            stability (parent project bf16 NaN bug).
    """

    model_type = "mythic_rdt_deepseek_v2"

    def __init__(
        self,
        prelude_layers: int = 1,
        coda_layers: int = 1,
        # Single-layer mode (v0/v1/v2): one base layer iterated T times.
        # Default 10 from experiments/01_phase0_probe (2026-04-26).
        recurrent_layer_idx: int = 10,
        # Block mode (v3+): iterate a CONSECUTIVE range of base layers
        # [recurrent_block_start, recurrent_block_end] (inclusive) T times.
        # When both are set, block mode takes precedence over
        # recurrent_layer_idx. v2 catastrophic-regression analysis
        # (memory: project_phase1_v2_catastrophic_regression.md) showed the
        # 3-layer skeleton (prelude=1/coda=1/single layer) cannot be
        # trained from 16M tokens; v3 uses a 19-layer block to keep all 27
        # base layers in play.
        recurrent_block_start: Optional[int] = None,
        recurrent_block_end: Optional[int] = None,
        # Block mode also changes the recurrence formula:
        #   block_mode=False (v0-v2): h_next = h + ls*gate*(inj + block_out)
        #     -- at gate≈0 the iteration discards block_out entirely. Fine
        #     for one layer (skipping it ≈ identity), broken for a 19-layer
        #     block (skipping 19 layers gives gibberish).
        #   block_mode=True (v3+): h_next = block_out + ls*gate*inj
        #     -- at gate=0 the iteration keeps block_out. Bit-exact base
        #     behavior at T=1 if no LoRA is injected.
        block_mode: bool = False,
        train_loop_iters: int = 1,
        max_loop_iters: int = 16,
        lti_log_a_init_low: float = 0.01,
        lti_log_a_init_high: float = 0.1,
        lti_b_init_std: float = 1e-4,
        depth_lora_rank: int = 8,
        depth_lora_alpha: float = 16.0,
        gate_init_bias: float = -3.0,
        layerscale_init: float = 1e-4,
        layerscale_clamp_max: Optional[float] = None,
        layerscale_per_channel: bool = False,
        halting_strategy: str = "fixed",
        base_model_path: Optional[str] = None,
        base_config_dict: Optional[dict[str, Any]] = None,
        recurrence_norm_dtype: str = "float32",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prelude_layers = int(prelude_layers)
        self.coda_layers = int(coda_layers)
        self.recurrent_layer_idx = int(recurrent_layer_idx)
        self.recurrent_block_start = (
            int(recurrent_block_start) if recurrent_block_start is not None else None
        )
        self.recurrent_block_end = (
            int(recurrent_block_end) if recurrent_block_end is not None else None
        )
        self.block_mode = bool(block_mode)
        self.train_loop_iters = int(train_loop_iters)
        self.max_loop_iters = int(max_loop_iters)
        self.lti_log_a_init_low = float(lti_log_a_init_low)
        self.lti_log_a_init_high = float(lti_log_a_init_high)
        self.lti_b_init_std = float(lti_b_init_std)
        self.depth_lora_rank = int(depth_lora_rank)
        self.depth_lora_alpha = float(depth_lora_alpha)
        self.gate_init_bias = float(gate_init_bias)
        self.layerscale_init = float(layerscale_init)
        self.layerscale_clamp_max = (
            float(layerscale_clamp_max) if layerscale_clamp_max is not None else None
        )
        self.layerscale_per_channel = bool(layerscale_per_channel)
        self.halting_strategy = str(halting_strategy)
        self.base_model_path = base_model_path
        self.base_config_dict = base_config_dict
        self.recurrence_norm_dtype = str(recurrence_norm_dtype)

        if halting_strategy not in {"fixed"}:
            raise ValueError(
                f"halting_strategy={halting_strategy!r} not supported in v0; "
                "only 'fixed' (ACT deferred to phase 7)"
            )
        if recurrence_norm_dtype not in {"float32", "bfloat16"}:
            raise ValueError(
                f"recurrence_norm_dtype={recurrence_norm_dtype!r} must be "
                "'float32' or 'bfloat16'"
            )
        if max_loop_iters < train_loop_iters:
            raise ValueError(
                f"max_loop_iters ({max_loop_iters}) must be >= "
                f"train_loop_iters ({train_loop_iters})"
            )
        # Block-mode validity (only when both endpoints are provided)
        if self.recurrent_block_start is not None or self.recurrent_block_end is not None:
            if self.recurrent_block_start is None or self.recurrent_block_end is None:
                raise ValueError(
                    "recurrent_block_start and recurrent_block_end must both be "
                    "set, or both None (single-layer mode via recurrent_layer_idx)"
                )
            if self.recurrent_block_end < self.recurrent_block_start:
                raise ValueError(
                    f"recurrent_block_end ({self.recurrent_block_end}) must be "
                    f">= recurrent_block_start ({self.recurrent_block_start})"
                )
            if self.recurrent_block_start < self.prelude_layers:
                raise ValueError(
                    f"recurrent_block_start ({self.recurrent_block_start}) must "
                    f"be >= prelude_layers ({self.prelude_layers})"
                )
        elif recurrent_layer_idx < prelude_layers:
            raise ValueError(
                f"recurrent_layer_idx ({recurrent_layer_idx}) must be >= "
                f"prelude_layers ({prelude_layers})"
            )

    @property
    def block_layer_indices(self) -> list[int]:
        """Resolved list of base-layer indices that form the recurrent block.

        Block mode: range(start, end+1).
        Single-layer mode: [recurrent_layer_idx].
        """
        if self.recurrent_block_start is not None and self.recurrent_block_end is not None:
            return list(range(self.recurrent_block_start, self.recurrent_block_end + 1))
        return [self.recurrent_layer_idx]

    def validate_against_base(self, num_hidden_layers: int) -> None:
        """Cross-check geometry against the loaded base model.

        Called from the wrapper __init__ once the base config is known.
        """
        if self.recurrent_block_start is not None and self.recurrent_block_end is not None:
            if self.recurrent_block_end >= num_hidden_layers - self.coda_layers:
                raise ValueError(
                    f"recurrent_block_end={self.recurrent_block_end} collides "
                    f"with coda window [{num_hidden_layers - self.coda_layers}, "
                    f"{num_hidden_layers}); block must precede coda"
                )
        elif self.recurrent_layer_idx >= num_hidden_layers - self.coda_layers:
            raise ValueError(
                f"recurrent_layer_idx={self.recurrent_layer_idx} collides with "
                f"coda window [{num_hidden_layers - self.coda_layers}, "
                f"{num_hidden_layers}); recurrent block must precede coda"
            )
        if self.prelude_layers + self.coda_layers >= num_hidden_layers:
            raise ValueError(
                f"prelude ({self.prelude_layers}) + coda ({self.coda_layers}) "
                f">= num_hidden_layers ({num_hidden_layers})"
            )


__all__ = ["MythicRDTConfig", "MythicRDTDeepseekV2Config"]
