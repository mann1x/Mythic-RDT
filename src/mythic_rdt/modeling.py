"""Mythic-RDT modeling: thin wrapper around a frozen base MoE.

Stage 1 wrapper for `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`. The
base model is loaded via `trust_remote_code=True` (its modeling code
ships in the HF repo) and is frozen verbatim. The wrapper reorders the
forward pass into:

    embed -> [prelude layers] -> e
                              -> for t in 0..T-1:
                                     block_out = layer[recurrent_idx](h)
                                     h = h + ls * gate * (LTI(h, e) + block_out)
                              -> [coda layers] -> norm -> lm_head

The recurrence machinery (`LTIInjection`, `IdentityBiasedGate`,
`PerLoopLayerScale`, `RecurrenceCell`) lives in `recurrence.py` and is
unit-tested in isolation.

Phase 0 (this file): the wrapper supports a `force_gate_zero=True` mode
that bypasses the recurrence machinery entirely and uses
`h <- block_out`. With T=1, this gives a forward pass that is bit-exact
with running the base's chosen middle layer once on the prelude output
(MASTER_PLAN.md §5 hard gate). It is a plumbing-correctness check, not
the runtime mode -- runtime uses `force_gate_zero=False`.

Phase 1+ (later): depth-LoRA on Q/K/V/O + router, fp32 RMSNorm inside
the loop body, optional saving via `save_pretrained` with `auto_map`.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration import MythicRDTDeepseekV2Config
from .loop_state import set_loop_t
from .recurrence import RecurrenceCell


class MythicRDTDeepseekV2ForCausalLM(nn.Module):
    """Recurrent-Depth wrapper around DeepSeek-Coder-V2-Lite-Instruct.

    Note: this is a plain `nn.Module`, not a `PreTrainedModel`. v0
    intentionally does not subclass HF's mixin; saving via the custom-
    code `auto_map` route is deferred to phase 4 (after fine-tune). The
    wrapper holds a reference to the loaded base; gradients flow only
    through the new RDT params (LTI, gate, LayerScale, depth-LoRA when
    added) -- the base is frozen at construction.

    Args:
        config: a `MythicRDTDeepseekV2Config`.
        base: optionally an already-loaded base model. If None, the
            wrapper loads it from `config.base_model_path` via
            `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.
            For Phase 0 / multi-probe scripts, pass an already-loaded
            base to avoid 30 GB reloads.
        torch_dtype: dtype to use when loading the base if `base=None`.
    """

    def __init__(
        self,
        config: MythicRDTDeepseekV2Config,
        base: Optional[nn.Module] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.config = config

        if base is None:
            if config.base_model_path is None:
                raise ValueError(
                    "MythicRDTDeepseekV2Config.base_model_path is None and "
                    "no `base` was passed; cannot load base model."
                )
            base = AutoModelForCausalLM.from_pretrained(
                config.base_model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )

        # Snapshot the base config so Mythic config is self-describing
        # for later save/reload -- avoids depending on the dynamic class.
        if config.base_config_dict is None:
            try:
                config.base_config_dict = base.config.to_dict()
            except Exception:
                # to_dict on remote-code configs sometimes pulls in
                # non-serializable extras; fall back to a minimal dict.
                config.base_config_dict = {
                    "hidden_size": base.config.hidden_size,
                    "num_hidden_layers": base.config.num_hidden_layers,
                    "model_type": getattr(base.config, "model_type", "unknown"),
                }

        config.validate_against_base(num_hidden_layers=base.config.num_hidden_layers)

        self.base = base

        # New trainable RDT params -- pure-Python, base-independent.
        self.recurrence = RecurrenceCell(
            hidden_size=base.config.hidden_size,
            n_iters=config.max_loop_iters,
            layerscale_init=config.layerscale_init,
            layerscale_clamp_max=config.layerscale_clamp_max,
            gate_init_bias=config.gate_init_bias,
            layerscale_per_channel=config.layerscale_per_channel,
            block_mode=config.block_mode,
            lti_log_a_init_low=config.lti_log_a_init_low,
            lti_log_a_init_high=config.lti_log_a_init_high,
            lti_b_init_std=config.lti_b_init_std,
        )
        # Match the base dtype so LTI Linear and other params don't fight
        # bf16 activations. Inside the loop we still cast to fp32 around
        # RMSNorm calls (phase 1+) per MASTER_PLAN.md stability rule.
        try:
            base_dtype = next(self.base.parameters()).dtype
        except StopIteration:
            base_dtype = torch.float32
        self.recurrence.to(dtype=base_dtype)

        # Freeze base; only RDT params are trainable in v0.
        for p in self.base.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_T(self, T: Optional[int]) -> int:
        if T is None:
            return (
                self.config.train_loop_iters
                if self.training
                else self.config.max_loop_iters
            )
        if T < 1 or T > self.config.max_loop_iters:
            raise ValueError(
                f"T={T} out of valid range [1, {self.config.max_loop_iters}]"
            )
        return T

    def _run_layer(
        self,
        layer: nn.Module,
        h: torch.Tensor,
        attn_4d: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """Call a DeepseekV2DecoderLayer; optionally return updated KV cache.

        Returns (hidden_states, present_key_value). When use_cache=False the
        second tuple element is always None and callers can ignore it.
        """
        out = layer(
            h,
            attention_mask=attn_4d,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=use_cache,
        )
        # When use_cache=True: layer returns (hidden, present_kv) (or 3-tuple
        # with attn weights at index 1 if output_attentions=True, which we
        # don't request). When False: returns (hidden,).
        if use_cache:
            present = out[1] if len(out) > 1 else None
            return out[0], present
        return out[0], None

    def _loop_step(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        t: int,
        rec_layers: list,
        attn_4d: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        force_gate_zero: bool,
    ) -> torch.Tensor:
        """One iteration of the recurrent loop. Pulled out so the same body
        can be called directly OR through `torch.utils.checkpoint`. Keeping
        `set_loop_t` INSIDE this function is intentional: when checkpoint
        re-executes the body during backward, the contextvar is set again
        before DepthLoRA reads it -- so the right T-slice adapter is picked
        on both forward and re-forward.

        `rec_layers` is a list of consecutive base layers forming the
        recurrent block (length 1 in single-layer mode, N in block mode).
        Each iteration passes the residual through ALL layers in order;
        `set_loop_t(t)` is called once per iteration (not per layer) so
        every block layer's DepthLoRA picks the same T-slice within an
        iteration.
        """
        set_loop_t(t)
        block_out = h
        for layer in rec_layers:
            block_out, _ = self._run_layer(layer, block_out, attn_4d, position_ids)
        if force_gate_zero and not self.recurrence.block_mode:
            # Legacy single-layer mode bypass: return the block output
            # directly, mimicking "run the layer once".
            return block_out
        cell_out = self.recurrence(h, e, block_out, t=t, force_gate_zero=force_gate_zero)
        return cell_out.h_next

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        T: Optional[int] = None,
        force_gate_zero: bool = False,
        force_bypass: bool = False,
        return_hidden_trace: bool = False,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[dict] = None,
        use_cache: bool = False,
        **_unused,
    ) -> torch.Tensor | dict[str, Any] | CausalLMOutputWithPast:
        """Forward pass through prelude -> recurrence loop -> coda -> lm_head.

        Args:
            input_ids: [batch, seq] token ids.
            attention_mask: optional [batch, seq] padding mask (1 = keep).
            T: number of recurrence iterations (defaults to
                `train_loop_iters` in train mode, `max_loop_iters` in
                eval mode). Must be in [1, max_loop_iters].
            force_gate_zero: Phase 0 plumbing-correctness mode. When
                True, the recurrence cell is bypassed and the loop body
                becomes simply `h <- block_out`. With T=1 this yields a
                forward pass that is bit-exact with running the base's
                chosen middle layer once on the prelude output (after
                coda + norm + lm_head).
            return_hidden_trace: if True, also return intermediate
                hidden states at each phase boundary (for debugging).

        Returns:
            Logits tensor [batch, seq, vocab], OR a dict with logits +
            hidden trace if `return_hidden_trace=True`.
        """
        T = self._resolve_T(T)
        cfg = self.config

        # KV cache: per-iteration `DynamicCache` shared across all layers
        # within one recurrence iteration. DS-V2's MLA expects a Cache object
        # with `.update(K, V, layer_idx, ...)` and `.get_seq_length(layer_idx)`
        # — each base layer writes to slot[self.layer_idx] of the same cache.
        # We need T separate caches because the block layers run T times,
        # each iteration with its own K/V history (same self.layer_idx but
        # different inputs after prior iters' recurrence-cell updates).
        # Cache structure: dict[int, DynamicCache] keyed by iteration t.
        # - past_key_values[0] holds: prelude (t=0) + block iter 0 + coda (t=0)
        # - past_key_values[t] for t>=1 holds: block iter t
        base_model = self.base.model  # the inner DeepseekV2Model
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if use_cache:
            from transformers.cache_utils import DynamicCache
            if past_key_values is None:
                past_key_values = {}
            # Ensure a cache exists for each recurrence iteration we will run.
            for t in range(T):
                if t not in past_key_values:
                    past_key_values[t] = DynamicCache()
            # past_kv_len = how many tokens are already cached on slot[0]
            # (first prelude layer). All slots on the same iteration cache
            # have the same length by construction.
            past_kv_len = int(past_key_values[0].get_seq_length(0))
        else:
            past_kv_len = 0

        # Position ids: must match HF's `prepare_inputs_for_generation` so that
        # left-padded batches get content-relative RoPE positions, not
        # absolute-buffer positions. Without this, padding tokens push every
        # content token's RoPE phase forward by `pad_len` -- harmless when the
        # 4d mask suppresses padded positions in attention BUT lethal when KV
        # caching, because cached K/V are encoded at the wrong phase and never
        # rewritten. Symptom: long generations diverge from base (HE-easy
        # masks the bug because prompts are short and similar-length, LCB
        # surfaces it because prompts are long with high length variance).
        if attention_mask is not None and attention_mask.dim() == 2:
            # Per-sequence content-relative positions, padded slots get 1 (
            # unused — masked out by attn_4d). Matches base.generate's
            # prepare_inputs_for_generation exactly.
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # For incremental decode (seq_len < attention_mask cols), keep
            # only the trailing `seq_len` positions corresponding to the
            # tokens actually being processed this call.
            if seq_len < position_ids.size(-1):
                position_ids = position_ids[:, -seq_len:]
        else:
            position_ids = torch.arange(
                past_kv_len, past_kv_len + seq_len,
                dtype=torch.long, device=device,
            ).unsqueeze(0).expand(bsz, -1)

        inputs_embeds = base_model.embed_tokens(input_ids)

        # Build 4d causal attention mask matching base's own pipeline.
        if base_model._use_flash_attention_2:
            attn_4d = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            attn_4d = _prepare_4d_causal_attention_mask(
                attention_mask,
                (bsz, seq_len),
                inputs_embeds,
                past_key_values_length=past_kv_len,
            )

        h = inputs_embeds
        trace: dict[str, torch.Tensor] = {}
        if return_hidden_trace:
            trace["after_embed"] = h.detach().clone()

        # ----- Prelude -----
        # All prelude layers write to slot[layer_idx] of the iteration-0 cache.
        prelude_cache = past_key_values[0] if use_cache else None
        for i in range(cfg.prelude_layers):
            h, _ = self._run_layer(
                base_model.layers[i], h, attn_4d, position_ids,
                past_key_value=prelude_cache, use_cache=use_cache,
            )
        if return_hidden_trace:
            trace["after_prelude"] = h.detach().clone()

        # Encoded-input snapshot for LTI re-injection.
        e = h

        # ----- Recurrent loop -----
        # Optional gradient checkpointing of the loop body: at T=8 we re-enter
        # the recurrent block 8x in one forward pass, so activation memory
        # scales O(T) per layer in the block. Trainer sets
        # `self._checkpoint_loop=True` when the `--checkpoint-loop` flag is on;
        # backward re-executes each step's forward (cost ~30% wall-time) but
        # holds only one step's activations at a time. Phase 0 (force_gate_zero)
        # skips this -- there is no learnable param in that mode anyway.
        rec_layers = [base_model.layers[i] for i in cfg.block_layer_indices]
        gate_zero_effective = bool(force_gate_zero or force_bypass)
        # force_bypass also forces layerscale to zero so that the LTI
        # contribution is fully suppressed even when the recurrence cell
        # path runs (only relevant in block_mode where force_gate_zero
        # alone still passes block_out through the cell).
        prev_clamp = None
        if force_bypass:
            prev_clamp = self.recurrence.layerscale.clamp_max
            self.recurrence.layerscale.clamp_max = 0.0
        try:
            use_ckpt = (
                self.training
                and not gate_zero_effective
                and getattr(self, "_checkpoint_loop", False)
            )
            if use_cache:
                # T-iter cache path: each iteration t has its own DynamicCache
                # shared across the block's layers. Recurrence cell is
                # per-token and needs no cache.
                for t in range(T):
                    set_loop_t(t)
                    block_out = h
                    iter_cache = past_key_values[t]
                    for li, layer in enumerate(rec_layers):
                        block_out, _ = self._run_layer(
                            layer, block_out, attn_4d, position_ids,
                            past_key_value=iter_cache, use_cache=True,
                        )
                    if gate_zero_effective and not self.recurrence.block_mode:
                        h = block_out
                    else:
                        cell_out = self.recurrence(
                            h, e, block_out, t=t,
                            force_gate_zero=gate_zero_effective,
                        )
                        h = cell_out.h_next
            else:
                for t in range(T):
                    if use_ckpt:
                        h = torch.utils.checkpoint.checkpoint(
                            self._loop_step,
                            h, e, t, rec_layers, attn_4d, position_ids, gate_zero_effective,
                            use_reentrant=False,
                        )
                    else:
                        h = self._loop_step(
                            h, e, t, rec_layers, attn_4d, position_ids, gate_zero_effective
                        )
        finally:
            if force_bypass:
                self.recurrence.layerscale.clamp_max = prev_clamp
        if return_hidden_trace:
            trace["after_recurrence"] = h.detach().clone()

        # ----- Coda -----
        # Coda layers reuse the iteration-0 cache (slot[layer_idx] of it).
        coda_cache = past_key_values[0] if use_cache else None
        n_total = base_model.config.num_hidden_layers
        for i in range(n_total - cfg.coda_layers, n_total):
            h, _ = self._run_layer(
                base_model.layers[i], h, attn_4d, position_ids,
                past_key_value=coda_cache, use_cache=use_cache,
            )
        if return_hidden_trace:
            trace["after_coda"] = h.detach().clone()

        h = base_model.norm(h)
        if return_hidden_trace:
            trace["after_norm"] = h.detach().clone()

        logits = self.base.lm_head(h)

        # Compute next-token-prediction loss when labels are provided.
        # Following HF causal-LM convention: shift logits left vs labels by 1.
        # NB: do NOT cast logits to fp32 here. logits is [B, S, vocab=102400]
        # which is ~800 MB at bf16 / B=2 / S=2048 -- the .float() copy was
        # +1.6 GB peak and matched the v1 OOM exactly. cross_entropy_loss
        # accumulates internally in fp32 from a bf16 input with no precision
        # loss for our scale. Use reshape(-1, V) instead of contiguous+view
        # to avoid an extra copy of the gathered slice.
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            vocab = logits.size(-1)
            shift_logits = logits[:, :-1, :].reshape(-1, vocab)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(
                shift_logits, shift_labels, ignore_index=-100,
            )

        if return_hidden_trace:
            return {"logits": logits, "loss": loss, "trace": trace}

        # Default: HF-style structured output when called as a Trainer model
        # (labels passed -> loss field populated). Plain logits when called
        # directly without labels (probe / smoke / phase-0 paths).
        if return_dict or labels is not None or use_cache:
            return CausalLMOutputWithPast(
                loss=loss, logits=logits,
                past_key_values=past_key_values if use_cache else None,
            )
        return logits

    # ------------------------------------------------------------------
    # Convenience: run base's full forward through prelude+rec+coda only
    # ------------------------------------------------------------------

    @torch.no_grad()
    def base_three_layer_pass(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Manual reference pipeline: embed -> layer[0..prelude] ->
        layer[recurrent_idx] -> layer[-coda..] -> norm -> lm_head.

        Used by the Phase 0 sanity probe to compare against the wrapper
        with `T=1, force_gate_zero=True`. Should be bit-exact.
        """
        cfg = self.config
        base_model = self.base.model
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)
        inputs_embeds = base_model.embed_tokens(input_ids)

        if base_model._use_flash_attention_2:
            attn_4d = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            attn_4d = _prepare_4d_causal_attention_mask(
                attention_mask,
                (bsz, seq_len),
                inputs_embeds,
                past_key_values_length=0,
            )

        h = inputs_embeds
        for i in range(cfg.prelude_layers):
            h = self._run_layer(base_model.layers[i], h, attn_4d, position_ids)

        for idx in cfg.block_layer_indices:
            h = self._run_layer(
                base_model.layers[idx], h, attn_4d, position_ids
            )

        n_total = base_model.config.num_hidden_layers
        for i in range(n_total - cfg.coda_layers, n_total):
            h = self._run_layer(base_model.layers[i], h, attn_4d, position_ids)

        h = base_model.norm(h)
        return self.base.lm_head(h)


__all__ = ["MythicRDTDeepseekV2ForCausalLM"]
