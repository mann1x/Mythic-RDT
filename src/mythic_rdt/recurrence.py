"""Recurrence machinery for Mythic-RDT.

Implements the OpenMythos LTI injection + retrofit-recurrence stability
mechanisms that wrap a frozen base transformer block to produce a
recurrent-depth loop. All modules here are pure new code (no base-model
dependency) and unit-testable in isolation.

Math reference:
- Parcae (Prairie et al. 2026): A := Diag(-exp(log_A)) parameterization
  guarantees spectral radius rho(A) < 1, making the recurrence
  contractive and gradient-stable across many loop iterations.
- Retrofit-Recurrence (arXiv 2511.07384): identity-biased gating starts
  the loop as a near-identity (gate ~ 0) so the pretrained weights see
  approximately their original input distribution at fine-tune init.
- LayerScale (Touvron et al. CaiT 2021): per-loop scalar (init 1e-4)
  protects pretrained representations during early curriculum.

Recurrence step:
    inj = A * h_t + B * e
    block_out = RecurrentBlock(h_t, e)
    g = sigmoid(gate_t)
    h_{t+1} = h_t + ls_t * g * (inj + block_out)

At init (gate bias = -3, layerscale = 1e-4, log_A small, B near zero):
    g ~ 0.047, ls = 1e-4 -> ls * g ~ 5e-6
=> the recurrence is approximately h_{t+1} ~ h_t + epsilon, i.e. the loop
   is a near-identity. A T=1 forward with gate=0 (NOT trained, set to
   exactly 0) is exactly equivalent to running the recurrent block once
   and discarding the LTI/gate contributions -- the bit-exact sanity gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# LTI injection (Parcae spectral-radius parameterization)
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """Linear time-invariant injection: out = A * h + B * e.

    A is parameterized as Diag(-exp(log_A)) so its eigenvalues are all
    strictly negative and bounded; with this sign + the residual update
    used in the recurrence cell, the discrete-time spectral radius of the
    update operator is < 1 by construction (Parcae construction).

    B is a full hidden x hidden Linear initialized to ~0 (small noise) so
    the input-injection contribution starts negligibly small and grows
    through fine-tune.

    Args:
        hidden_size: residual stream dimensionality.
        log_a_init_low: lower bound of uniform init for log_A.
        log_a_init_high: upper bound of uniform init for log_A.
        b_init_std: std of B initialization (small for near-zero start).
    """

    def __init__(
        self,
        hidden_size: int,
        log_a_init_low: float = 0.01,
        log_a_init_high: float = 0.1,
        b_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # log_A parameterization: A = Diag(-exp(log_A))
        # log_A initialized small-positive so |A_ii| ~ exp(log_A) starts small.
        log_a = torch.empty(hidden_size).uniform_(log_a_init_low, log_a_init_high)
        self.log_a = nn.Parameter(log_a)

        # B: hidden x hidden Linear, init ~ N(0, b_init_std), no bias
        self.B = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.B.weight, mean=0.0, std=b_init_std)

    @property
    def A_diag(self) -> torch.Tensor:
        """Returns the diagonal of A, i.e. -exp(log_A). Always negative."""
        return -torch.exp(self.log_a)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Compute A * h + B * e, broadcasting A across batch and seq dims.

        Args:
            h: [batch, seq, hidden] current hidden state.
            e: [batch, seq, hidden] encoded input (re-injected each step).

        Returns:
            [batch, seq, hidden] injection contribution.
        """
        # Element-wise: h * A_diag (broadcasts over batch, seq).
        a_h = h * self.A_diag
        b_e = self.B(e)
        return a_h + b_e


# ---------------------------------------------------------------------------
# Identity-biased gate (retrofit-recurrence stability)
# ---------------------------------------------------------------------------


class IdentityBiasedGate(nn.Module):
    """Per-loop scalar gate, initialized so sigmoid(bias) ~ 0.

    With bias = -3 by default, sigmoid(-3) ~ 0.047. The gate multiplies
    the recurrence contribution (inj + block_out), so at init the loop
    barely moves the residual stream off its current trajectory, letting
    the pretrained weights see distributions close to their training.

    During curriculum fine-tune the gate's scalar bias parameter is
    learned; it can grow to open the gate fully (sigmoid ~ 1) for
    arbitrary T loop iterations.

    Args:
        n_iters: number of recurrence iterations T (one gate per iter).
        init_bias: initial bias (logit) for sigmoid. -3 -> ~ 0.047.
    """

    def __init__(self, n_iters: int, init_bias: float = 0.0) -> None:
        super().__init__()
        self.n_iters = n_iters
        # Per-iteration scalar bias; sigmoid maps to [0, 1] open factor.
        self.bias = nn.Parameter(torch.full((n_iters,), float(init_bias)))

    def forward(self, t: int) -> torch.Tensor:
        """Returns sigmoid(bias[t]) as a scalar tensor."""
        if t < 0 or t >= self.n_iters:
            raise IndexError(
                f"IdentityBiasedGate: t={t} out of range [0, {self.n_iters})"
            )
        return torch.sigmoid(self.bias[t])

    def all_open_factors(self) -> torch.Tensor:
        """Returns sigmoid(bias) for all T iterations as a [T] tensor."""
        return torch.sigmoid(self.bias)


# ---------------------------------------------------------------------------
# Per-loop LayerScale
# ---------------------------------------------------------------------------


class PerLoopLayerScale(nn.Module):
    """Per-loop scalar (or diagonal) LayerScale, init 1e-4.

    LayerScale (Touvron et al. CaiT) protects pretrained representations
    during early curriculum: the scale is so small at init that the
    recurrence contribution is dominated by the residual h_t, and the
    base block sees inputs nearly identical to its training distribution.

    Args:
        n_iters: number of recurrence iterations T (one scale per iter).
        hidden_size: if > 0, use a per-channel diagonal LayerScale of
            shape [hidden_size]; if 0, use a single scalar per iter.
        init_value: initial scale value (1e-4 standard).
    """

    def __init__(
        self,
        n_iters: int,
        hidden_size: int = 0,
        init_value: float = 1e-4,
        clamp_max: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.hidden_size = hidden_size
        self.clamp_max = clamp_max
        if hidden_size > 0:
            # Per-channel diagonal LayerScale: shape [T, hidden].
            self.scale = nn.Parameter(
                torch.full((n_iters, hidden_size), float(init_value))
            )
        else:
            # Single scalar per iter: shape [T].
            self.scale = nn.Parameter(torch.full((n_iters,), float(init_value)))

    def forward(self, t: int) -> torch.Tensor:
        """Returns scale[t] (scalar or [hidden] tensor)."""
        if t < 0 or t >= self.n_iters:
            raise IndexError(
                f"PerLoopLayerScale: t={t} out of range [0, {self.n_iters})"
            )
        s = self.scale[t]
        if self.clamp_max is not None:
            s = s.clamp(max=self.clamp_max)
        return s


# ---------------------------------------------------------------------------
# Depth-LoRA: per-iteration low-rank adapter on a Linear's weight
# ---------------------------------------------------------------------------


class DepthLoRA(nn.Module):
    """Per-iteration LoRA adapter for a Linear layer's weight.

    Applies T independent rank-r LoRA additions: at iteration t the
    adapted weight is  W + (B_t @ A_t) * scaling, where W is the frozen
    base weight (provided externally), and (A_t, B_t) are the two LoRA
    matrices for that iteration.

    LoRA convention: A is [r, in_features], B is [out_features, r], the
    delta is B @ A (shape [out, in]). At init A is Kaiming-uniform and B
    is zero, so the LoRA contribution is zero -- the base layer behaves
    identically to its frozen self until fine-tune populates B.

    Args:
        in_features: input dim of the wrapped Linear.
        out_features: output dim of the wrapped Linear.
        n_iters: number of recurrence iterations T (T independent LoRAs).
        rank: LoRA rank.
        alpha: LoRA scaling: scaling = alpha / rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_iters: int,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_iters = n_iters
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        # T independent LoRAs: A is [T, r, in], B is [T, out, r].
        self.lora_A = nn.Parameter(torch.empty(n_iters, rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(n_iters, out_features, rank))

        # Standard LoRA init: A ~ Kaiming-uniform, B = 0
        for t in range(n_iters):
            nn.init.kaiming_uniform_(self.lora_A[t], a=math.sqrt(5))
        # lora_B is already zeros

    def delta(self, t: int) -> torch.Tensor:
        """Returns the iteration-t weight delta: [out, in]."""
        if t < 0 or t >= self.n_iters:
            raise IndexError(
                f"DepthLoRA: t={t} out of range [0, {self.n_iters})"
            )
        return self.lora_B[t] @ self.lora_A[t] * self.scaling

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Compute LoRA contribution for input x at iteration t.

        Returns:
            x @ A_t.T @ B_t.T * scaling, shape [..., out_features].
        """
        if t < 0 or t >= self.n_iters:
            raise IndexError(
                f"DepthLoRA: t={t} out of range [0, {self.n_iters})"
            )
        # x @ A.T -> [..., r]; then @ B.T -> [..., out]; then * scaling
        h = torch.nn.functional.linear(x, self.lora_A[t])
        h = torch.nn.functional.linear(h, self.lora_B[t])
        return h * self.scaling


# ---------------------------------------------------------------------------
# Recurrence cell: ties LTI + gate + LayerScale together
# ---------------------------------------------------------------------------


@dataclass
class RecurrenceCellOutput:
    h_next: torch.Tensor
    gate_value: torch.Tensor  # sigmoid(bias_t) -- for logging
    injection: torch.Tensor  # A*h + B*e -- for inspection
    block_out: torch.Tensor  # RecurrentBlock(h, e) -- for inspection


class RecurrenceCell(nn.Module):
    """One iteration of the Mythic-RDT recurrence.

    Composes LTIInjection + IdentityBiasedGate + PerLoopLayerScale and
    applies the residual update:

        h_{t+1} = h_t + ls_t * g_t * (inj + block_out)

    The recurrent block itself is NOT owned by this module -- caller
    passes the per-iteration block_out tensor, which is computed
    externally (e.g. by running a frozen transformer block + depth-LoRA).
    This decoupling keeps the recurrence math pure and the cell unit-
    testable without any base model.

    Args:
        hidden_size: residual stream dim.
        n_iters: max number of recurrence iterations T.
        layerscale_init: init value for per-loop LayerScale.
        gate_init_bias: init bias for identity-biased gate.
        layerscale_per_channel: if True, use [T, hidden] LayerScale;
            else single scalar per iteration.
        lti_log_a_init_low / high / b_init_std: LTI init params.
    """

    def __init__(
        self,
        hidden_size: int,
        n_iters: int,
        layerscale_init: float = 1e-4,
        layerscale_clamp_max: Optional[float] = None,
        gate_init_bias: float = 0.0,
        layerscale_per_channel: bool = False,
        block_mode: bool = False,
        block_mode_residual: bool = False,
        lti_log_a_init_low: float = 0.01,
        lti_log_a_init_high: float = 0.1,
        lti_b_init_std: float = 1e-4,
        lti_residual_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_iters = n_iters
        self.block_mode = bool(block_mode)
        # v6W (2026-05-10, council finding): in block_mode_residual the LTI
        # contribution was previously dropped entirely, leaving A_diag/B_proj
        # as dead weight (no forward use, no gradient). Re-introduce it at a
        # small fixed scale so LTI parameters become productive without
        # re-introducing the unbounded ||h|| growth that motivated dropping
        # them. Default 0.0 = pre-v6W behavior (LTI dead). Try 0.01 in v6W.
        self.lti_residual_scale = float(lti_residual_scale)
        # Fix A (2026-04-29): when True AND block_mode=True, use h-residual
        # formula instead of the broken `h_next = block_out + ε·inj` form.
        # See memory/project_recurrence_root_cause_block_mode.md for the
        # drift-probe diagnosis. The residual formula bounds ||h|| growth
        # per iteration and keeps coda's input in its training distribution.
        self.block_mode_residual = bool(block_mode_residual)

        self.lti = LTIInjection(
            hidden_size=hidden_size,
            log_a_init_low=lti_log_a_init_low,
            log_a_init_high=lti_log_a_init_high,
            b_init_std=lti_b_init_std,
        )
        self.gate = IdentityBiasedGate(n_iters=n_iters, init_bias=gate_init_bias)
        self.layerscale = PerLoopLayerScale(
            n_iters=n_iters,
            hidden_size=hidden_size if layerscale_per_channel else 0,
            init_value=layerscale_init,
            clamp_max=layerscale_clamp_max,
        )

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        block_out: torch.Tensor,
        t: int,
        force_gate_zero: bool = False,
    ) -> RecurrenceCellOutput:
        """Execute one recurrence step.

        Args:
            h: [batch, seq, hidden] current hidden state.
            e: [batch, seq, hidden] encoded input (re-injected each step).
            block_out: [batch, seq, hidden] output of the recurrent
                transformer block at this iteration (computed externally).
            t: iteration index in [0, n_iters).
            force_gate_zero: if True, override gate to exactly 0 -- used
                for the bit-exact T=1 sanity gate (Phase 0).

        Returns:
            RecurrenceCellOutput with the next hidden state and traces.
        """
        injection = self.lti(h, e)
        gate_value = (
            torch.zeros((), device=h.device, dtype=h.dtype)
            if force_gate_zero
            else self.gate(t).to(h.dtype)
        )
        ls = self.layerscale(t).to(h.dtype)
        if self.block_mode:
            if self.block_mode_residual:
                # Fix A (2026-04-29): h-residual blend. mix ∈ [0, 1] gates
                # how much block_out displaces h per iteration:
                #   mix=0  -> h_next = h          (true near-identity)
                #   mix=1  -> h_next = block_out  (current v3+ behavior)
                #   mix=½  -> h_next = ½(h + block_out) (gradual update)
                # ||h|| stays bounded around its iter-0 value because each
                # iteration is a convex combination, not a replacement.
                # LTI injection — by default dropped (legacy behavior); v6W+
                # re-introduces it at a small FIXED scale (lti_residual_scale,
                # default 0.0). v6X (2026-05-12 council fix): the LTI add is
                # now INSIDE the `mix` blend so gate=0 -> h_next=h is true
                # bit-identity to base even when LTI is active. v6W placed
                # the LTI add OUTSIDE the gate, breaking the gate=0 identity
                # invariant and contributing to the T=4 LCB collapse:
                #   v6W (broken):
                #     h_next = h + mix*(block_out-h) + s·injection
                #     gate=0 -> h_next = h + s·injection ≠ h
                #   v6X (correct):
                #     h_next = h + mix*((block_out-h) + s·injection)
                #     gate=0 -> h_next = h (true identity)
                # 0.01 stays the recommended scale; under v6X gating the
                # trained gate can fully suppress LTI when needed, so the
                # noise mode that wrecked v6W generation is gated off.
                mix = (ls * gate_value).clamp(min=0.0, max=1.0)
                if self.lti_residual_scale != 0.0:
                    delta = (block_out - h) + self.lti_residual_scale * injection
                else:
                    delta = block_out - h
                h_next = h + mix * delta
            else:
                # v3+ original block_mode: h_next = block_out + ε·inj.
                # KNOWN BROKEN at T>=2 because there's no h-residual to
                # bound ||h|| growth across iterations (coda receives
                # ||h||~5x training distribution -> syntactic gibberish).
                # See memory/project_recurrence_root_cause_block_mode.md.
                # Kept for backward-compat with v6A and earlier ckpts.
                h_next = block_out + ls * gate_value * injection
        else:
            # v0-v2: original retrofit-recurrence formula. At gate≈0 the
            # iteration discards block_out (loop is near-identity). Works
            # only when the "block" is a single layer the model can afford
            # to skip.
            h_next = h + ls * gate_value * (injection + block_out)
        return RecurrenceCellOutput(
            h_next=h_next,
            gate_value=gate_value,
            injection=injection,
            block_out=block_out,
        )

    @torch.no_grad()
    def initial_open_factor(self) -> float:
        """Returns the initial gate*layerscale product at t=0 for logging.

        Sanity check: at default init, this should be ~ 5e-6 -- the loop
        starts as a near-identity by an order of magnitude.
        """
        g0 = float(self.gate(0).item())
        ls0 = self.layerscale(0)
        ls0_scalar = float(ls0.mean().item()) if ls0.dim() > 0 else float(ls0.item())
        return g0 * ls0_scalar
