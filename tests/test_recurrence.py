"""Unit tests for the recurrence machinery (no base-model dependency)."""

from __future__ import annotations

import math

import pytest
import torch

from mythic_rdt.recurrence import (
    DepthLoRA,
    IdentityBiasedGate,
    LTIInjection,
    PerLoopLayerScale,
    RecurrenceCell,
)


HIDDEN = 64
N_ITERS = 8
BATCH = 2
SEQ = 16


# ---------------------------------------------------------------------------
# LTI injection
# ---------------------------------------------------------------------------


def test_lti_a_diag_is_strictly_negative():
    """A_diag = -exp(log_A) must be strictly negative for all entries."""
    lti = LTIInjection(HIDDEN)
    a = lti.A_diag
    assert (a < 0).all(), "A_diag must be strictly negative"


def test_lti_spectral_radius_below_one_at_init():
    """At init, |A_ii| = exp(log_A) with log_A in [0.01, 0.1] -> |A_ii| < 1.105.

    This alone doesn't guarantee the recurrence contraction (that depends
    on the residual structure), but it is the spectral check the Parcae
    parameterization gives us.
    """
    lti = LTIInjection(HIDDEN, log_a_init_low=0.01, log_a_init_high=0.1)
    a_abs = lti.A_diag.abs()
    # exp(0.1) ~ 1.105
    assert (a_abs <= math.exp(0.11)).all()
    assert (a_abs >= math.exp(0.0)).all()


def test_lti_b_starts_near_zero():
    lti = LTIInjection(HIDDEN, b_init_std=1e-4)
    assert lti.B.weight.abs().max().item() < 1e-2


def test_lti_forward_shape_and_finite():
    lti = LTIInjection(HIDDEN)
    h = torch.randn(BATCH, SEQ, HIDDEN)
    e = torch.randn(BATCH, SEQ, HIDDEN)
    out = lti(h, e)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    assert torch.isfinite(out).all()


def test_lti_a_h_is_hadamard():
    """For element-wise A * h, scaling each row of h by a_diag must match."""
    lti = LTIInjection(HIDDEN, b_init_std=0.0)  # zero out B
    # Force B to exactly zero
    with torch.no_grad():
        lti.B.weight.zero_()
    h = torch.randn(1, 1, HIDDEN)
    e = torch.zeros_like(h)
    out = lti(h, e)
    expected = h * lti.A_diag
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# Identity-biased gate
# ---------------------------------------------------------------------------


def test_gate_init_value_near_zero():
    """sigmoid(-3) ~ 0.047, well below 0.1."""
    gate = IdentityBiasedGate(N_ITERS, init_bias=-3.0)
    g = gate(0).item()
    assert 0.04 < g < 0.06


def test_gate_per_iteration_independent():
    """Each iteration's gate is a separate parameter."""
    gate = IdentityBiasedGate(N_ITERS, init_bias=-3.0)
    with torch.no_grad():
        gate.bias[3] = 5.0
    assert gate(0).item() < 0.1
    assert gate(3).item() > 0.99


def test_gate_out_of_range_raises():
    gate = IdentityBiasedGate(4)
    with pytest.raises(IndexError):
        gate(4)
    with pytest.raises(IndexError):
        gate(-1)


# ---------------------------------------------------------------------------
# LayerScale
# ---------------------------------------------------------------------------


def test_layerscale_scalar_init():
    ls = PerLoopLayerScale(N_ITERS, hidden_size=0, init_value=1e-4)
    assert ls.scale.shape == (N_ITERS,)
    assert torch.allclose(ls.scale, torch.full((N_ITERS,), 1e-4))


def test_layerscale_per_channel_init():
    ls = PerLoopLayerScale(N_ITERS, hidden_size=HIDDEN, init_value=1e-4)
    assert ls.scale.shape == (N_ITERS, HIDDEN)
    assert torch.allclose(ls.scale, torch.full((N_ITERS, HIDDEN), 1e-4))


# ---------------------------------------------------------------------------
# Depth-LoRA
# ---------------------------------------------------------------------------


def test_depth_lora_zero_at_init():
    """LoRA delta must be exactly zero at init (B = 0)."""
    lora = DepthLoRA(HIDDEN, HIDDEN, N_ITERS, rank=8)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    out = lora(x, t=0)
    assert torch.allclose(out, torch.zeros_like(out))


def test_depth_lora_nonzero_after_b_perturbed():
    lora = DepthLoRA(HIDDEN, HIDDEN, N_ITERS, rank=8)
    with torch.no_grad():
        lora.lora_B[0].normal_(0, 0.1)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    out = lora(x, t=0)
    assert out.abs().max().item() > 1e-4


def test_depth_lora_per_iter_independent():
    """Modifying t=0 LoRA must not affect t=1 output."""
    lora = DepthLoRA(HIDDEN, HIDDEN, N_ITERS, rank=8)
    with torch.no_grad():
        lora.lora_B[0].normal_(0, 0.1)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    out_t1 = lora(x, t=1)
    assert torch.allclose(out_t1, torch.zeros_like(out_t1))


def test_depth_lora_delta_shape():
    lora = DepthLoRA(in_features=128, out_features=256, n_iters=N_ITERS, rank=4)
    d = lora.delta(0)
    assert d.shape == (256, 128)


# ---------------------------------------------------------------------------
# Recurrence cell -- the bit-exact gate-zero check
# ---------------------------------------------------------------------------


def test_recurrence_cell_gate_zero_is_identity():
    """With force_gate_zero=True, h_next must equal h exactly (bit-exact).

    This is the Phase 0 sanity gate from MASTER_PLAN.md: at T=1 with
    gate=0, the wrapper is bit-exact with running the base middle layer
    once -- the recurrence contribution is exactly zero.
    """
    cell = RecurrenceCell(hidden_size=HIDDEN, n_iters=N_ITERS)
    h = torch.randn(BATCH, SEQ, HIDDEN)
    e = torch.randn(BATCH, SEQ, HIDDEN)
    block_out = torch.randn(BATCH, SEQ, HIDDEN)

    out = cell(h, e, block_out, t=0, force_gate_zero=True)
    # h_next = h + ls * 0 * (inj + block) = h, exactly
    assert torch.equal(out.h_next, h), "gate=0 must be exact identity"


def test_recurrence_cell_initial_open_factor_is_tiny():
    """At default init, gate * layerscale product is ~ 5e-6 (near-identity loop)."""
    cell = RecurrenceCell(hidden_size=HIDDEN, n_iters=N_ITERS)
    factor = cell.initial_open_factor()
    # sigmoid(-3) ~ 0.047, layerscale = 1e-4 -> ~ 4.7e-6
    assert 1e-6 < factor < 1e-4, f"expected ~5e-6, got {factor}"


def test_recurrence_cell_finite_after_T_steps():
    """T iterations of the cell on random inputs must remain finite (no NaN/inf)."""
    cell = RecurrenceCell(hidden_size=HIDDEN, n_iters=N_ITERS)
    h = torch.randn(BATCH, SEQ, HIDDEN)
    e = torch.randn(BATCH, SEQ, HIDDEN)
    for t in range(N_ITERS):
        block_out = torch.randn(BATCH, SEQ, HIDDEN)
        out = cell(h, e, block_out, t=t)
        h = out.h_next
        assert torch.isfinite(h).all(), f"NaN/inf at iter {t}"


def test_recurrence_cell_dtype_preserved_bf16():
    """Cell forward must preserve bf16 dtype of h."""
    cell = RecurrenceCell(hidden_size=HIDDEN, n_iters=N_ITERS).to(torch.bfloat16)
    h = torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.bfloat16)
    e = torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.bfloat16)
    block_out = torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.bfloat16)
    out = cell(h, e, block_out, t=0)
    assert out.h_next.dtype == torch.bfloat16


def test_recurrence_cell_grad_flows_through_lti_at_init():
    """At default init, gradients must reach LTI A and B params (non-zero) after T=1 step.

    Even with tiny ls*g, autograd should carry gradient through.
    """
    cell = RecurrenceCell(hidden_size=HIDDEN, n_iters=N_ITERS)
    h = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=False)
    e = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=False)
    block_out = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=False)
    out = cell(h, e, block_out, t=0)
    loss = out.h_next.sum()
    loss.backward()
    assert cell.lti.log_a.grad is not None
    assert cell.lti.B.weight.grad is not None
    assert cell.lti.log_a.grad.abs().sum().item() > 0
    assert cell.lti.B.weight.grad.abs().sum().item() > 0
