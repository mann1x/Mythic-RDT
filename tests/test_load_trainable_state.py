"""Tests for `_load_trainable_state` strictness — guards bug-050."""

from __future__ import annotations

import os
import unittest

import torch
import torch.nn as nn

from mythic_rdt.training.trainer import (
    CheckpointShapeMismatchError,
    _load_trainable_state,
    _trainable_state_dict,
)


class _Toy(nn.Module):
    """Stand-in for the wrapper: one trainable T-axis tensor + one scalar."""

    def __init__(self, T: int, hidden: int = 8):
        super().__init__()
        # Mimic a per-T LoRA-B style tensor with leading T axis.
        self.lora_B = nn.Parameter(torch.zeros(T, hidden, hidden))
        # Mimic gate.bias shape [T].
        self.gate_bias = nn.Parameter(torch.zeros(T))
        # Mimic a non-T-axis tensor (LTI A).
        self.lti_A = nn.Parameter(torch.eye(hidden))


class TestLoadTrainableStateStrict(unittest.TestCase):

    def setUp(self):
        # Make sure the env override is OFF for these tests.
        os.environ.pop("MYTHIC_LOAD_LENIENT", None)

    # ------------------------------------------------------------------
    # Happy paths

    def test_exact_shape_match_loads(self):
        src = _Toy(T=4)
        dst = _Toy(T=4)
        # Set source to non-zero so we can verify copy.
        with torch.no_grad():
            src.lora_B.fill_(0.5)
            src.gate_bias.fill_(1.5)
            src.lti_A.mul_(2.0)
        state = _trainable_state_dict(src)
        loaded, missing, unexpected = _load_trainable_state(dst, state)
        self.assertEqual(loaded, 3)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertTrue(torch.allclose(dst.lora_B, src.lora_B))
        self.assertTrue(torch.allclose(dst.gate_bias, src.gate_bias))
        self.assertTrue(torch.allclose(dst.lti_A, src.lti_A))

    def test_t_axis_expansion_loads(self):
        """Ckpt T=1 → wrapper T=4: leading slice copied, remainder kept at init.

        This is the v3-T1 → v4 transition we rely on for --init-from-checkpoint.
        """
        src = _Toy(T=1)
        dst = _Toy(T=4)
        with torch.no_grad():
            src.lora_B.fill_(0.7)
            src.gate_bias.fill_(2.5)
        state = _trainable_state_dict(src)
        loaded, missing, unexpected = _load_trainable_state(dst, state)
        self.assertEqual(loaded, 3)
        self.assertEqual(unexpected, [])
        # Leading T-slice copied:
        self.assertTrue(torch.allclose(dst.lora_B[0], src.lora_B[0]))
        self.assertEqual(float(dst.gate_bias[0]), 2.5)
        # Remaining slices stayed at fresh init (zero):
        self.assertTrue(torch.all(dst.lora_B[1:] == 0))
        self.assertTrue(torch.all(dst.gate_bias[1:] == 0))

    # ------------------------------------------------------------------
    # bug-050 guard: T-axis SHRINKAGE must raise in strict mode

    def test_t_axis_shrinkage_strict_raises(self):
        """Ckpt T=4 → wrapper T=1: bug-050. MUST raise in strict mode."""
        src = _Toy(T=4)
        dst = _Toy(T=1)
        with torch.no_grad():
            src.lora_B.fill_(0.9)
        state = _trainable_state_dict(src)
        with self.assertRaises(CheckpointShapeMismatchError) as cm:
            _load_trainable_state(dst, state)  # default strict=True
        msg = str(cm.exception)
        self.assertIn("T-axis=", msg)
        self.assertIn("max-loop-iters", msg)
        self.assertIn("bug-050", msg)

    def test_t_axis_shrinkage_lenient_does_not_raise(self):
        """strict=False: shrinkage tolerated, marked unexpected, no exception."""
        src = _Toy(T=4)
        dst = _Toy(T=1)
        state = _trainable_state_dict(src)
        loaded, missing, unexpected = _load_trainable_state(dst, state, strict=False)
        # lti_A still loads (no T axis); the two T-axis tensors are rejected.
        self.assertEqual(loaded, 1)
        # Both T-axis tensors should appear as unexpected with shrinkage marker.
        self.assertEqual(len(unexpected), 2)
        self.assertTrue(all("T-shrinkage" in u for u in unexpected))

    def test_env_override_forces_lenient(self):
        """MYTHIC_LOAD_LENIENT=1 turns strict=True into strict=False with a warning."""
        src = _Toy(T=4)
        dst = _Toy(T=1)
        state = _trainable_state_dict(src)
        os.environ["MYTHIC_LOAD_LENIENT"] = "1"
        try:
            loaded, missing, unexpected = _load_trainable_state(dst, state)  # default strict=True
        finally:
            os.environ.pop("MYTHIC_LOAD_LENIENT", None)
        # Did not raise; same outcome as strict=False above.
        self.assertEqual(loaded, 1)
        self.assertEqual(len(unexpected), 2)

    # ------------------------------------------------------------------
    # Other shape mismatch (not just T-axis) also raises

    def test_unrelated_shape_mismatch_strict_raises(self):
        """Hidden-dim mismatch on an otherwise-matching tensor name → raise."""
        src = _Toy(T=2, hidden=8)
        dst = _Toy(T=2, hidden=16)
        state = _trainable_state_dict(src)
        with self.assertRaises(CheckpointShapeMismatchError) as cm:
            _load_trainable_state(dst, state)
        self.assertIn("incompatible", str(cm.exception))

    # ------------------------------------------------------------------
    # Missing / unexpected (non-fatal) keys do not raise even in strict

    def test_unexpected_key_does_not_raise(self):
        """A ckpt key not present in the target is a warning, not fatal.

        (E.g., user removed a LoRA target between runs — load the rest.)
        """
        dst = _Toy(T=2)
        state = _trainable_state_dict(dst)
        # Add a bogus key not present in the destination module.
        state["nonexistent_param"] = torch.zeros(3)
        loaded, missing, unexpected = _load_trainable_state(dst, state)  # strict=True
        self.assertEqual(loaded, 3)
        self.assertEqual(missing, [])
        self.assertIn("nonexistent_param", unexpected)

    def test_missing_key_does_not_raise(self):
        """A target param that the ckpt lacks is a warning, not fatal."""
        src = _Toy(T=2)
        dst = _Toy(T=2)
        state = _trainable_state_dict(src)
        # Drop one param from the state — simulate a ckpt taken before this
        # tensor was added.
        del state["lti_A"]
        loaded, missing, unexpected = _load_trainable_state(dst, state)
        self.assertEqual(loaded, 2)
        self.assertEqual(unexpected, [])
        self.assertIn("lti_A", missing)


if __name__ == "__main__":
    unittest.main()
