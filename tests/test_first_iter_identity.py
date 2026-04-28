"""Tests for the v6A architectural invariant: first_iter_identity.

When `first_iter_identity=True`, the t=0 iteration of the recurrence loop
must return `block_out` directly (no LTI / gate / LayerScale add). This
makes the wrapper output at T=1 byte-for-byte equal to base.

We DON'T load the 31 GB DSCoder base for this test — empirically the
identity is already proven by the v6A LCB-10 smoke (T=1 = base = 40 %, see
`eval_results/v6a_untrained_lcb10.json`). What this test guards is that
**future refactors of the loop body cannot silently drop the invariant**
without failing the suite.

Two complementary checks:
  1. Config field is present, defaults False, accepts True.
  2. `_loop_step` source contains the t=0 short-circuit branch returning
     block_out under `first_iter_identity`.
"""

from __future__ import annotations

import inspect
import unittest

from mythic_rdt.configuration import MythicRDTDeepseekV2Config


class TestFirstIterIdentityConfig(unittest.TestCase):

    def test_default_false_for_backward_compat(self):
        cfg = MythicRDTDeepseekV2Config()
        self.assertFalse(cfg.first_iter_identity,
                         "Default must be False to preserve compatibility with "
                         "v3-T1, v4, v5 checkpoints whose LoRA-B / LTI / gate "
                         "weights were trained against the old (non-identity) loop body.")

    def test_explicit_true_persists(self):
        cfg = MythicRDTDeepseekV2Config(first_iter_identity=True)
        self.assertTrue(cfg.first_iter_identity)

    def test_explicit_false_persists(self):
        cfg = MythicRDTDeepseekV2Config(first_iter_identity=False)
        self.assertFalse(cfg.first_iter_identity)

    def test_field_survives_round_trip(self):
        cfg = MythicRDTDeepseekV2Config(first_iter_identity=True)
        d = cfg.to_dict()
        self.assertIn("first_iter_identity", d)
        self.assertEqual(d["first_iter_identity"], True)
        cfg2 = MythicRDTDeepseekV2Config(**{k: v for k, v in d.items() if k != "model_type"})
        self.assertTrue(cfg2.first_iter_identity)


class TestFirstIterIdentityLoopStep(unittest.TestCase):
    """Static-source check: locks the v6A invariant against accidental removal.

    If a future refactor drops the t=0 short-circuit (e.g. someone "cleans up"
    the conditional thinking it's dead code), this test fails loud BEFORE the
    next training run silently regresses the architecture.
    """

    def setUp(self):
        # Import lazily so a partial mythic_rdt installation (no torch/transformers)
        # doesn't crash this whole module.
        from mythic_rdt.modeling import MythicRDTDeepseekV2ForCausalLM
        self.src = inspect.getsource(MythicRDTDeepseekV2ForCausalLM._loop_step)

    def test_first_iter_identity_check_present(self):
        self.assertIn("first_iter_identity", self.src,
                      "_loop_step must reference first_iter_identity to honor v6A invariant")

    def test_t_eq_zero_short_circuit_present(self):
        # Either `t == 0` or `t==0`. Tighter than just searching for "0":
        # specifically the equality test on the iteration counter.
        self.assertTrue(
            "t == 0" in self.src or "t==0" in self.src,
            "_loop_step must short-circuit on t == 0 when first_iter_identity is set",
        )

    def test_returns_block_out_in_identity_branch(self):
        # The branch under first_iter_identity should `return block_out` (no
        # cell call, no recurrence add). Look for the exact return.
        self.assertIn("return block_out", self.src,
                      "_loop_step's first_iter_identity branch must return block_out directly")

    def test_uses_getattr_for_safe_default(self):
        # `getattr(self.config, "first_iter_identity", False)` is the right
        # idiom — it lets old config dicts (without the field) work without
        # crashing. Direct attribute access (self.config.first_iter_identity)
        # would break loading legacy v3/v4/v5 checkpoints.
        self.assertIn('getattr(self.config, "first_iter_identity"', self.src,
                      "Use getattr with default False to preserve backward-compat with "
                      "configs that predate the field (v3-T1 / v4 / v5 ckpts).")


if __name__ == "__main__":
    unittest.main()
