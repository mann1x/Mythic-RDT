"""T-curriculum for the recurrence loop during fine-tune.

Per MASTER_PLAN.md §6 stability rule:
    Curriculum: T=2 -> T=4 -> T=8 -> T=16, mixed-T sampling per phase.

A phase is a (start_step, T_distribution) pair. The trainer queries
`curriculum.sample_T(global_step)` once per micro-batch and passes that T to
the wrapper's forward. Sampling is deterministic given (global_step, seed) so
checkpoint resume produces the same T sequence as a non-interrupted run.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CurriculumPhase:
    """A training phase: from `start_step` (inclusive) onward, sample T from
    `weights` (a dict T -> relative weight, normalized internally)."""
    start_step: int
    weights: dict[int, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("CurriculumPhase weights must be non-empty")
        for T, w in self.weights.items():
            if T < 1:
                raise ValueError(f"T must be >=1, got {T}")
            if w < 0:
                raise ValueError(f"weight must be >=0, got {w} for T={T}")
        if sum(self.weights.values()) <= 0:
            raise ValueError("at least one positive weight required")


class Curriculum:
    """Ordered list of phases. The active phase at `global_step` is the
    last phase whose `start_step <= global_step`.

    Sampling is deterministic per (seed, global_step):
        idx = hash(seed || step) % normalized_cumulative_weight
    """

    def __init__(self, phases: Sequence[CurriculumPhase], seed: int = 0xCAFE) -> None:
        if not phases:
            raise ValueError("Curriculum needs at least one phase")
        sorted_phases = sorted(phases, key=lambda p: p.start_step)
        if sorted_phases[0].start_step != 0:
            raise ValueError("First curriculum phase must have start_step=0")
        # Reject duplicate start_steps so iteration is unambiguous.
        seen: set[int] = set()
        for p in sorted_phases:
            if p.start_step in seen:
                raise ValueError(f"duplicate start_step {p.start_step}")
            seen.add(p.start_step)
        self.phases: list[CurriculumPhase] = list(sorted_phases)
        self.seed = int(seed)

    def active_phase(self, global_step: int) -> CurriculumPhase:
        active = self.phases[0]
        for p in self.phases:
            if p.start_step <= global_step:
                active = p
            else:
                break
        return active

    def sample_T(self, global_step: int, micro_batch_idx: int = 0) -> int:
        """Deterministic T sample. Adding micro_batch_idx lets gradient-
        accumulation steps within one optimizer step still see varied T while
        staying reproducible across runs."""
        phase = self.active_phase(global_step)
        # Compute a stable u in [0, 1) from (seed, step, micro_batch_idx).
        h = hashlib.sha256(
            struct.pack("<QQQ", self.seed & 0xFFFFFFFFFFFFFFFF,
                                 global_step & 0xFFFFFFFFFFFFFFFF,
                                 micro_batch_idx & 0xFFFFFFFFFFFFFFFF)
        ).digest()
        u_int = int.from_bytes(h[:8], "little")
        u = u_int / float(1 << 64)
        # Pick T by cumulative weight.
        items = sorted(phase.weights.items())
        total = sum(w for _, w in items)
        target = u * total
        acc = 0.0
        for T, w in items:
            acc += w
            if target < acc:
                return int(T)
        return int(items[-1][0])

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "phases": [
                {"start_step": p.start_step, "weights": p.weights}
                for p in self.phases
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Curriculum":
        return cls(
            phases=[
                CurriculumPhase(
                    start_step=int(p["start_step"]),
                    weights={int(k): float(v) for k, v in p["weights"].items()},
                )
                for p in d["phases"]
            ],
            seed=int(d.get("seed", 0xCAFE)),
        )


def default_curriculum(
    warmup_steps: int = 200,
    phase2_start: int = 1000,
    phase3_start: int = 3000,
) -> Curriculum:
    """The MASTER_PLAN.md curriculum: T=2 warm -> mixed {2,4} -> mixed {2,4,8}.

    Defaults sized for a moderate run (~5k-10k steps total). Override if
    running shorter smoke tests or longer production runs.
    """
    return Curriculum(
        phases=[
            CurriculumPhase(start_step=0, weights={2: 1.0}),
            CurriculumPhase(start_step=warmup_steps, weights={2: 0.7, 4: 0.3}),
            CurriculumPhase(start_step=phase2_start, weights={2: 0.4, 4: 0.4, 8: 0.2}),
            CurriculumPhase(start_step=phase3_start, weights={2: 0.2, 4: 0.3, 8: 0.5}),
        ]
    )


def v3_t1_only_curriculum() -> Curriculum:
    """v3-T1-isolation: every step samples T=1.

    Diagnostic curriculum used to test whether the wrapper at T=1 can
    reach base-loss-equivalent. If yes, the wrapper architecture +
    block_mode formula is sound and v4 can stack recurrence on top.
    If no, no amount of T>1 training will save it.
    """
    return Curriculum(phases=[CurriculumPhase(start_step=0, weights={1: 1.0})])


def v3_conservative_curriculum(
    t1_steps: int = 100,
    t2_start: int = 100,
) -> Curriculum:
    """v3-conservative: T=1 anchor warmup, then T=2 only.

    Designed for a feasibility test of the block_mode wrapper. Stays at
    low T to keep per-step compute manageable on a 19-layer block.

    - Step 0..t1_steps: T=1 only (anchor to base behavior since at T=1
      with block_mode, init wrapper ≈ base; gives gradient a clean target).
    - Step t1_steps..end: T=2 (one recurrent depth iteration on top of base).
    """
    return Curriculum(
        phases=[
            CurriculumPhase(start_step=0, weights={1: 1.0}),
            CurriculumPhase(start_step=t1_steps, weights={2: 1.0}),
        ]
    )


def v3_balanced_curriculum(
    t1_steps: int = 100,
    mix_start: int = 200,
    t4_dominant: int = 600,
) -> Curriculum:
    """v3-balanced: T=1 anchor -> mixed T=2 -> mixed T=2/4 -> T=4-heavy.

    Tests whether multiple iterations of the block produce a meaningful
    quality gain over T=1.
    """
    return Curriculum(
        phases=[
            CurriculumPhase(start_step=0, weights={1: 1.0}),
            CurriculumPhase(start_step=t1_steps, weights={1: 0.3, 2: 0.7}),
            CurriculumPhase(start_step=mix_start, weights={1: 0.2, 2: 0.4, 4: 0.4}),
            CurriculumPhase(start_step=t4_dominant, weights={2: 0.2, 4: 0.8}),
        ]
    )


def v4_anchored_curriculum(
    t1_steps: int = 80,
    mix12_start: int = 160,
    mix124_start: int = 280,
) -> Curriculum:
    """v4-anchored: T=1 anchor -> mixed T=1/2 -> mixed T=1/2/4 -> T=2/4-heavy.

    Sized for a ~400-step v4 first run (v3-balanced scaled down from ~700+).
    Stays within max-loop-iters=4 (T=8 deferred to v5).

    Pairs with KL-to-base anchor (--kl-anchor-alpha ~0.05, --kl-anchor-every ~8)
    to keep the trained wrapper close to base behavior as recurrence depth grows.
    The 1pp regression observed at v3-T1 (95% vs base 100%) motivates the anchor:
    each extra T iteration is another chance for LoRAs to push hidden states off
    the base manifold; KL keeps that bounded.
    """
    return Curriculum(
        phases=[
            CurriculumPhase(start_step=0, weights={1: 1.0}),
            CurriculumPhase(start_step=t1_steps, weights={1: 0.4, 2: 0.6}),
            CurriculumPhase(start_step=mix12_start, weights={1: 0.2, 2: 0.5, 4: 0.3}),
            CurriculumPhase(start_step=mix124_start, weights={2: 0.4, 4: 0.6}),
        ]
    )


__all__ = [
    "Curriculum",
    "CurriculumPhase",
    "default_curriculum",
    "v3_t1_only_curriculum",
    "v3_conservative_curriculum",
    "v3_balanced_curriculum",
    "v4_anchored_curriculum",
]
