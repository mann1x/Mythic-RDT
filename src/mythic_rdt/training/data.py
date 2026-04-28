"""Streaming data pipeline: FineWeb-Edu prose + a code dataset, packed.

Both sources are pulled via `datasets.load_dataset(..., streaming=True)` so
nothing is materialized to disk -- important for vast.ai pods with limited
storage and for long curriculum runs that can outlive any single download.

Packing strategy: simple greedy-fill. Stream tokens from the source iterator,
accumulate into a buffer, slice off `seq_len`-sized blocks. No EOS-on-sample
boundary so cross-sample attention is allowed -- standard for language model
pretrain. If we observe destabilization later we can add document-mask
attention; for v0 we don't.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch
from torch.utils.data import IterableDataset


@dataclass
class _SourceSpec:
    """One stream source: HF dataset id + config + the field that holds text.

    Optional `filter_field`/`filter_value`: if set, drop rows whose
    `row[filter_field] != filter_value` (used to grab Python-only rows from
    `bigcode/the-stack-smol`, which has a single `default` config and a
    `lang` field).
    """
    repo_id: str
    config: Optional[str]
    split: str
    text_field: str
    weight: float
    name: str
    filter_field: Optional[str] = None
    filter_value: Optional[str] = None


def _default_sources() -> list[_SourceSpec]:
    """The 70/30 prose/code mix.

    - `HuggingFaceFW/fineweb-edu` "default": standard educational-prose stream.
    - `bigcode/the-stack-smol` "data/python": small Python subset of The Stack.
      Gated -- requires HF org access (request at the dataset page) and a
      valid `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env var on the running host.
    """
    return [
        _SourceSpec(
            repo_id="HuggingFaceFW/fineweb-edu",
            config="default",
            split="train",
            text_field="text",
            weight=0.7,
            name="fineweb-edu",
        ),
        _SourceSpec(
            repo_id="bigcode/the-stack-smol",
            config=None,
            split="train",
            text_field="content",
            weight=0.3,
            name="the-stack-smol-python",
            filter_field="lang",
            filter_value="Python",
        ),
    ]


def _open_source(spec: _SourceSpec):
    """Lazy import so this file imports cleanly even without `datasets` (e.g.
    when running the lora_inject unit tests in a stripped env)."""
    from datasets import load_dataset
    kwargs = {"split": spec.split, "streaming": True}
    if spec.config is not None:
        return load_dataset(spec.repo_id, spec.config, **kwargs)
    return load_dataset(spec.repo_id, **kwargs)


class PackedDataset(IterableDataset):
    """Stream tokens from weighted sources, pack into seq_len blocks.

    Yields dicts with `input_ids` (LongTensor [seq_len]) and `labels` (same;
    Trainer handles the shift internally via causal-LM convention -- we just
    set labels = input_ids.clone() and let HF's loss do label-shift).

    Args:
        tokenizer: any HF tokenizer with `encode(text, add_special_tokens=...)`
            and a `bos_token_id`.
        seq_len: target sequence length (e.g. 2048).
        sources: list of _SourceSpec; uses default 70/30 mix if None.
        seed: RNG seed for source-mixing draws (so resume is reproducible).
        skip_blocks: skip the first N packed blocks at startup. Used as a
            poor-man's resume aid -- with streaming datasets we cannot truly
            seek, so we replay+drop. Trainer's resume sets this from
            global_step * batch_size * grad_accum.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        sources: Optional[list[_SourceSpec]] = None,
        seed: int = 0,
        skip_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.sources = sources or _default_sources()
        self.seed = int(seed)
        self.skip_blocks = int(skip_blocks)
        if self.skip_blocks > 0:
            print(
                f"[data] skip_blocks={self.skip_blocks} -- replaying stream "
                f"from start and dropping until skip is met."
            )

    def _interleave_text(self) -> Iterator[str]:
        rng = random.Random(self.seed)
        iters = []
        weights = []
        for spec in self.sources:
            ds = _open_source(spec)
            iters.append(iter(ds))
            weights.append(spec.weight)
        total = sum(weights)
        cum = []
        acc = 0.0
        for w in weights:
            acc += w / total
            cum.append(acc)
        while iters:
            u = rng.random()
            choice = 0
            for i, c in enumerate(cum):
                if u < c:
                    choice = i
                    break
            spec = self.sources[choice]
            try:
                # Apply per-source filter (e.g. lang=Python) by skipping rows
                # that don't match. Bounded inner loop so a long mismatch
                # streak doesn't starve the other sources.
                row = None
                for _ in range(2000):
                    candidate = next(iters[choice])
                    if spec.filter_field is None:
                        row = candidate
                        break
                    if candidate.get(spec.filter_field) == spec.filter_value:
                        row = candidate
                        break
                if row is None:
                    # 2000 consecutive non-matches -- yield to other sources
                    continue
            except StopIteration:
                # source exhausted; drop it and renormalize.
                iters.pop(choice)
                w_drop = weights.pop(choice)
                total -= w_drop
                if not iters or total <= 0:
                    return
                cum = []
                acc = 0.0
                for w in weights:
                    acc += w / total
                    cum.append(acc)
                continue
            text = row.get(spec.text_field, "")
            if isinstance(text, str) and text:
                yield text

    def __iter__(self) -> Iterator[dict]:
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        buf: list[int] = []
        if bos_id is not None:
            buf.append(int(bos_id))
        skipped = 0
        for text in self._interleave_text():
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            if eos_id is not None:
                buf.append(int(eos_id))
            while len(buf) >= self.seq_len:
                block = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                if skipped < self.skip_blocks:
                    skipped += 1
                    continue
                t = torch.as_tensor(block, dtype=torch.long)
                yield {"input_ids": t, "labels": t.clone()}


def build_packed_dataset(
    tokenizer,
    seq_len: int = 2048,
    sources: Optional[list[_SourceSpec]] = None,
    seed: int = 0,
    skip_blocks: int = 0,
) -> PackedDataset:
    return PackedDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        sources=sources,
        seed=seed,
        skip_blocks=skip_blocks,
    )


__all__ = ["PackedDataset", "build_packed_dataset", "_SourceSpec"]
