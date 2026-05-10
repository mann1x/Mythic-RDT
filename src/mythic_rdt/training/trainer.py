"""MythicRDTTrainer: a `transformers.Trainer` subclass that

  - samples a per-microbatch T from a `Curriculum` and threads it into the
    wrapper's forward (via the `T=` kwarg, not via the wrapper's training-mode
    default);
  - persists ONLY trainable state (recurrence cell + injected DepthLoRA params)
    on save, and restores them on load. The frozen base lives outside the
    checkpoint -- callers pass `--base` at resume time, same as at first run.
  - relies on Trainer's built-in SIGINT/SIGTERM handling and step-based save
    schedule for resume safety (per `feedback_finetune_resumable.md`).

`compute_loss` calls `self.model(input_ids=..., labels=..., T=t)`. Our wrapper
supports `T=` and returns a `CausalLMOutputWithPast` with `.loss` populated
when labels are present, so Trainer's normal training loop just works.

Save/load: by default Trainer saves the full state_dict via
`model.save_pretrained(output_dir)`. Our wrapper does NOT subclass
`PreTrainedModel`, so we override `_save` / `_load_from_checkpoint` to
torch.save({trainable_state, curriculum, config}) into a single file.

Why not subclass PreTrainedModel: the base inside is itself a
trust_remote_code PreTrainedModel; nesting two PreTrainedModels and trying
to round-trip both is messy. The Phase 4 publish path will vendor a clean
PreTrainedModel; for Phase 1 fine-tune we just need to keep training state
durable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn

try:
    from transformers import Trainer, TrainingArguments
except ImportError:  # pragma: no cover -- Trainer is required at runtime
    Trainer = None  # type: ignore
    TrainingArguments = None  # type: ignore

from .curriculum import Curriculum
from .lora_inject import LoRAInjectedLinear


# Filename used inside each checkpoint dir for our trainable-only state.
TRAINABLE_STATE_FN = "mythic_rdt_trainable.pt"
CURRICULUM_STATE_FN = "mythic_rdt_curriculum.json"


def count_trainable(module: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for diagnostic logging."""
    total = 0
    train = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            train += n
    return train, total


def _trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Pull only parameters with requires_grad=True. Buffers (e.g. LayerNorm
    running_mean) are NOT included -- the wrapper has none of those that
    need persisting (the base's are part of the frozen base, restored at
    load time from the same base path)."""
    out: dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        if p.requires_grad:
            out[name] = p.detach().cpu().clone()
    return out


class CheckpointShapeMismatchError(RuntimeError):
    """Raised by `_load_trainable_state(strict=True)` when checkpoint tensors
    cannot be loaded into the target module without silent data loss.

    The most common (and dangerous) case is **T-axis shrinkage**: the
    checkpoint was trained with a larger `max_loop_iters` than the wrapper
    being loaded into. Skipping those tensors silently loads almost nothing
    and the wrapper runs at init values ≈ base behavior — which produces
    false-positive eval scores that look like "wrapper preserves base"
    (bug-050, 2026-04-28). Always raise unless the caller explicitly opts out.
    """


def _load_trainable_state(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> tuple[int, list[str], list[str]]:
    """Load trainable params back. Returns (loaded_count, missing, unexpected).

    `missing` are trainable params not present in the checkpoint (a freshly
    added LoRA adapter, for example). `unexpected` are keys in the checkpoint
    that no current parameter matches (e.g. you removed a LoRA target) or
    that have an incompatible shape.

    Cross-architecture T-axis **expansion** (ckpt smaller, target bigger) is
    always supported: copy ckpt slices into the leading T-slices of target,
    leave higher slices at fresh init. This preserves bit-exact T=1 behavior
    of the loaded checkpoint while leaving room for new T-iter slices to
    learn. Used by `--init-from-checkpoint` for v3-T1 → v4 transitions.

    T-axis **shrinkage** (ckpt bigger, target smaller) and any other
    shape mismatch is REJECTED:
      - `strict=True` (default): raises `CheckpointShapeMismatchError` with
        an actionable message naming the offending tensors and pointing at
        `--max-loop-iters` as the usual fix.
      - `strict=False`: appended to `unexpected` with a "shape mismatch"
        marker; load continues with that tensor at fresh init. ONLY pass
        `strict=False` from a caller that has explicitly verified the
        scenario is safe (very rare — usually it isn't).

    Setting env var `MYTHIC_LOAD_LENIENT=1` forces `strict=False` regardless
    of the argument — last-resort escape hatch, prints a loud warning.
    """
    import os
    env_lenient = os.environ.get("MYTHIC_LOAD_LENIENT", "").strip() in ("1", "true", "True", "yes")
    if env_lenient and strict:
        print(
            "[trainer] WARNING: MYTHIC_LOAD_LENIENT=1 set — _load_trainable_state "
            "is running in lenient mode. Silent shape mismatches WILL be tolerated. "
            "This can produce false-positive eval scores (see bug-050)."
        )
        strict = False

    loaded = 0
    missing: list[str] = []
    unexpected: list[str] = []
    fatal: list[str] = []  # populated regardless of strict; raised at end if strict
    own = dict(module.named_parameters())
    own_trainable = {n: p for n, p in own.items() if p.requires_grad}
    for n, t in state.items():
        if n not in own_trainable:
            unexpected.append(n)
            continue
        target = own_trainable[n]
        t_dev = t.to(target.device).to(target.dtype)
        if t_dev.shape == target.shape:
            with torch.no_grad():
                target.copy_(t_dev)
            loaded += 1
        elif (t_dev.dim() == target.dim()
              and t_dev.shape[1:] == target.shape[1:]
              and t_dev.shape[0] < target.shape[0]):
            # T-axis EXPANSION (ckpt smaller, target bigger): supported, copy in.
            with torch.no_grad():
                target[: t_dev.shape[0]].copy_(t_dev)
            loaded += 1
        elif (t_dev.dim() == target.dim()
              and t_dev.shape[1:] == target.shape[1:]
              and t_dev.shape[0] > target.shape[0]):
            # T-axis SHRINKAGE (ckpt bigger, target smaller): bug-050 trap.
            msg = (
                f"{n}: ckpt T-axis={t_dev.shape[0]} > target T-axis={target.shape[0]} "
                f"(shapes {tuple(t.shape)} vs {tuple(target.shape)})"
            )
            fatal.append(msg)
            unexpected.append(f"{n} T-shrinkage {tuple(t.shape)} > {tuple(target.shape)}")
        else:
            msg = (
                f"{n}: shape {tuple(t.shape)} incompatible with target {tuple(target.shape)}"
            )
            fatal.append(msg)
            unexpected.append(f"{n} shape {tuple(t.shape)} != {tuple(target.shape)}")
    for n in own_trainable:
        if n not in state:
            missing.append(n)

    if strict and fatal:
        # Detect the canonical bug-050 case to emit the actionable hint.
        has_t_shrinkage = any("T-axis=" in m for m in fatal)
        n_show = min(5, len(fatal))
        details = "\n  ".join(fatal[:n_show])
        more = f"\n  ... and {len(fatal) - n_show} more" if len(fatal) > n_show else ""
        hint = ""
        if has_t_shrinkage:
            hint = (
                "\n\nFIX: the checkpoint was trained with a larger max_loop_iters "
                "than the current wrapper.\n"
                "  - In humaneval_smoke.py: pass --T-values that includes the "
                "checkpoint's training T_max (e.g. `--T-values 1 2 4` for a v4 "
                "ckpt), OR pass `--max-loop-iters 4` explicitly.\n"
                "  - In finetune_phase1.py: raise --max-loop-iters to match the "
                "init checkpoint, OR remove --init-from-checkpoint.\n"
                "Reference: bug-050 in `.wolf/buglog.json`, "
                "`memory/feedback_smoke_max_loop_iters.md`."
            )
        raise CheckpointShapeMismatchError(
            f"_load_trainable_state: {len(fatal)} tensor(s) cannot be loaded "
            f"into the current wrapper without silent fallback to init values:\n"
            f"  {details}{more}{hint}\n\n"
            f"To override (NOT recommended), pass strict=False or set "
            f"MYTHIC_LOAD_LENIENT=1."
        )

    return loaded, missing, unexpected


if Trainer is not None:

    class MythicRDTTrainer(Trainer):
        """Trainer that injects a per-microbatch T from a Curriculum and
        persists only trainable state."""

        def __init__(
            self,
            *args,
            curriculum: Curriculum,
            kl_anchor_alpha: float = 0.0,
            kl_anchor_every: int = 0,
            margin_alpha: float = 0.0,
            margin_nats: float = 0.02,
            distill_alpha: float = 0.0,
            dual_t_lo: int = 1,
            dual_t_hi: int = 0,  # 0 means: use config.max_loop_iters
            focal_gamma: float = 0.0,
            teacher_distill_alpha: float = 0.0,
            teacher_logits_path: Optional[str] = None,
            teacher_distill_temperature: float = 1.0,
            teacher_refinement_mask: bool = False,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.curriculum = curriculum
            self._microbatch_counter: int = 0
            self._global_block_counter: int = 0
            self._block_counter_initialized: bool = False
            self.kl_anchor_alpha = float(kl_anchor_alpha)
            self.kl_anchor_every = int(kl_anchor_every)
            self.margin_alpha = float(margin_alpha)
            self.margin_nats = float(margin_nats)
            self.distill_alpha = float(distill_alpha)
            self.dual_t_lo = int(dual_t_lo)
            self.dual_t_hi = int(dual_t_hi)
            self.focal_gamma = float(focal_gamma)
            self.teacher_distill_alpha = float(teacher_distill_alpha)
            self.teacher_distill_temperature = float(teacher_distill_temperature)
            self.teacher_refinement_mask = bool(teacher_refinement_mask)
            self._teacher_indices = None  # [N, L, K] int32, CPU
            self._teacher_values = None   # [N, L, K] bf16, CPU
            self._teacher_alignment_mask = None  # [N, L] bool, CPU — cross-vocab caches only
            self._teacher_meta = None
            if self.teacher_distill_alpha > 0 and teacher_logits_path:
                cache = torch.load(teacher_logits_path, map_location="cpu", weights_only=False)
                self._teacher_indices = cache["indices"]
                self._teacher_values = cache["values"]
                # NEW (v6V): cross-vocab caches store an alignment_mask field
                # [N, L] bool indicating which student positions actually
                # received a valid projected teacher logit. Same-vocab caches
                # don't have this — every position is aligned by construction.
                self._teacher_alignment_mask = cache.get("alignment_mask")
                self._teacher_meta = cache.get("meta", {})
                xv_marker = ""
                if self._teacher_alignment_mask is not None:
                    cov = float(self._teacher_alignment_mask.float().mean().item())
                    xv_marker = f" alignment_mask=ON coverage={cov:.2%}"
                print(f"[trainer] teacher distill active: alpha={self.teacher_distill_alpha} "
                      f"T={self.teacher_distill_temperature} "
                      f"refinement_mask={self.teacher_refinement_mask} "
                      f"cache_shape={tuple(self._teacher_indices.shape)}{xv_marker} "
                      f"meta={self._teacher_meta}")
            self.dual_t_active = (self.margin_alpha > 0.0 or self.distill_alpha > 0.0
                                   or self.teacher_distill_alpha > 0.0)
            if self.kl_anchor_alpha > 0 and self.kl_anchor_every > 0:
                print(f"[trainer] KL-to-base anchor active: "
                      f"alpha={self.kl_anchor_alpha}, every={self.kl_anchor_every} steps")
            if self.dual_t_active:
                print(f"[trainer] dual-T training active: lo={self.dual_t_lo}, "
                      f"hi={self.dual_t_hi or 'max_loop_iters'}, "
                      f"margin_alpha={self.margin_alpha}, margin_nats={self.margin_nats}, "
                      f"distill_alpha={self.distill_alpha}")
            if self.focal_gamma > 0:
                print(f"[trainer] focal CE active on T=4 path: gamma={self.focal_gamma} "
                      f"(weight = (1 - p_T1_correct)^gamma per token)")

        # ------------------------------------------------------------------
        # Forward + loss
        # ------------------------------------------------------------------

        def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
            """Override: sample T from curriculum (or run dual-T), pass into wrapper.

            v4 path (dual_t_active=False): sample one T per microbatch from the
            curriculum, single forward, optional KL-to-base anchor.

            v5 path (dual_t_active=True): two forward passes per microbatch at
            t_lo and t_hi. Loss = mean(CE_lo, CE_hi) + margin_alpha * ReLU(CE_hi
            - CE_lo + margin_nats) + distill_alpha * KL(lo || hi.detach()).
            The margin term forces high-T to outperform low-T in CE; the
            distill term forces low-T's output distribution to copy high-T's.
            Together they create the T-specialization gradient signal that v4
            lacked (probe results 2026-04-28: T=4 actively worse than T=1
            because no loss term punished that).
            """
            self._microbatch_counter += 1

            if self.dual_t_active:
                t_lo = max(1, self.dual_t_lo)
                t_hi = self.dual_t_hi or int(getattr(model.config, "max_loop_iters", 4))
                if t_hi <= t_lo:
                    t_hi = t_lo + 1  # guard

                out_lo = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs.get("labels"),
                    T=t_lo,
                )
                out_hi = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs.get("labels"),
                    T=t_hi,
                )
                ce_lo = out_lo.loss if hasattr(out_lo, "loss") else out_lo["loss"]

                # T=4 CE: optionally apply focal weighting using T=1 model's
                # per-token confidence on the ground-truth token. Concentrates
                # the T=4 gradient on tokens where T=1 is uncertain (= where
                # recurrence has room to add value). T=1 path stays unweighted
                # so it still trains uniformly. See cerebrum 2026-04-29
                # design notes for the rationale and risks.
                if self.focal_gamma > 0 and "labels" in inputs and inputs.get("labels") is not None:
                    hi_logits = out_hi.logits  # [B, L, V]
                    labels = inputs["labels"]   # [B, L], -100 = ignore
                    # next-token shift (HF causal LM convention)
                    shift_logits = hi_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    valid = shift_labels != -100
                    # T=1 confidence on the correct token (no_grad: weight only)
                    with torch.no_grad():
                        lo_logits_shift = out_lo.logits[..., :-1, :].contiguous()
                        # cast to fp32 for numerically-stable softmax under bf16
                        p_lo = torch.softmax(lo_logits_shift.float(), dim=-1)
                        gt_idx = shift_labels.clamp(min=0).unsqueeze(-1)
                        p_correct = p_lo.gather(-1, gt_idx).squeeze(-1)  # [B, L-1]
                    focal_w = (1.0 - p_correct).pow(self.focal_gamma) * valid.float()
                    # per-token CE on the T=4 path (no reduction)
                    ce_per_tok = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).view(shift_labels.shape)
                    # focal-weighted mean (normalize by sum of weights, not token
                    # count, so loss scale is comparable to standard CE)
                    w_sum = focal_w.sum().clamp(min=1.0)
                    ce_hi = (focal_w * ce_per_tok).sum() / w_sum
                    try:
                        self.log({
                            "focal_w_mean": float(focal_w[valid].mean().detach().item()),
                            "focal_w_max": float(focal_w[valid].max().detach().item()),
                        })
                    except Exception:
                        pass
                else:
                    ce_hi = out_hi.loss if hasattr(out_hi, "loss") else out_hi["loss"]

                loss = 0.5 * (ce_lo + ce_hi)

                # Margin term: penalize when hi-T isn't strictly better than lo-T.
                if self.margin_alpha > 0:
                    margin_term = torch.nn.functional.relu(ce_hi - ce_lo + self.margin_nats)
                    loss = loss + self.margin_alpha * margin_term
                    try:
                        self.log({"margin_loss": float(margin_term.detach().item())})
                    except Exception:
                        pass

                # Teacher distill term: sparse top-K KL between wrapper T_hi and
                # precomputed BF16 base teacher logits (same data_seed → same
                # block order). Gives gradient on EVERY token's distribution
                # (not just argmax / hard-only), which addresses the focal-γ
                # "starves easy correct tokens" problem at root.
                if self.teacher_distill_alpha > 0 and self._teacher_indices is not None:
                    B = inputs["input_ids"].shape[0]
                    # Initialize block counter to match resume position on first call.
                    if not self._block_counter_initialized:
                        eff_batch = (self.args.per_device_train_batch_size
                                     * max(1, self.args.gradient_accumulation_steps))
                        self._global_block_counter = int(self.state.global_step) * eff_batch
                        self._block_counter_initialized = True
                    blk_lo = self._global_block_counter
                    blk_hi = blk_lo + B
                    cache_n = self._teacher_indices.shape[0]
                    if blk_hi > cache_n:
                        # Past end of cache: skip teacher loss this batch.
                        pass
                    else:
                        # Slice cache → GPU. Shapes [B, L, K].
                        t_idx = self._teacher_indices[blk_lo:blk_hi].to(out_hi.logits.device)
                        t_val = self._teacher_values[blk_lo:blk_hi].to(out_hi.logits.device)
                        # Shift to next-token (HF causal LM convention) so we
                        # match labels[:,1:].
                        hi_logits = out_hi.logits[..., :-1, :].contiguous()
                        t_idx_s = t_idx[..., :-1, :].contiguous().long()
                        t_val_s = t_val[..., :-1, :].contiguous().float()
                        labels_s = (inputs.get("labels")[..., 1:].contiguous()
                                    if inputs.get("labels") is not None
                                    else torch.zeros_like(t_idx_s[..., 0]))
                        valid = (labels_s != -100)
                        # v6V: AND in the alignment_mask if the cache has one.
                        # Cross-vocab caches mark positions where projection
                        # failed (alignment skipped) with mask=False; we MUST
                        # exclude those from the loss or they contribute pure
                        # zeros and dilute the gradient.
                        if self._teacher_alignment_mask is not None:
                            am = self._teacher_alignment_mask[blk_lo:blk_hi].to(
                                out_hi.logits.device
                            )
                            am_s = am[..., :-1].contiguous()  # [B, L-1]
                            valid = valid & am_s
                        # Refinement mask (v6R+): only apply distill on tokens where
                        # the wrapper at T_lo disagrees with the teacher's top-1
                        # — i.e. the tokens we already know T=1 gets wrong. This
                        # focuses the distill gradient on tokens where recurrence
                        # has room to refine, instead of anchoring every token
                        # (including agreement-majority) to the teacher.
                        refine_mask = None
                        if self.teacher_refinement_mask:
                            lo_logits_shift = out_lo.logits[..., :-1, :].contiguous()
                            lo_argmax = lo_logits_shift.argmax(dim=-1)         # [B, L-1]
                            teacher_top1 = t_idx_s[..., 0]                     # [B, L-1]
                            refine_mask = (lo_argmax != teacher_top1)          # disagreement
                            valid = valid & refine_mask
                        # Gather wrapper logits at teacher's top-K vocab indices.
                        student_topk = hi_logits.gather(-1, t_idx_s)  # [B, L-1, K]
                        # Temperature-scaled softmax over the K-vocab subset.
                        T_temp = self.teacher_distill_temperature
                        log_q = torch.nn.functional.log_softmax(student_topk.float() / T_temp, dim=-1)
                        p = torch.nn.functional.softmax(t_val_s / T_temp, dim=-1)
                        # KL(p || q) per token.
                        kl_per_tok = (p * (p.clamp(min=1e-12).log() - log_q)).sum(dim=-1)
                        n_valid = valid.sum().clamp(min=1)
                        kl_distill = (kl_per_tok * valid.float()).sum() / n_valid
                        # Hinton T^2 scaling so alpha is comparable to CE units.
                        loss = loss + self.teacher_distill_alpha * (T_temp * T_temp) * kl_distill
                        try:
                            log_dict = {
                                "teacher_distill_loss": float(kl_distill.detach().item()),
                                "teacher_block_idx": float(blk_lo),
                            }
                            if refine_mask is not None:
                                # frac of valid (non-padded) tokens where wrapper T_lo
                                # disagrees with teacher top-1 = the tokens carrying
                                # distill loss. Lower = wrapper closer to teacher already.
                                base_valid = (labels_s != -100)
                                log_dict["teacher_refine_mask_frac"] = float(
                                    (refine_mask & base_valid).sum().float()
                                    / base_valid.sum().clamp(min=1).float()
                                )
                            self.log(log_dict)
                        except Exception:
                            pass
                    self._global_block_counter += B

                # Distill term: KL(lo || hi.detach()) per token.
                if self.distill_alpha > 0:
                    lo_logits = out_lo.logits
                    hi_logits_detached = out_hi.logits.detach()
                    distill = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(lo_logits, dim=-1),
                        torch.nn.functional.softmax(hi_logits_detached, dim=-1),
                        reduction="batchmean",
                    ) / lo_logits.size(1)
                    loss = loss + self.distill_alpha * distill
                    try:
                        self.log({"distill_loss": float(distill.detach().item())})
                    except Exception:
                        pass

                # Always log per-T CE so wandb shows the spread (probe-3 visibility).
                try:
                    self.log({
                        "ce_lo_T": float(ce_lo.detach().item()),
                        "ce_hi_T": float(ce_hi.detach().item()),
                        "ce_gap": float((ce_hi - ce_lo).detach().item()),
                        "T_lo": float(t_lo),
                        "T_hi": float(t_hi),
                    })
                except Exception:
                    pass

                # Outputs returned to caller = hi-T outputs (so KL anchor below
                # operates on the higher-T variant if also enabled — keeps anchor
                # honest about the wrapper's "main" output during dual-T training).
                outputs = out_hi
            else:
                T = self.curriculum.sample_T(
                    global_step=self.state.global_step,
                    micro_batch_idx=self._microbatch_counter,
                )
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs.get("labels"),
                    T=T,
                )
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

            # KL-to-base anchor (v4+): periodically pull wrapper logits toward
            # the frozen base's logits on the same batch. Uses force_bypass on
            # the same wrapper instance (no extra model resident) under no_grad
            # for the base pass.
            if (self.kl_anchor_alpha > 0
                    and self.kl_anchor_every > 0
                    and (self.state.global_step % self.kl_anchor_every == 0)):
                with torch.no_grad():
                    base_out = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        T=1,
                        force_bypass=True,
                        return_dict=True,
                    )
                w_logits = outputs.logits
                b_logits = base_out.logits if hasattr(base_out, "logits") else base_out
                # KL(base || wrapper) per-token. `batchmean` reduction sums over
                # (seq, vocab) and divides by batch, so divide by seq_len here
                # to get per-token-position average KL nats — same unit as CE
                # loss. Lets `kl_anchor_alpha` be sized in the same scale as
                # the LM loss (alpha=0.05 = 5% of per-token KL pressure).
                kl = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(w_logits, dim=-1),
                    torch.nn.functional.softmax(b_logits, dim=-1),
                    reduction="batchmean",
                ) / w_logits.size(1)
                loss = loss + self.kl_anchor_alpha * kl
                # One-shot log per anchored step (visible in wandb as kl_anchor).
                try:
                    self.log({"kl_anchor": float(kl.detach().item())})
                except Exception:
                    pass

            return (loss, outputs) if return_outputs else loss

        # Reset the microbatch counter every optimizer step so it tracks
        # micro-batches WITHIN a step, not globally.
        def training_step(self, model, inputs, num_items_in_batch=None):
            self._microbatch_counter = 0
            return super().training_step(model, inputs, num_items_in_batch)

        # ------------------------------------------------------------------
        # Save / load: trainable-only
        # ------------------------------------------------------------------

        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            output_dir = output_dir or self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            # 1. Trainable params (LTI A/B, gate, LayerScale, all DepthLoRA pairs).
            train_state = _trainable_state_dict(self.model)
            torch.save(train_state, os.path.join(output_dir, TRAINABLE_STATE_FN))
            # 2. Curriculum (small JSON; survives across schema bumps).
            with open(os.path.join(output_dir, CURRICULUM_STATE_FN), "w", encoding="utf-8") as f:
                json.dump(self.curriculum.to_dict(), f, indent=2)
            # 3. The wrapper's MythicRDTConfig (so we can re-build the
            #    architecture at resume without re-passing every flag).
            try:
                cfg = self.model.config
                with open(os.path.join(output_dir, "mythic_rdt_config.json"), "w", encoding="utf-8") as f:
                    json.dump(cfg.to_dict(), f, indent=2, default=str)
            except Exception as exc:
                print(f"[trainer] WARN: could not serialize wrapper config: {exc}")
            # 4. A short manifest with counts + LoRA records, for human audit.
            train_n, total_n = count_trainable(self.model)
            lora_records = []
            for name, mod in self.model.named_modules():
                if isinstance(mod, LoRAInjectedLinear):
                    lora_records.append({
                        "name": name,
                        "in_features": mod.in_features,
                        "out_features": mod.out_features,
                        "rank": mod.lora.rank,
                        "n_iters": mod.lora.n_iters,
                    })
            manifest = {
                "trainable_params": train_n,
                "total_params": total_n,
                "lora_modules": lora_records,
                "global_step": int(self.state.global_step),
            }
            with open(os.path.join(output_dir, "mythic_rdt_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # Trainer also expects training_args.bin so resume can reconstruct
            # TrainingArguments; let parent handle that and the trainer state
            # (global_step, epoch, etc.).
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
            model = model if model is not None else self.model
            ckpt = Path(resume_from_checkpoint)
            train_state_path = ckpt / TRAINABLE_STATE_FN
            if not train_state_path.exists():
                # Fall back to parent for full-state checkpoints (none expected).
                return super()._load_from_checkpoint(resume_from_checkpoint, model)
            state = torch.load(train_state_path, map_location="cpu", weights_only=True)
            loaded, missing, unexpected = _load_trainable_state(model, state)
            print(
                f"[trainer] resumed trainable state: loaded={loaded} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing:
                print(f"[trainer]   missing (first 5): {missing[:5]}")
            if unexpected:
                print(f"[trainer]   unexpected (first 5): {unexpected[:5]}")
            curr_path = ckpt / CURRICULUM_STATE_FN
            if curr_path.exists():
                with open(curr_path, "r", encoding="utf-8") as f:
                    self.curriculum = Curriculum.from_dict(json.load(f))
                print(f"[trainer]   curriculum restored ({len(self.curriculum.phases)} phases)")

else:
    MythicRDTTrainer = None  # type: ignore


def build_training_args(
    output_dir: str,
    seq_len: int,
    per_device_batch: int,
    grad_accum: int,
    max_steps: int,
    save_steps: int,
    learning_rate: float,
    warmup_steps: int,
    bf16: bool = True,
    report_to: Optional[list[str]] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    """Construct TrainingArguments tuned for resumable Phase 1 fine-tune.

    Per memory rule:
      - save_strategy="steps", save_steps=N, save_total_limit=3
      - output_dir on persistent disk (caller passes the right path).
    """
    if TrainingArguments is None:
        raise RuntimeError("transformers is required for TrainingArguments")
    # WandB env wiring: TrainingArguments respects WANDB_* env vars when
    # report_to includes "wandb"; we set them here so callers don't have to.
    if report_to and "wandb" in report_to:
        if wandb_project:
            os.environ.setdefault("WANDB_PROJECT", wandb_project)
        if wandb_run_name:
            os.environ.setdefault("WANDB_NAME", wandb_run_name)
        # Disable wandb's symlink-on-checkpoint (broken on some FUSE mounts).
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
    # NB: bf16/fp16 in TrainingArguments enable torch.autocast wrapping the
    # forward+backward. Our model is ALREADY natively bf16 (loaded via
    # torch_dtype=bf16); enabling autocast on top can cause subtle dtype
    # mismatches inside DS-Coder's MLA (e.g. `query_states[:, :, :, k:] = q_pe`
    # raising "destination Float vs source BFloat16" because autocast white-
    # listed an upstream op to fp32 while the destination was pre-allocated
    # in bf16). Keep autocast off; the model's own dtype carries the cast.
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,  # base is frozen; checkpointing buys little
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=max(1, save_steps // 10),
        report_to=report_to or [],  # default no logging; pass ["wandb"] to enable
        run_name=wandb_run_name,
        dataloader_num_workers=0,  # streaming dataset prefers main-thread loop
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )


__all__ = [
    "MythicRDTTrainer",
    "build_training_args",
    "count_trainable",
    "TRAINABLE_STATE_FN",
    "CURRICULUM_STATE_FN",
]
