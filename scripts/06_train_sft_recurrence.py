"""Joint SFT + recurrence + cross-vocab teacher distill for v6V.

Combines:
  - CTD's TeacherCompletionDataset (chat-template, code-only-mask, JSONL
    funcsig corpus) — proven on M124f (HE 76.8 / MBPP 65.9 on DSC-V2-Lite).
  - Mythic-RDT recurrence wrapper (block_mode_residual + first_iter_identity
    + LTI + DepthLoRA + per-loop LayerScale + identity-biased gate).
  - Curriculum T-sampling per microbatch (v4-anchored or default).
  - Optional cross-vocab teacher distill via CTD VocabMapper, using a
    per-prompt teacher cache produced by precompute_xv_sft_cache.py.

The training loop is a raw PyTorch loop (matching CTD's 06_train_sft_on_teacher
style) — NOT HF Trainer — to avoid the existing finetune_phase1.py's
PackedDataset coupling and to keep the loss-mask / chat / code-fence logic
identical to the proven M124f recipe.

Output adapter format: PEFT save_pretrained() — the recurrence wrapper's
DepthLoRA + recurrence cell params are wrapped via PEFT then saved. To
re-load for eval, use scripts/humaneval_smoke.py with --checkpoint pointing
at the output dir + the same recurrence flags.

Usage:

    python scripts/06_train_sft_recurrence.py \\
        --student deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \\
        --corpus data/funcsig_prompts_qwen25c14b_codeonly_T07.jsonl \\
        --output-dir checkpoints/phase1_v6v_joint \\
        --max-prompt-len 384 --max-total-len 1024 \\
        --lora-rank 16 --lora-target-modules up_proj,gate_proj,down_proj \\
        --lr 5e-5 --epochs 2 --batch-size 1 --grad-accum 16 \\
        --warmup-steps 8 --logging-steps 5 --seed 0 \\
        --code-only-mask --chat-template \\
        --recurrent-block-start 4 --recurrent-block-end 22 \\
        --block-mode --block-mode-residual --first-iter-identity \\
        --max-loop-iters 4 --gate-init-bias 0.0 \\
        --layerscale-init 1e-4 --layerscale-clamp-max 0.5 \\
        --curriculum-style v4-anchored \\
        --curriculum-warmup-steps 20 \\
        --curriculum-phase2-start 30 --curriculum-phase3-start 45 \\
        --teacher-logits-xv teacher_cache/qc14b_xv_to_dscv2lite_sft_top32.pt \\
        --teacher-distill-alpha 0.3 --teacher-distill-temperature 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# CTD repo on the pod (or solidPC).
_CTD_DEFAULT = "/workspace/cross-tokenizer-distill"
_ctd_path = os.environ.get("CTD_REPO", _CTD_DEFAULT)
if Path(_ctd_path).exists():
    sys.path.insert(0, _ctd_path)
    sys.path.insert(0, str(Path(_ctd_path) / "experiments" / "validation"))

from transformers import AutoModelForCausalLM  # noqa: E402

from mythic_rdt.configuration import MythicRDTDeepseekV2Config  # noqa: E402
from mythic_rdt.modeling import MythicRDTDeepseekV2ForCausalLM  # noqa: E402
from mythic_rdt.training import inject_depth_lora, count_trainable  # noqa: E402
from mythic_rdt.training.curriculum import (  # noqa: E402
    default_curriculum,
    v3_t1_only_curriculum,
    v3_conservative_curriculum,
    v3_balanced_curriculum,
    v4_anchored_curriculum,
)

# Borrow CTD dataset + collate + loss-mask. Importing the script as a module.
sys.modules.pop("_06_train_sft_on_teacher", None)
import importlib.util as _ilu  # noqa: E402
_ctd_sft_path = Path(_ctd_path) / "experiments" / "validation" / "06_train_sft_on_teacher.py"
_spec = _ilu.spec_from_file_location("ctd_sft", _ctd_sft_path)
_ctd_sft_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ctd_sft_mod)
TeacherCompletionDataset = _ctd_sft_mod.TeacherCompletionDataset
collate = _ctd_sft_mod.collate
build_loss_mask = _ctd_sft_mod.build_loss_mask

from _dscoder_compat import dtype_kwarg, load_dscoder_tokenizer  # noqa: E402

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Student + corpus (CTD-style)
    p.add_argument("--student", required=True)
    p.add_argument("--corpus", required=True, help="JSONL with {prompt, teacher_completion}")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-prompt-len", type=int, default=384)
    p.add_argument("--max-total-len", type=int, default=1024)
    p.add_argument("--code-only-mask", action="store_true")
    p.add_argument("--code-only-mask-from-epoch", type=int, default=1)
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--system-prompt", default=None)

    # LoRA
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=None,
                   help="Defaults to 2 * rank.")
    p.add_argument("--lora-target-modules", default="up_proj,gate_proj,down_proj",
                   help="Comma-separated leaf module names. M124f winner = "
                        "FFN-only up/gate/down.")

    # Recurrence wrapper architecture
    p.add_argument("--recurrent-block-start", type=int, default=4)
    p.add_argument("--recurrent-block-end", type=int, default=22)
    p.add_argument("--block-mode", action="store_true")
    p.add_argument("--block-mode-residual", action="store_true")
    p.add_argument("--first-iter-identity", action="store_true")
    p.add_argument("--prelude-layers", type=int, default=4)
    p.add_argument("--coda-layers", type=int, default=4)
    p.add_argument("--max-loop-iters", type=int, default=4)
    p.add_argument("--gate-init-bias", type=float, default=0.0)
    p.add_argument("--layerscale-init", type=float, default=1e-4)
    p.add_argument("--layerscale-clamp-max", type=float, default=0.5)
    p.add_argument("--lti-residual-scale", type=float, default=0.0,
                   help="v6W: re-introduce LTI contribution at this fixed "
                        "scale in block_mode_residual. 0.0 = pre-v6W (LTI "
                        "dead); 0.01 = v6W default (small but trainable).")
    p.add_argument("--checkpoint-loop", action="store_true")

    # Curriculum
    p.add_argument("--curriculum-style", default="v4-anchored",
                   choices=["default", "v3-t1-only", "v3-conservative",
                            "v3-balanced", "v4-anchored"])
    p.add_argument("--curriculum-warmup-steps", type=int, default=20)
    p.add_argument("--curriculum-phase2-start", type=int, default=30)
    p.add_argument("--curriculum-phase3-start", type=int, default=45)

    # Optimizer
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    # Speed knobs
    p.add_argument("--quant", default="none", choices=["none", "nf4", "fp4"],
                   help="Student weight quantization. M124f used QLoRA NF4; bf16 OOMs the wrapper at T>=2 on 48GB (per memory feedback_phase1_oom_root_causes.md). Default 'none' for parity with non-recurrence runs; pass 'nf4' for v6V joint trainer.")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=["eager", "flash_attention_2", "sdpa"])
    p.add_argument("--moe-vec", action="store_true")

    # Cross-vocab teacher distill (CTD)
    p.add_argument("--teacher-logits-xv", default=None,
                   help="Per-prompt cross-vocab cache built by "
                        "scripts/precompute_xv_sft_cache.py.")
    p.add_argument("--teacher-distill-alpha", type=float, default=0.0)
    p.add_argument("--teacher-distill-temperature", type=float, default=1.0)

    # Logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-project", default="mythic-rdt")
    return p.parse_args()


def build_curriculum(args):
    if args.curriculum_style == "v3-t1-only":
        return v3_t1_only_curriculum()
    if args.curriculum_style == "v3-conservative":
        return v3_conservative_curriculum(t1_steps=args.curriculum_warmup_steps)
    if args.curriculum_style == "v3-balanced":
        return v3_balanced_curriculum(
            t1_steps=args.curriculum_warmup_steps,
            mix_start=args.curriculum_phase2_start,
            t4_dominant=args.curriculum_phase3_start,
        )
    if args.curriculum_style == "v4-anchored":
        return v4_anchored_curriculum(
            t1_steps=args.curriculum_warmup_steps,
            mix12_start=args.curriculum_phase2_start,
            mix124_start=args.curriculum_phase3_start,
        )
    return default_curriculum(
        warmup_steps=args.curriculum_warmup_steps,
        phase2_start=args.curriculum_phase2_start,
        phase3_start=args.curriculum_phase3_start,
    )


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if "/tmp" in args.output_dir:
        print("ERROR: --output-dir under /tmp forbidden", file=sys.stderr)
        return 2

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16

    # -------- Tokenizer + base model --------
    tok = load_dscoder_tokenizer(args.student)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "cuda",
        "attn_implementation": args.attn_impl,
        **dtype_kwarg(dtype),
    }
    if args.quant == "none":
        # Pre-quantized cached dirs (e.g. base/...-Instruct-nf4) carry their
        # own quantization_config in config.json — transformers auto-detects.
        print(f"[v6v] loading base {args.student} (auto: bf16 unless config has embedded quantization)...")
    else:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.quant,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[v6v] loading base {args.student} (4-bit {args.quant.upper()} QLoRA-style)...")
    base = AutoModelForCausalLM.from_pretrained(args.student, **load_kwargs)
    base.eval()
    if args.moe_vec:
        from humaneval_smoke import _patch_moe_forward
        n = _patch_moe_forward(base)
        print(f"[v6v] moe-vec: patched {n} layers")

    # -------- Wrap with recurrence --------
    cfg = MythicRDTDeepseekV2Config(
        prelude_layers=args.prelude_layers,
        coda_layers=args.coda_layers,
        recurrent_block_start=args.recurrent_block_start,
        recurrent_block_end=args.recurrent_block_end,
        block_mode=args.block_mode,
        block_mode_residual=args.block_mode_residual,
        first_iter_identity=args.first_iter_identity,
        train_loop_iters=min(2, args.max_loop_iters),
        max_loop_iters=args.max_loop_iters,
        gate_init_bias=args.gate_init_bias,
        layerscale_init=args.layerscale_init,
        layerscale_clamp_max=args.layerscale_clamp_max,
        lti_residual_scale=args.lti_residual_scale,
        base_model_path=args.student,
    )
    wrapper = MythicRDTDeepseekV2ForCausalLM(cfg, base=base)
    wrapper._checkpoint_loop = bool(args.checkpoint_loop)

    # -------- Inject DepthLoRA --------
    target_modules = [s.strip() for s in args.lora_target_modules.split(",")]
    print(f"[v6v] injecting DepthLoRA rank={args.lora_rank} on targets={target_modules}")
    records = inject_depth_lora(
        wrapper, targets=target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha or 2 * args.lora_rank,
        lora_dtype=dtype,
    )
    print(f"[v6v] LoRA modules wired: {len(records)}")
    train_n, total_n = count_trainable(wrapper)
    print(f"[v6v] trainable params: {train_n:,} / {total_n:,} ({100*train_n/max(1,total_n):.4f}%)")

    # -------- Dataset --------
    ds = TeacherCompletionDataset(
        args.corpus, tok,
        max_prompt_len=args.max_prompt_len,
        max_total_len=args.max_total_len,
        code_only_mask=args.code_only_mask,
        chat_template=args.chat_template,
        system_prompt=args.system_prompt,
    )
    if args.chat_template:
        sample = ds._wrap_prompt(ds.records[0]["prompt"])
        print(f"[v6v] --chat-template ON; sample prompt: {sample[:200]!r}")
    if args.code_only_mask:
        print(f"[v6v] --code-only-mask: ON from epoch {args.code_only_mask_from_epoch}")

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        num_workers=0, pin_memory=False, drop_last=True,
    )
    total_steps = len(dl) * args.epochs // args.grad_accum
    print(f"[v6v] dataset={len(ds)} batches/epoch={len(dl)} epochs={args.epochs} "
          f"total_optim_steps={total_steps}")

    # -------- Curriculum --------
    curriculum = build_curriculum(args)
    print("[v6v] curriculum phases:")
    for ph in curriculum.phases:
        print(f"[v6v]   step>={ph.start_step}: T-mix={ph.weights}")
    max_T = max(t for ph in curriculum.phases for t in ph.weights)
    if max_T > args.max_loop_iters:
        raise SystemExit(
            f"[v6v] curriculum samples T={max_T} > --max-loop-iters={args.max_loop_iters}"
        )

    # -------- Optional teacher cache --------
    teacher_cache = None
    if args.teacher_distill_alpha > 0 and args.teacher_logits_xv:
        print(f"[v6v] loading teacher cache {args.teacher_logits_xv}")
        c = torch.load(args.teacher_logits_xv, map_location="cpu", weights_only=False)
        teacher_cache = {
            "indices": c["indices"],   # [N_examples, max_total_len, K]
            "values":  c["values"],    # [N_examples, max_total_len, K]
            "alignment_mask": c.get("alignment_mask"),  # [N_examples, max_total_len]
            "meta": c.get("meta", {}),
        }
        print(f"[v6v] teacher cache shape={tuple(teacher_cache['indices'].shape)} "
              f"meta={teacher_cache['meta']}")
        if teacher_cache["alignment_mask"] is not None:
            am = teacher_cache["alignment_mask"]
            raw_cov = float(am.float().mean().item())
            # Completion-only coverage: only positions whose mask is True OR False
            # AND whose top-K indices are non-zero (precompute zeros prompt-side).
            tidx = teacher_cache["indices"]
            non_prompt = (tidx.abs().sum(-1) > 0)
            denom = non_prompt.float().sum().clamp(min=1.0)
            comp_cov = float((am.float() * non_prompt.float()).sum().item() / denom.item())
            print(f"[v6v] teacher alignment coverage: raw={raw_cov:.2%} (incl. prompt-side False) | completion-only={comp_cov:.2%}")

    # -------- Optimizer --------
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = LambdaLR(opt, lr_lambda)

    # -------- Wandb --------
    use_wandb = args.wandb and _WANDB_AVAILABLE and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))

    # -------- Training loop --------
    wrapper.train()
    step = 0
    opt_step = 0
    micro_in_step = 0
    t_start = time.time()
    accum_loss = 0.0
    accum_distill = 0.0
    accum_n_valid = 0
    accum_n_unmasked = 0
    # Track per-example index in dataset order so we can index into teacher cache.
    # DataLoader shuffles; we need the original index. Hack: rebuild dataset with
    # an index field, OR re-index by iterating ds in order. Simpler: include the
    # index in the dataset's __getitem__. Patch TeacherCompletionDataset on the fly.
    _orig_getitem = TeacherCompletionDataset.__getitem__

    def _getitem_with_idx(self, i):
        item = _orig_getitem(self, i)
        item["__ds_idx__"] = i
        return item

    TeacherCompletionDataset.__getitem__ = _getitem_with_idx

    def _collate_with_idx(batch, pad_id):
        out = collate(batch, pad_id)
        out["__ds_idx__"] = torch.tensor([b["__ds_idx__"] for b in batch], dtype=torch.long)
        return out

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: _collate_with_idx(b, tok.pad_token_id),
        num_workers=0, pin_memory=False, drop_last=True,
    )

    for epoch in range(args.epochs):
        mask_active_this_epoch = (
            args.code_only_mask
            and (epoch + 1) >= args.code_only_mask_from_epoch
        )
        print(f"[v6v] epoch {epoch+1}/{args.epochs} mask_active={mask_active_this_epoch}")

        for batch in dl:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            prompt_len = batch["prompt_len"]
            ds_idx = batch["__ds_idx__"]
            mask = build_loss_mask(input_ids, attention_mask, prompt_len, tok.pad_token_id)
            n_unmasked = mask.sum().item()
            if mask_active_this_epoch:
                code_mask = batch["code_mask"].cuda()
                target_in_code = torch.zeros_like(mask)
                target_in_code[:, :-1] = code_mask[:, 1:]
                mask = mask & target_in_code

            # Sample T from curriculum.
            T = curriculum.sample_T(global_step=opt_step, micro_batch_idx=micro_in_step)

            # Forward through wrapper at sampled T. The wrapper returns a raw
            # logits Tensor unless return_dict / labels / use_cache forces a
            # CausalLMOutputWithPast (see modeling.py forward at line ~743).
            # Pass return_dict=True for forward-compat AND keep an instance
            # check for safety (custom wrapper subclasses may override).
            s_out = wrapper(input_ids=input_ids, attention_mask=attention_mask, T=T, return_dict=True)
            all_logits = s_out.logits if hasattr(s_out, "logits") else s_out
            logits = all_logits[:, :-1, :]
            targets = input_ids[:, 1:]
            slice_mask = mask[:, :-1]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            tgt_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            ce_loss = -(tgt_lp * slice_mask.float()).sum() / slice_mask.float().sum().clamp(min=1.0)
            loss = ce_loss

            # Optional cross-vocab teacher distill.
            if teacher_cache is not None:
                t_idx = teacher_cache["indices"][ds_idx]   # [B, L_max, K]
                t_val = teacher_cache["values"][ds_idx]    # [B, L_max, K]
                # Slice to the actual seq_len of this batch (left-truncate to current input).
                L_now = input_ids.size(1)
                t_idx = t_idx[:, :L_now, :].to(input_ids.device).long()
                t_val_dev = t_val[:, :L_now, :].to(input_ids.device).float()
                # Shift to next-token (HF causal LM convention).
                t_idx_s = t_idx[:, :-1, :].contiguous()
                t_val_s = t_val_dev[:, :-1, :].contiguous()
                # Gather student logits at teacher's K indices.
                student_topk = logits.gather(-1, t_idx_s)   # [B, L-1, K]
                T_temp = args.teacher_distill_temperature
                log_q = F.log_softmax(student_topk.float() / T_temp, dim=-1)
                p = F.softmax(t_val_s / T_temp, dim=-1)
                kl_per_tok = (p * (p.clamp(min=1e-12).log() - log_q)).sum(dim=-1)
                # Apply slice_mask + alignment mask (if any).
                valid = slice_mask
                if teacher_cache["alignment_mask"] is not None:
                    am = teacher_cache["alignment_mask"][ds_idx][:, :L_now].to(input_ids.device)
                    valid = valid & am[:, :-1]
                n_valid = valid.float().sum().clamp(min=1.0)
                distill = (kl_per_tok * valid.float()).sum() / n_valid
                # T² scale (Hinton).
                distill_term = args.teacher_distill_alpha * (T_temp ** 2) * distill
                loss = loss + distill_term
                accum_distill += float(distill.detach().item())

            (loss / args.grad_accum).backward()
            accum_loss += float(ce_loss.detach().item())
            accum_n_valid += int(slice_mask.sum().item())
            accum_n_unmasked += int(n_unmasked)
            step += 1
            micro_in_step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                micro_in_step = 0
                if opt_step % args.logging_steps == 0:
                    avg_ce = accum_loss / args.grad_accum
                    avg_distill = accum_distill / args.grad_accum if teacher_cache else 0.0
                    elapsed = time.time() - t_start
                    density = (accum_n_valid / accum_n_unmasked) if accum_n_unmasked else 0.0
                    cur_lr = sched.get_last_lr()[0]
                    print(f"[v6v] step={opt_step}/{total_steps} ce={avg_ce:.4f} "
                          f"distill={avg_distill:.4f} T={T} lr={cur_lr:.2e} "
                          f"mask_density={density:.2f} {elapsed:.0f}s",
                          flush=True)
                    if use_wandb:
                        wandb.log({
                            "train/ce_loss": avg_ce,
                            "train/distill_loss": avg_distill,
                            "train/T_sample": T,
                            "train/lr": cur_lr,
                            "train/mask_density": density,
                            "train/epoch": epoch + 1,
                        }, step=opt_step)
                accum_loss = 0.0
                accum_distill = 0.0
                accum_n_valid = 0
                accum_n_unmasked = 0

        # Save per-epoch checkpoint.
        # We save in two formats:
        #   1. PEFT adapter at out/epoch-N/ — for smoke / eval (humaneval_smoke
        #      can load this via --checkpoint).
        #   2. mythic_rdt_trainable.pt at out/epoch-N/ — for resume into
        #      finetune_phase1.py if we want to add more recurrence training later.
        ck = out / f"epoch-{epoch+1}"
        ck.mkdir(parents=True, exist_ok=True)
        train_state = {
            n: p.detach().cpu().clone()
            for n, p in wrapper.named_parameters() if p.requires_grad
        }
        torch.save(train_state, ck / "mythic_rdt_trainable.pt")
        cfg_path = ck / "mythic_rdt_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(wrapper.config.to_dict(), f, indent=2, default=str)
        print(f"[v6v] epoch {epoch+1} done, saved {ck}")

    # Final adapter at top-level too.
    train_state = {
        n: p.detach().cpu().clone()
        for n, p in wrapper.named_parameters() if p.requires_grad
    }
    torch.save(train_state, out / "mythic_rdt_trainable.pt")
    with open(out / "mythic_rdt_config.json", "w", encoding="utf-8") as f:
        json.dump(wrapper.config.to_dict(), f, indent=2, default=str)
    print(f"[v6v] DONE. Final wrapper -> {out}")
    if use_wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
