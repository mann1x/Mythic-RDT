#!/usr/bin/env python
"""Mythic-RDT Stage 1 Phase 1 fine-tune entry point.

Per MASTER_PLAN.md §5 phase 1:
    train recurrence cell + depth-LoRA on the recurrent layer (T-curriculum
    2->4->8) against FineWeb-Edu prose + Stack-smol Python code, packed
    seq=2048 at first then scaled up.

Per `feedback_finetune_resumable.md`:
    - SIGINT/SIGTERM safe (Trainer handles it).
    - Checkpoint = model state + optimizer + scheduler + RNG + step + curriculum
      + sampler position (Trainer + our _save override).
    - --resume <output_dir> picks up the latest checkpoint and continues.
    - Save to persistent disk, NOT /tmp.

Usage on pod (after `bash scripts/setup_pod_env.sh`):

    source /workspace/venv-tf4/bin/activate
    cd /workspace/mythic-rdt

    # Fresh run, bf16 frozen base, no quant, with wandb:
    WANDB_API_KEY=<key> python scripts/finetune_phase1.py \\
        --base base/DeepSeek-Coder-V2-Lite-Instruct \\
        --output-dir checkpoints/phase1_v1 \\
        --seq-len 2048 --per-device-batch 2 --grad-accum 4 \\
        --max-steps 5000 --save-steps 200 \\
        --wandb-project mythic-rdt --wandb-run-name phase1-v1

    # Same with QLoRA (4-bit base, frees ~22 GB VRAM):
    ... --quant nf4

    # Resume from a checkpoint:
    ... --resume checkpoints/phase1_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

# torch 2.6 made `weights_only=True` the default in torch.load. HF Trainer's
# rng_state.pth contains numpy state (np.random.get_state()) which serializes
# numpy arrays + dtype objects via numpy._core.multiarray._reconstruct. Those
# are not in torch's default safe-globals allowlist, so resume crashes with
# `_pickle.UnpicklingError: WeightsUnpickler error: Unsupported global ...`.
# Allow-list the small, well-known numpy reconstructors that Trainer needs.
try:
    import numpy as _np
    torch.serialization.add_safe_globals([
        _np.dtype,
        _np.ndarray,
        _np._core.multiarray._reconstruct,
        _np._core.multiarray.scalar,
    ])
    # Numpy dtype subclasses (uint32 etc.) used by RandomState bit-generator state.
    for _dt_name in ("UInt32DType", "Int64DType", "Float64DType", "BoolDType"):
        _dt = getattr(_np.dtypes, _dt_name, None)
        if _dt is not None:
            torch.serialization.add_safe_globals([_dt])
except Exception as _exc:  # numpy missing is fatal elsewhere; warn here.
    print(f"[ft] WARN: could not register numpy safe-globals for torch.load: {_exc}")

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from mythic_rdt.configuration import MythicRDTDeepseekV2Config  # noqa: E402
from mythic_rdt.modeling import MythicRDTDeepseekV2ForCausalLM  # noqa: E402
from mythic_rdt.training import (  # noqa: E402
    MythicRDTTrainer,
    build_packed_dataset,
    build_training_args,
    count_trainable,
    inject_depth_lora,
)
from mythic_rdt.training.trainer import (  # noqa: E402
    TRAINABLE_STATE_FN,
    _load_trainable_state,
)
from mythic_rdt.training.curriculum import (  # noqa: E402
    default_curriculum,
    v3_balanced_curriculum,
    v3_conservative_curriculum,
    v3_t1_only_curriculum,
    v4_anchored_curriculum,
)
from _dscoder_compat import dtype_kwarg, load_dscoder_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mythic-RDT Stage 1 Phase 1 fine-tune")
    p.add_argument("--base", type=str, default="base/DeepSeek-Coder-V2-Lite-Instruct")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Persistent disk path. NEVER /tmp (memory rule).")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from this checkpoint dir. Trainer auto-detects "
                        "the latest sub-checkpoint inside.")
    p.add_argument("--init-from-checkpoint", type=str, default=None,
                   help="Pre-load a previous run's TRAINABLE state (LoRA + gate "
                        "+ LayerScale + LTI) into the wrapper, then start "
                        "FRESH training (new optimizer, new scheduler, new "
                        "global_step). Use this when you want to start a new "
                        "run from a previous wrapper checkpoint AND the "
                        "architecture has changed (e.g., bumped --max-loop-iters "
                        "from 1 to 4, which expands LoRA T-slices). T-axis "
                        "expansion is handled automatically — leading slices "
                        "get the checkpoint values, new slices keep init.")
    # Architecture (must match Phase 0 unless we deliberately change it).
    p.add_argument("--recurrent-layer-idx", type=int, default=10,
                   help="Single-layer mode: which base layer to iterate. "
                        "Ignored when --recurrent-block-start/end are set.")
    p.add_argument("--recurrent-block-start", type=int, default=None,
                   help="Block-mode v3+: start index of the consecutive "
                        "recurrent block (inclusive).")
    p.add_argument("--recurrent-block-end", type=int, default=None,
                   help="Block-mode v3+: end index of the consecutive "
                        "recurrent block (inclusive).")
    p.add_argument("--block-mode", action="store_true",
                   help="Use the v3 recurrence formula h_next = block_out + "
                        "ls*gate*inj (block_out passes through). Required "
                        "when running multi-layer blocks; otherwise the loop "
                        "discards block_out at gate≈0 init.")
    p.add_argument("--block-mode-residual", action="store_true",
                   help="Fix A (v6F+): use h-residual formula "
                        "h_next = h + mix*(block_out - h) where mix=(ls*gate).clamp(0,1). "
                        "Bounds ||h|| growth across iterations so coda receives "
                        "input in its training distribution. Requires --block-mode. "
                        "See memory/project_recurrence_root_cause_block_mode.md.")
    p.add_argument("--first-iter-identity", action="store_true",
                   help="v6A architectural fix: t=0 iteration of the recurrence "
                        "loop is unconditionally identity (h_next = block_out). "
                        "At T=1 the wrapper output is byte-for-byte equal to "
                        "base. T>=1 iterations inject normally. See "
                        "memory/project_phase1_v6_diagnosis.md.")
    p.add_argument("--prelude-layers", type=int, default=1)
    p.add_argument("--coda-layers", type=int, default=1)
    p.add_argument("--max-loop-iters", type=int, default=8,
                   help="Upper bound on T; LoRA holds this many adapter slices.")
    p.add_argument("--gate-init-bias", type=float, default=0.0,
                   help="Init bias for IdentityBiasedGate. v3 default 0.0 "
                        "(sigmoid=0.5). v0-v2 used -3.0 which saturated "
                        "the gradient and pinned gate.bias dead.")
    p.add_argument("--layerscale-init", type=float, default=1e-4)
    p.add_argument("--layerscale-clamp-max", type=float, default=None,
                   help="Optional upper clamp on PerLoopLayerScale. v3 "
                        "recommendation: 1e-2 to bound recurrent "
                        "perturbation per iteration.")
    p.add_argument("--lti-residual-scale", type=float, default=0.0,
                   help="v6W (council finding): re-introduce LTI contribution "
                        "at this fixed scale in block_mode_residual. 0.0 = "
                        "pre-v6W (LTI A_diag/B_proj are dead weight). 0.01 "
                        "= v6W default (small enough to stay bounded over "
                        "T iterations, large enough for productive gradient).")
    # LoRA
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", type=str, nargs="+",
                   default=["self_attn.q_proj_or_q_a", "self_attn.o_proj"])
    # Curriculum
    p.add_argument("--curriculum-style", type=str, default="default",
                   choices=["default", "v3-t1-only", "v3-conservative",
                            "v3-balanced", "v4-anchored"],
                   help="Which curriculum function to build. v3-t1-only = "
                        "T=1 every step (foundation isolation test). "
                        "v3-conservative = T=1 warmup + T=2. "
                        "v3-balanced = T=1/2/4 mixed. "
                        "v4-anchored = T=1 anchor -> mix {1,2} -> mix {1,2,4} -> "
                        "T={2,4}-heavy, sized for ~400 steps; pair with KL anchor.")
    p.add_argument("--curriculum-warmup-steps", type=int, default=200)
    p.add_argument("--curriculum-phase2-start", type=int, default=1000)
    p.add_argument("--curriculum-phase3-start", type=int, default=3000)
    # KL-to-base anchor (v4+): pulls wrapper logits toward base on the same
    # batch every N steps. Cheap insurance against drift compounding with T.
    p.add_argument("--kl-anchor-alpha", type=float, default=0.0,
                   help="Weight on KL(base || wrapper) added to loss. "
                        "0.0 disables. v4-anchored default: 0.05.")
    p.add_argument("--kl-anchor-every", type=int, default=0,
                   help="Apply KL anchor every N optimizer steps. "
                        "0 disables. v4-anchored default: 8 (~6%% extra cost).")
    # v5: dual-T training with margin + distill — fixes T-specialization gap.
    # Probe 2026-04-28 confirmed v4-anchored ckpt-400 has T=4 CE *higher* than
    # T=1 (0.3-0.8 nats worse), because nothing in the loss penalized that.
    p.add_argument("--margin-alpha", type=float, default=0.0,
                   help="v5: weight on ReLU(CE_hi - CE_lo + margin_nats). "
                        "Forces high-T to outperform low-T by `margin_nats`. "
                        "0 disables (v4 single-T mode). Suggested: 0.1.")
    p.add_argument("--margin-nats", type=float, default=0.02,
                   help="v5: required CE improvement gap (nats) from low-T to "
                        "high-T. Default 0.02 (small but enforceable).")
    p.add_argument("--distill-alpha", type=float, default=0.0,
                   help="v5: weight on per-token KL(low-T || high-T.detach()). "
                        "Forces low-T to copy high-T's output distribution. "
                        "0 disables. Suggested: 0.05.")
    p.add_argument("--dual-t-lo", type=int, default=1,
                   help="v5: low T value used in dual-T forward. Default 1.")
    p.add_argument("--dual-t-hi", type=int, default=0,
                   help="v5: high T value (0 = config.max_loop_iters). "
                        "Activates when margin-alpha or distill-alpha > 0; "
                        "doubles forward cost per microbatch.")
    # v6J: focal-weighted CE on the T=4 (high-T) path.
    p.add_argument("--teacher-distill-alpha", type=float, default=0.0,
                   help="If >0, add per-token sparse-top-K KL between wrapper "
                        "T_hi and a precomputed BF16 teacher logits cache. "
                        "Requires --teacher-logits.")
    p.add_argument("--teacher-logits", type=str, default=None,
                   help="Path to .pt produced by precompute_bf16_teacher_logits.py "
                        "(must match --data-seed and --seq-len).")
    p.add_argument("--teacher-distill-temperature", type=float, default=1.0,
                   help="Hinton temperature for teacher KL (loss scaled by T^2).")
    p.add_argument("--teacher-refinement-mask", action="store_true",
                   help="v6R+: only apply teacher distill on tokens where the "
                        "wrapper's T_lo argmax DISAGREES with the teacher's "
                        "top-1. Focuses the distill gradient on tokens we "
                        "already know T=1 gets wrong. Lets recurrence refine "
                        "the disagreement-set without anchoring agreement-"
                        "majority tokens to the teacher distribution.")

    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="v6J: focal weight exponent on T=4 CE per token. "
                        "weight_per_token = (1 - p_T1_correct)^gamma. "
                        "0 = standard CE (default). 1.0 = mild focal. "
                        "Concentrates T=4 gradient on tokens where T=1 is uncertain. "
                        "T=1 (low-T) CE stays unweighted so it trains uniformly.")
    # Trainer / data
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Start at 2k; scale up to 4k/8k as VRAM allows.")
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--data-seed", type=int, default=0)
    # Quantization
    p.add_argument("--quant", type=str, default="none", choices=["none", "nf4", "fp4"],
                   help="Quantize the FROZEN base to NF4/FP4 via bitsandbytes "
                        "(QLoRA). Frees ~22 GB VRAM at the cost of ~10-30%% "
                        "throughput. Phase 0 bit-exactness no longer holds.")
    # Speed knobs (A/B isolation — both default to OFF for parity with prior runs)
    p.add_argument("--attn-impl", type=str, default="eager",
                   choices=["eager", "flash_attention_2", "sdpa"],
                   help="Attention implementation forwarded to base "
                        "from_pretrained. eager = legacy default. "
                        "flash_attention_2 needs flash-attn installed "
                        "(GLIBC>=2.32 or polyfill on local).")
    p.add_argument("--moe-vec", action="store_true",
                   help="Monkey-patch DeepseekV2MoE.forward with sorted/"
                        "grouped dispatch (no per-expert Python mask loop). "
                        "Numerically equivalent in fp32; bf16 may diverge "
                        "by ~1e-3.")
    # Memory: gradient-checkpoint the recurrent loop body
    p.add_argument("--checkpoint-loop", action="store_true",
                   help="Gradient-checkpoint the recurrent loop body. Holds "
                        "only one loop step's activations at a time during "
                        "backward (vs O(T) without). Recommended at T>=4 or "
                        "any time activation memory is tight. ~30%% wall-time "
                        "overhead. v1 OOM'd at T=2 with this off; left off "
                        "by default so callers opt in explicitly.")
    p.add_argument("--ac-cpu-offload", action="store_true",
                   help="Stage checkpointed activations on pinned CPU memory "
                        "during forward and prefetch back during backward "
                        "(Unsloth+NVIDIA 2026 'double-buffered AC' borrow, "
                        "https://unsloth.ai/blog/nvidia-collab). Requires "
                        "--checkpoint-loop. PyTorch >=2.6 pipelines the H2D "
                        "transfer with compute via internal streams. Trades "
                        "~0.2-0.5 GB extra GPU buffer for full activation "
                        "offload (saves 2-5 GB at T=4 sl=1024 on a 19-layer "
                        "block). Free if PCIe bandwidth isn't your bottleneck; "
                        "expect 0-5%% wall-time overhead.")
    # Logging
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true",
                   help="Force-disable wandb even if WANDB_API_KEY is set.")
    return p.parse_args()


def _load_base_model(base_path: str, dtype: torch.dtype, quant: str,
                     attn_impl: str = "eager"):
    """Load DS-Coder, optionally with bitsandbytes 4-bit quantization."""
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "cuda",
        "attn_implementation": attn_impl,
        **dtype_kwarg(dtype),
    }
    if quant != "none":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "QLoRA requested (--quant) but transformers BitsAndBytesConfig "
                "is unavailable. Install bitsandbytes."
            ) from exc
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant,                 # "nf4" or "fp4"
            bnb_4bit_compute_dtype=dtype,              # bf16
            bnb_4bit_use_double_quant=True,            # extra ~0.4 bit savings
        )
        load_kwargs["quantization_config"] = bnb
        # device_map can stay "cuda" -- bnb is GPU-resident.
        print(f"[ft] loading base with {quant.upper()} quantization (compute={dtype})")
    else:
        print(f"[ft] loading base in {dtype} (no quantization)")
    return AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)


def main() -> int:
    args = parse_args()
    dtype = torch.bfloat16

    # WandB enable logic: opt-in via project name + key in env, opt-out via flag.
    use_wandb = (
        not args.no_wandb
        and (args.wandb_project is not None)
        and bool(os.environ.get("WANDB_API_KEY"))
    )
    report_to = ["wandb"] if use_wandb else []
    if args.wandb_project and not use_wandb:
        if args.no_wandb:
            print("[ft] --no-wandb set; wandb disabled")
        elif not os.environ.get("WANDB_API_KEY"):
            print("[ft] WANDB_API_KEY not in env; wandb disabled "
                  "(set it on solidPC and forward via SSH SendEnv)")

    # Fail-fast guards.
    if not Path(args.base).exists():
        print(f"ERROR: base path missing: {args.base}", file=sys.stderr)
        return 2
    if "/tmp" in args.output_dir:
        print("ERROR: --output-dir under /tmp is forbidden (memory rule). "
              "Use a persistent disk path.", file=sys.stderr)
        return 2

    print(f"[ft] base={args.base}  output_dir={args.output_dir}  quant={args.quant}")

    base = _load_base_model(args.base, dtype=dtype, quant=args.quant,
                            attn_impl=args.attn_impl)
    base.eval()
    print(f"[ft] attn_implementation={args.attn_impl}")
    if args.moe_vec:
        # Reuse the helper from humaneval_smoke.py — single source of truth.
        sys.path.insert(0, str(Path(__file__).parent))
        from humaneval_smoke import _patch_moe_forward  # noqa: E402
        n_patched = _patch_moe_forward(base)
        print(f"[ft] moe-vec: patched {n_patched} DeepseekV2MoE layers")

    cfg = MythicRDTDeepseekV2Config(
        prelude_layers=args.prelude_layers,
        coda_layers=args.coda_layers,
        recurrent_layer_idx=args.recurrent_layer_idx,
        recurrent_block_start=args.recurrent_block_start,
        recurrent_block_end=args.recurrent_block_end,
        block_mode=args.block_mode,
        block_mode_residual=args.block_mode_residual,
        first_iter_identity=args.first_iter_identity,
        train_loop_iters=min(2, args.max_loop_iters),  # initial T; trainer overrides per-step
        max_loop_iters=args.max_loop_iters,
        gate_init_bias=args.gate_init_bias,
        layerscale_init=args.layerscale_init,
        layerscale_clamp_max=args.layerscale_clamp_max,
        lti_residual_scale=args.lti_residual_scale,
        base_model_path=args.base,
    )
    wrapper = MythicRDTDeepseekV2ForCausalLM(cfg, base=base)
    # The recurrence cell is built inside __init__ in the right dtype.
    # Surface the gradient-checkpoint flag onto the wrapper so its forward()
    # picks it up. Read inside the loop with `getattr(self, "_checkpoint_loop", False)`.
    wrapper._checkpoint_loop = bool(args.checkpoint_loop)
    wrapper._checkpoint_loop_cpu_offload = bool(args.ac_cpu_offload)
    if wrapper._checkpoint_loop:
        print("[ft] gradient-checkpointing recurrent loop body (use_reentrant=False)")
    if wrapper._checkpoint_loop_cpu_offload:
        if not wrapper._checkpoint_loop:
            raise SystemExit(
                "[ft] --ac-cpu-offload requires --checkpoint-loop (the offload "
                "context only takes effect inside torch.utils.checkpoint)."
            )
        print("[ft] AC CPU-offload active: saved tensors stage to pinned host "
              "memory during forward, prefetched in backward (Unsloth+NVIDIA "
              "2026 borrow). Expect +0.2-0.5 GB GPU buffer, -2 to -5 GB peak.")

    # Inject depth-LoRA on chosen Linears of the recurrent layer.
    records = inject_depth_lora(
        wrapper,
        targets=args.lora_targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dtype=dtype,
    )
    for r in records:
        print(f"[ft]   lora wired: {r.qualified_name}  "
              f"in={r.in_features} out={r.out_features} "
              f"rank={r.rank} T={r.n_iters} base_dtype={r.base_dtype}")

    train_n, total_n = count_trainable(wrapper)
    print(f"[ft] trainable params: {train_n:,} / {total_n:,} "
          f"({100*train_n/max(1, total_n):.4f}%)")

    # Optional: pre-load trainable state from a previous run (without resuming
    # its optimizer / scheduler / step counter). T-axis expansion handled.
    if args.init_from_checkpoint:
        init_path = Path(args.init_from_checkpoint)
        # Accept either a checkpoint-N subdir or a parent dir (auto-pick latest).
        if not (init_path / TRAINABLE_STATE_FN).exists():
            sub = sorted(init_path.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
            if not sub:
                print(f"ERROR: --init-from-checkpoint path {init_path} has no "
                      f"{TRAINABLE_STATE_FN} and no checkpoint-* subdir.",
                      file=sys.stderr)
                return 2
            init_path = sub[-1]
        state_path = init_path / TRAINABLE_STATE_FN
        print(f"[ft] init-from-checkpoint: loading trainable state from {state_path}")
        state = torch.load(str(state_path), map_location="cpu", weights_only=True)
        loaded, missing, unexpected = _load_trainable_state(wrapper, state)
        print(f"[ft] init-from-checkpoint: loaded={loaded} "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[ft]   missing (first 5): {missing[:5]}")
        if unexpected:
            print(f"[ft]   unexpected (first 5): {unexpected[:5]}")

    tokenizer = load_dscoder_tokenizer(args.base)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resume handling. `--resume <dir>` points at the OUTPUT directory of a
    # previous run; Trainer's train(resume_from_checkpoint=True) then
    # auto-detects the latest checkpoint-N subdir inside output_dir. We also
    # pin output_dir to that same path so new checkpoints land alongside the
    # old ones (vs writing a fresh tree).
    resume_flag: Optional[bool] = None
    skip_blocks = 0
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"ERROR: --resume path {resume_dir} does not exist", file=sys.stderr)
            return 2
        # Use args.resume as the output dir (continue writing there).
        args.output_dir = str(resume_dir)
        latest = sorted(resume_dir.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[1]))
        if not latest:
            print(f"[ft] WARN: --resume {resume_dir} has no checkpoint-* subdir; "
                  f"starting fresh in this output_dir.")
        else:
            resume_flag = True
            sd = json.loads((latest[-1] / "trainer_state.json").read_text())
            consumed = int(sd.get("global_step", 0)) * args.per_device_batch * args.grad_accum
            skip_blocks = consumed
            print(f"[ft] resume: latest={latest[-1].name}, "
                  f"skip_blocks={skip_blocks} (replay+drop streaming data to align "
                  f"with global_step={sd.get('global_step')})")

    train_ds = build_packed_dataset(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        seed=args.data_seed,
        skip_blocks=skip_blocks,
    )

    if args.curriculum_style == "v3-t1-only":
        curriculum = v3_t1_only_curriculum()
    elif args.curriculum_style == "v3-conservative":
        curriculum = v3_conservative_curriculum(
            t1_steps=args.curriculum_warmup_steps,
        )
    elif args.curriculum_style == "v3-balanced":
        curriculum = v3_balanced_curriculum(
            t1_steps=args.curriculum_warmup_steps,
            mix_start=args.curriculum_phase2_start,
            t4_dominant=args.curriculum_phase3_start,
        )
    elif args.curriculum_style == "v4-anchored":
        curriculum = v4_anchored_curriculum(
            t1_steps=args.curriculum_warmup_steps,
            mix12_start=args.curriculum_phase2_start,
            mix124_start=args.curriculum_phase3_start,
        )
    else:
        curriculum = default_curriculum(
            warmup_steps=args.curriculum_warmup_steps,
            phase2_start=args.curriculum_phase2_start,
            phase3_start=args.curriculum_phase3_start,
        )
    print("[ft] curriculum phases:")
    for ph in curriculum.phases:
        print(f"[ft]   step>={ph.start_step}: T-mix={ph.weights}")
    # Fail fast: curriculum must not sample T > max_loop_iters or the wrapper's
    # _resolve_T raises mid-step. (Hit during resume-test: default curriculum's
    # phase 3 samples T=8 but smoke ran with --max-loop-iters 4.)
    max_T_in_curriculum = max(t for ph in curriculum.phases for t in ph.weights)
    if max_T_in_curriculum > args.max_loop_iters:
        raise ValueError(
            f"Curriculum samples T up to {max_T_in_curriculum} but "
            f"--max-loop-iters={args.max_loop_iters}. Either bump "
            f"--max-loop-iters to >= {max_T_in_curriculum} (LoRA holds that many "
            f"adapter slices, which costs ~lora_rank*hidden bf16 params per slice) "
            f"or pass a curriculum that stays within {args.max_loop_iters}."
        )

    targs = build_training_args(
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        bf16=True,
        report_to=report_to,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    trainer = MythicRDTTrainer(
        model=wrapper,
        args=targs,
        train_dataset=train_ds,
        curriculum=curriculum,
        kl_anchor_alpha=args.kl_anchor_alpha,
        kl_anchor_every=args.kl_anchor_every,
        margin_alpha=args.margin_alpha,
        margin_nats=args.margin_nats,
        distill_alpha=args.distill_alpha,
        dual_t_lo=args.dual_t_lo,
        dual_t_hi=args.dual_t_hi,
        focal_gamma=args.focal_gamma,
        teacher_distill_alpha=args.teacher_distill_alpha,
        teacher_logits_path=args.teacher_logits,
        teacher_distill_temperature=args.teacher_distill_temperature,
        teacher_refinement_mask=args.teacher_refinement_mask,
    )
    print(f"[ft] starting training; resume_flag={resume_flag!r}")
    trainer.train(resume_from_checkpoint=resume_flag)
    print("[ft] training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
