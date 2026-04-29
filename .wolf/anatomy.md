# anatomy.md

> Auto-maintained by OpenWolf. Last scanned: 2026-04-29T09:18:44.571Z
> Files: 92 tracked | Anatomy hits: 0 | Misses: 0

## ../../../../root/.claude/projects/-srv-dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5-backup-models-Mythic-RDT/memory/

- `feedback_finetune_resumable.md` (~611 tok)
- `feedback_init_from_checkpoint_pattern.md` (~896 tok)
- `feedback_monitor_gpu_at_training_start.md` (~716 tok)
- `feedback_phase1_oom_root_causes.md` — Declares GPUs (~578 tok)
- `feedback_pod_sync.md` (~343 tok)
- `feedback_pyc_purge_after_modeling_patch.md` (~907 tok)
- `feedback_python_unbuffered.md` (~418 tok)
- `feedback_smoke_max_loop_iters.md` (~855 tok)
- `MEMORY.md` — Mythic-RDT memory index (~1305 tok)
- `project_dscoder_5x_blocker.md` — Symptoms (transformers 5.6.2 + torch 2.11 + DS-Coder-V2-Lite-Instruct) (~1502 tok)
- `project_dscoder_5x_kvcache_blocker.md` (~977 tok)
- `project_flash_attn_install_blocked.md` — U __libc_single_threaded@GLIBC_2.32 (~1479 tok)
- `project_phase1_v2_catastrophic_regression.md` — Declares of (~1657 tok)
- `project_phase1_v2_gate_bias_dead.md` (~539 tok)
- `project_phase1_v3_t1_validation.md` (~838 tok)
- `project_phase1_v4_anchored_corrected_verdict.md` (~1205 tok)
- `project_phase1_v4_anchored_verdict.md` (~944 tok)
- `project_phase1_v5_verdict.md` (~838 tok)
- `project_phase1_v6_diagnosis.md` — src/mythic_rdt/modeling.py, _loop_step: (~1248 tok)
- `project_v6a_post_fix_verdict.md` (~1147 tok)
- `project_v6e_inference_path_bug.md` (~1183 tok)

## ../../../../tmp/

- `make_prompts.py` — Generate a 100-prompt JSONL on the pod: 50 HumanEval problems + 50 wiki-prose. (~430 tok)
- `pod_setup.sh` — Setup script run on the vast.ai pod after the project tar is uploaded. (~691 tok)

## ../.wolf/

- `buglog.json` (~10774 tok)
- `cerebrum.md` — Cerebrum (~1919 tok)

## ./

- `.gitignore` — Git ignore rules (~389 tok)
- `BASE_DEEPSEEK_CODER_V2_LITE.md` — Mythic-RDT Stage 1 Base — DeepSeek-Coder-V2-Lite-Instruct (~3010 tok)
- `BASE_DEEPSEEK_V2_LITE.md` — Mythic-RDT Stage 1 Base — DeepSeek-V2-Lite-Chat (~2316 tok)
- `BASE_GEMMA4_98E_V3.md` — Mythic-RDT Stage 2 Base — Gemma 4 26B-A4B 98e v3 (~1736 tok)
- `BASE_MODEL_ANALYSIS.md` — Mythic-RDT — Base Model Analysis (~1727 tok)
- `CLAUDE.md` — Mythic-Gemma4 — Working Notes (~2112 tok)
- `MASTER_PLAN.md` — Mythic-Gemma4 — Master Plan (~4672 tok)
- `README.md` — Project documentation (~1518 tok)
- `STATUS.md` — Mythic-RDT — Current Status (Stage 1) (~6413 tok)

## .caliber/

- `.caliber-state.json` (~45 tok)
- `dismissed-checks.json` (~288 tok)
- `error-log.md` — Generation Error — 2026-04-26T14:37:41.056Z (~126 tok)
- `manifest.json` (~796 tok)
- `refresh-hook.log` (~107 tok)
- `score-history.jsonl` (~57 tok)

## .caliber/backups/2026-04-26T14-49-42-697Z/

- `CLAUDE.md` — Mythic-Gemma4 — Working Notes (~1626 tok)

## .caliber/cache/

- `fingerprint.json` — Declares path (~11025 tok)

## .caliber/learning/

- `current-session.jsonl` (~2669 tok)
- `state.json` (~40 tok)

## .claude/

- `settings.json` (~1108 tok)

## .claude/hooks/

- `caliber-check-sync.sh` (~166 tok)
- `caliber-freshness-notify.sh` (~161 tok)
- `caliber-session-freshness.sh` (~161 tok)

## .claude/rules/

- `conversion-scripts.md` — Conversion script conventions (~354 tok)
- `eval-rules.md` — lm_eval mandatory rules (every invocation, no exceptions) (~391 tok)
- `modeling-shape.md` — HF custom-code shape (Mythic-Gemma4) (~426 tok)
- `openwolf.md` (~313 tok)

## .claude/skills/eval-gemma4/

- `SKILL.md` — eval-gemma4 (~2627 tok)

## .claude/skills/fetch-base-model/

- `SKILL.md` — Fetch Base Model (~2251 tok)

## .claude/skills/find-skills/

- `SKILL.md` — Find Skills (~616 tok)

## .claude/skills/llama-server-gemma4/

- `SKILL.md` — llama-server for Gemma 4 GGUF (~1816 tok)

## .claude/skills/mythic-rdt-surgery/

- `SKILL.md` — Mythic-RDT Surgery — Phase 1 Wrapper (~3243 tok)

## .claude/skills/phase0-sanity/

- `SKILL.md` — Phase 0 Sanity / Decision Gate (~2656 tok)

## .claude/skills/save-learning/

- `SKILL.md` — Save Learning (~637 tok)

## .claude/skills/setup-caliber/

- `SKILL.md` — Setup Caliber (~1941 tok)

## base/DeepSeek-Coder-V2-Lite-Instruct/ (LARGE — do not Read raw weights)

- `modeling_deepseek.py` — PyTorch DeepSeek model. PATCHED for transformers 5.x (is_torch_fx_available fallback, see .wolf/buglog.json bug-001). (~22587 tok)

## experiments/01_phase0_probe/

- `HYPOTHESIS_AND_RESULT.md` — Experiment 01 — Phase 0 layer-quality probe (Stage 1, DS-Coder-V2-Lite) (~2485 tok)

## scripts/

- `_diag_chat_decode.py` — One-off diagnostic: render chat prompt, run base.generate, dump raw output. (~612 tok)
- `_diag_minimal_gen.py` — Compare four base.generate configurations on a single HumanEval prompt: (~801 tok)
- `_diag_tokenizer.py` — Tokenizer roundtrip + decode method comparison. (~335 tok)
- `_dscoder_compat.py` — Cross-version compatibility helpers for DS-Coder-V2-Lite-Instruct. (~544 tok)
- `_probe_logit_identity.py` — Tiny logit-identity probe: wrapper(T=1, first_iter_identity=True) vs base. (~1503 tok)
- `_probe_v6e_identity.py` — Direct A/B: wrapper v6E (first_iter_identity, LoRA disabled at t=0) at T=1 vs base. (~1466 tok)
- `_probe_wrapper_generate_lcb.py` — LCB-realistic A/B: wrapper.generate(T=1, v6E) vs base.generate() byte-equality. (~2152 tok)
- `_probe_wrapper_generate.py` — End-to-end probe: wrapper.generate() vs base.generate() byte-equality at v6E + T=1. (~1478 tok)
- `finetune_phase1.py` — Mythic-RDT Stage 1 Phase 1 fine-tune entry point. (~6469 tok)
- `humaneval_smoke.py` — HumanEval-20 smoke test — Phase 1 sanity gate (Stage 1, DS-Coder-V2-Lite). (~13324 tok)
- `measure_base_loss.py` — Measure mean cross-entropy loss of the BASE model on the same packed (~838 tok)
- `phase0_probe_layers.py` — Phase 0 layer-quality probe (Stage 1, DS-Coder-V2-Lite-Instruct). (~6633 tok)
- `phase0_sanity.py` — Phase 0 hard gate (Stage 1, DS-Coder-V2-Lite-Instruct). (~2465 tok)
- `probe_then_v6e_chain.sh` — probe_then_v6e_chain.sh (~732 tok)
- `setup_pod_env.sh` — Set up the Mythic-RDT inference/eval/finetune environment on a vast.ai (~562 tok)

## scripts/pod_runner/

- `run_v6a_bigsmoke.sh` — v6A big-smoke: HE-164 + LCB-medium-30 at T=1/2/4 split-process. (~497 tok)
- `run_v6a.sh` — v6A: dual-T training with margin + distill, on top of architectural fix (~870 tok)
- `run_v6e.sh` — v6E: same recipe as v6A (dual-T 1↔4, margin + distill + KL-anchor) but with (~753 tok)

## src/mythic_rdt/

- `__init__.py` — Mythic-RDT: Recurrent-Depth Transformer wrapping MoE bases. (~190 tok)
- `configuration.py` — Mythic-RDT configuration classes. (~3342 tok)
- `loop_state.py` — Tiny contextvar holding the current recurrence loop iteration t. (~413 tok)
- `modeling.py` — Mythic-RDT modeling: thin wrapper around a frozen base MoE. (~9679 tok)
- `recurrence.py` — Recurrence machinery for Mythic-RDT. (~4420 tok)

## src/mythic_rdt/training/

- `__init__.py` — Mythic-RDT Phase 1 training pieces: depth-LoRA injection, curriculum, Trainer. (~160 tok)
- `curriculum.py` — T-curriculum for the recurrence loop during fine-tune. (~2313 tok)
- `data.py` — Streaming data pipeline: FineWeb-Edu prose + a code dataset, packed. (~2163 tok)
- `lora_inject.py` — Inject `DepthLoRA` adapters into the recurrent layer's projections. (~2428 tok)
- `trainer.py` — MythicRDTTrainer: a `transformers.Trainer` subclass that (~6942 tok)

## tests/

- `test_first_iter_identity.py` — Tests for the v6A architectural invariant: first_iter_identity. (~1236 tok)
- `test_load_trainable_state.py` — Tests for `_load_trainable_state` strictness — guards bug-050. (~1862 tok)
