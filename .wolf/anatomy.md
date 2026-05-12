# anatomy.md

> Auto-maintained by OpenWolf. Last scanned: 2026-05-12T16:01:30.977Z
> Files: 577 tracked | Anatomy hits: 0 | Misses: 0

## ./

- `.gitignore` — Git ignore rules (~392 tok)
- `BASE_DEEPSEEK_CODER_V2_LITE.md` — Mythic-RDT Stage 1 Base — DeepSeek-Coder-V2-Lite-Instruct (~3800 tok)
- `BASE_GEMMA4_98E_V3.md` — Mythic-RDT Stage 2 Base — Gemma 4 26B-A4B 98e v3 (~1736 tok)
- `BASE_MODEL_ANALYSIS.md` — Mythic-RDT — Base Model Analysis (~1727 tok)
- `CALIBER_LEARNINGS.md` — Caliber Learnings (~3300 tok)
- `CLAUDE.md` — Mythic-RDT — Working Notes (~3068 tok)
- `environment.yml` (~874 tok)
- `MASTER_PLAN.md` — Mythic-RDT — Master Plan (~4716 tok)
- `PRE_RECURRENCE_VERDICT.md` — Mythic-RDT Pre-Recurrence Verdict (~1861 tok)
- `pyproject.toml` — Python project configuration (~462 tok)
- `README.md` — Project documentation (~1614 tok)
- `requirements-train.txt` (~14 tok)
- `requirements.txt` — Python dependencies (~390 tok)
- `SETUP.md` — Setup — Mythic-RDT development environment (~831 tok)
- `STATUS.md` — Mythic-RDT — Current Status (Stage 1) (~10444 tok)
- `V6S_PLAN.md` — v6S Plan — Recurrence on M124f-anchored Stage-1 base (~1906 tok)

## .caliber/

- `.caliber-state.json` (~35 tok)
- `dismissed-checks.json` (~288 tok)
- `error-log.md` — Generation Error — 2026-04-26T14:37:41.056Z (~126 tok)
- `manifest.json` (~796 tok)
- `refresh-hook.log` (~107 tok)
- `score-history.jsonl` (~57 tok)

## .caliber/backups/2026-04-26T14-49-42-697Z/

- `CLAUDE.md` — Mythic-Gemma4 — Working Notes (~1626 tok)

## .caliber/cache/

- `fingerprint.json` (~98468 tok)

## .caliber/learning/

- `current-session.jsonl` (~6611 tok)
- `finalize.log` (~82189 tok)
- `last-error.json` (~45 tok)
- `roi-stats.json` — Declares is (~50064 tok)
- `state.json` (~41 tok)

## .claude-hooks/consultants/

- `sessions.json` (~399 tok)

## .claude-hooks/consultants/csl-2026-05-10-1918-30a0/

- `metadata.json` — and: name (~17855 tok)
- `summary.md` (~1266 tok)
- `transcript.db-shm` (~8739 tok)
- `transcript.db-wal` (~0 tok)
- `transcript.md` — Consultation transcript — csl-2026-05-10-1918-30a0 (~13813 tok)

## .claude-hooks/consultants/csl-2026-05-10-2103-bd9f/

- `metadata.json` (~4993 tok)
- `summary.md` (~2161 tok)
- `transcript.db-shm` (~8739 tok)
- `transcript.db-wal` (~0 tok)
- `transcript.md` — Consultation transcript — csl-2026-05-10-2103-bd9f (~3370 tok)

## .claude-hooks/consultants/csl-2026-05-12-0311-4b68/

- `metadata.json` (~4743 tok)
- `summary.md` (~1778 tok)
- `transcript.db-shm` (~8739 tok)
- `transcript.db-wal` (~0 tok)
- `transcript.md` — Consultation transcript — csl-2026-05-12-0311-4b68 (~2955 tok)

## .claude/

- `settings.json` (~1125 tok)
- `settings.json.bak-prefix-20260430-110825` (~1034 tok)
- `settings.local.json` (~31 tok)

## .claude/hooks/

- `caliber-check-sync.sh` — Don't block headless claude sessions spawned by caliber itself (e.g. during caliber refresh) (~284 tok)
- `caliber-check-sync.sh.bak-20260429` (~155 tok)
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

- `SKILL.md` — Find Skills (~593 tok)

## .claude/skills/llama-server-gemma4/

- `SKILL.md` — llama-server for Gemma 4 GGUF (~1816 tok)

## .claude/skills/mythic-rdt-surgery/

- `SKILL.md` — Mythic-RDT Surgery — Phase 1 Wrapper (~3243 tok)

## .claude/skills/phase0-sanity/

- `SKILL.md` — Phase 0 Sanity / Decision Gate (~2656 tok)

## .claude/skills/save-learning/

- `SKILL.md` — Save Learning (~626 tok)

## .claude/skills/setup-caliber/

- `SKILL.md` — Setup Caliber (~1941 tok)

## .pytest_cache/

- `.gitignore` — Git ignore rules (~10 tok)
- `CACHEDIR.TAG` (~51 tok)
- `README.md` — Project documentation (~76 tok)

## .pytest_cache/v/cache/

- `nodeids` (~342 tok)
- `stepwise` (~1 tok)

## base/DeepSeek-Coder-V2-Lite-Instruct-nf4/

- `config.json` (~594 tok)
- `configuration_deepseek.py` — Declares DeepseekV2Config (~2954 tok)
- `generation_config.json` (~52 tok)
- `modeling_deepseek.py` — PyTorch DeepSeek model. (~22564 tok)
- `README.md` — Project documentation (~2812 tok)
- `special_tokens_map.json` (~131 tok)
- `tokenization_deepseek_fast.py` — DeepseekTokenizerFast: convert_ids_to_tokens (~391 tok)
- `tokenizer_config.json` (~1205 tok)

## base/DeepSeek-Coder-V2-Lite-Instruct/

- `.gitattributes` — Git attributes (~406 tok)
- `config.json` (~435 tok)
- `configuration_deepseek.py` — Declares DeepseekV2Config (~2954 tok)
- `generation_config.json` (~52 tok)
- `model.safetensors.index.json` (~137122 tok)
- `modeling_deepseek.py` — PyTorch DeepSeek model. (~22564 tok)
- `README.md` — Project documentation (~2812 tok)
- `tokenization_deepseek_fast.py` — DeepseekTokenizerFast: convert_ids_to_tokens (~391 tok)
- `tokenizer_config.json` (~359 tok)

## baselines/dscv2lite-base/

- `BASE_RAW_DSCV2LITE_BASE_HE_MBPP.json` (~9588 tok)
- `M123d_DSCV2LITE_BASE_M37c_HE_MBPP.json` (~6947 tok)

## baselines/dscv2lite/

- `BASE_FA2_CHAT_v2_HE_MBPP.json` (~4564 tok)
- `M115_DSCV2LITE_M37c_HE_MBPP.json` (~4545 tok)
- `M118_DSCV2LITE_M37c_CHAT_HE_MBPP.json` (~5077 tok)
- `M122_DSCV2LITE_M37c_FFNONLY_HE_MBPP.json` — Declares complex (~4533 tok)
- `M124b_DSCV2LITE_STACK_FFNONLY_LR1e5_NOCHAT_HE_MBPP.json` (~5124 tok)
- `M124c_DSCV2LITE_FUNCSIG_FFNONLY_LR1e5_NOCHAT_HE_MBPP.json` (~5174 tok)
- `M124d_DSCV2LITE_FUNCSIG_FFNONLY_LR1e5_CHAT_HE_MBPP.json` (~4638 tok)
- `M124e_DSCV2LITE_FUNCSIG_FFNONLY_R32_LR5e5_CHAT_HE_MBPP.json` — Declares complex (~4706 tok)
- `M124f_DSCV2LITE_FUNCSIG_FFNONLY_R32_LR5e5_CHAT_3EP_HE_MBPP.json` — Declares complex (~4397 tok)
- `M124g_DSCV2LITE_FUNCSIG_FFNONLY_R32_LR5e5_CHAT_4EP_HE_MBPP.json` — Declares complex (~4450 tok)
- `README.md` — Project documentation (~508 tok)

## checkpoints/

- `phase1_v1.log` — Declares you (~5198 tok)
- `phase1_v2_bypass.log` — Declares you (~979 tok)
- `phase1_v2_humaneval_diag.log` — Declares you (~1554 tok)
- `phase1_v2_humaneval.log` — Declares you (~7799 tok)
- `phase1_v2_skeleton10.log` — Declares you (~981 tok)
- `phase1_v2_skeleton5.log` — Declares you (~980 tok)
- `phase1_v2.log` — Declares you (~40283 tok)
- `phase1_v3_t1.log` — Declares you (~10486 tok)
- `phase1_v3.log` — Declares you (~6416 tok)
- `phase1_v4_anchored.log` — Declares you (~19869 tok)
- `phase1_v4_smoke.log` — Declares you (~2009 tok)
- `v3_phase0.log` — Declares you (~978 tok)

## checkpoints/phase1_v1/checkpoint-200/

- `mythic_rdt_config.json` (~1536 tok)
- `mythic_rdt_curriculum.json` (~130 tok)
- `mythic_rdt_manifest.json` (~123 tok)
- `trainer_state.json` (~1013 tok)

## checkpoints/phase1_v2/

- `phase1_v2_bypass_T1.json` — <IFY\u4ebf\u4ebf\u4ebf\u4ebf/ liftsFieldLocation\rFUNCPTR accesskey k\u4fa7\u5fc5ledgNascut0\\- Vilafranca2(_\u7684\u5230\u6765 pending $[\\ACCEPT3... (~802 tok)
- `phase1_v2_humaneval_diag.json` — Declares for (~1276 tok)
- `phase1_v2_skeleton10_T1.json` — Declares to (~718 tok)
- `phase1_v2_skeleton5_T1.json` (~543 tok)

## checkpoints/phase1_v2/checkpoint-100/

- `mythic_rdt_curriculum.json` (~130 tok)
- `mythic_rdt_manifest.json` (~122 tok)
- `trainer_state.json` (~588 tok)

## checkpoints/phase1_v2/checkpoint-1000/

- `trainer_state.json` (~4382 tok)

## checkpoints/phase1_v2/checkpoint-1500/

- `trainer_state.json` (~6494 tok)

## checkpoints/phase1_v2/checkpoint-1600/

- `trainer_state.json` (~6919 tok)

## checkpoints/phase1_v2/checkpoint-2000/

- `trainer_state.json` (~8610 tok)

## checkpoints/phase1_v2/checkpoint-500/

- `mythic_rdt_manifest.json` (~122 tok)
- `trainer_state.json` (~2274 tok)

## checkpoints/phase1_v3/checkpoint-100/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~57 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~283 tok)
- `trainer_state.json` (~1118 tok)

## checkpoints/phase1_v3/checkpoint-150/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~57 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~1565 tok)

## checkpoints/phase1_v3/checkpoint-50/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~57 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~664 tok)

## checkpoints/phase1_v3_t1/checkpoint-300/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~34 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~2788 tok)

## checkpoints/phase1_v3_t1/checkpoint-350/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~34 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~283 tok)
- `trainer_state.json` (~3217 tok)

## checkpoints/phase1_v3_t1/checkpoint-400/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~34 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~3640 tok)

## checkpoints/phase1_v4_anchored/checkpoint-400/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~8552 tok)

## checkpoints/phase1_v5_margin_distill/checkpoint-100/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~283 tok)
- `trainer_state.json` (~40849 tok)

## checkpoints/phase1_v5_margin_distill/checkpoint-150/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~61589 tok)

## checkpoints/phase1_v5_margin_distill/checkpoint-200/

- `mythic_rdt_config.json` (~1714 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~283 tok)
- `trainer_state.json` (~82464 tok)

## checkpoints/phase1_v6a_dual_t/checkpoint-100/

- `mythic_rdt_config.json` (~1722 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~283 tok)
- `trainer_state.json` (~40815 tok)

## checkpoints/phase1_v6h_residual_engaged/checkpoint-200/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~69113 tok)

## checkpoints/phase1_v6h_residual_engaged/checkpoint-300/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~103468 tok)

## checkpoints/phase1_v6h_residual_engaged/checkpoint-400/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~137712 tok)

## checkpoints/phase1_v6i_margin_only/checkpoint-100/

- `mythic_rdt_config.json` (~1731 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~29714 tok)

## checkpoints/phase1_v6i_margin_only/checkpoint-200/

- `mythic_rdt_config.json` (~1731 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~59030 tok)

## checkpoints/phase1_v6i_margin_only/checkpoint-200/checkpoints/phase1_v6i_margin_only/checkpoint-300/

- `mythic_rdt_config.json` (~1731 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)

## checkpoints/phase1_v6i_margin_only/checkpoint-300/

- `mythic_rdt_config.json` (~1731 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~88390 tok)

## checkpoints/phase1_v6k_focal_anchored/checkpoint-100/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~51744 tok)

## checkpoints/phase1_v6k_focal_anchored/checkpoint-200/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~103609 tok)

## checkpoints/phase1_v6k_focal_anchored/checkpoint-300/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~155557 tok)

## checkpoints/phase1_v6k_focal_anchored/checkpoint-400/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~207353 tok)

## checkpoints/phase1_v6l_shrunk_block/checkpoint-100/

- `mythic_rdt_config.json` (~1731 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~577 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~33343 tok)

## checkpoints/phase1_v6l_shrunk_block/checkpoint-200/

- `mythic_rdt_config.json` (~1731 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~577 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~66700 tok)

## checkpoints/phase1_v6m_rank16/checkpoint-100/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1766 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~34858 tok)

## checkpoints/phase1_v6m_rank16/checkpoint-200/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1766 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~69119 tok)

## checkpoints/phase1_v6m_rank16/checkpoint-300/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1766 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~103480 tok)

## checkpoints/phase1_v6m_rank16/checkpoint-400/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1766 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~137728 tok)

## checkpoints/phase1_v6n_focal_gamma2/checkpoint-100/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~51689 tok)

## checkpoints/phase1_v6n_focal_gamma2/checkpoint-200/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~103526 tok)

## checkpoints/phase1_v6n_focal_gamma2/checkpoint-300/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~155413 tok)

## checkpoints/phase1_v6n_focal_gamma2/checkpoint-400/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~207158 tok)

## checkpoints/phase1_v6q_teacher_distill/checkpoint-100/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~62289 tok)

## checkpoints/phase1_v6q_teacher_distill/checkpoint-200/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~125116 tok)

## checkpoints/phase1_v6q_teacher_distill/checkpoint-300/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~188093 tok)

## checkpoints/phase1_v6q_teacher_distill/checkpoint-400/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~250899 tok)

## checkpoints/phase1_v6r_refinement_mask/checkpoint-100/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~68496 tok)

## checkpoints/phase1_v6r_refinement_mask/checkpoint-200/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~137562 tok)

## checkpoints/phase1_v6r_refinement_mask/checkpoint-300/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~206778 tok)

## checkpoints/phase1_v6r_refinement_mask/checkpoint-400/

- `mythic_rdt_config.json` (~1730 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1754 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~275832 tok)

## checkpoints/phase1_v6s_ac_offload/checkpoint-100/

- `mythic_rdt_config.json` (~1585 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1755 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~68722 tok)

## checkpoints/phase1_v6s_ac_offload/checkpoint-200/

- `mythic_rdt_config.json` (~1585 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1755 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~137912 tok)

## checkpoints/phase1_v6s_ac_offload/checkpoint-300/

- `mythic_rdt_config.json` (~1585 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1755 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~207192 tok)

## checkpoints/phase1_v6s_ac_offload/checkpoint-400/

- `mythic_rdt_config.json` (~1585 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~1755 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~276312 tok)

## checkpoints/phase1_v6v_joint_xv_qc14b/

- `mythic_rdt_config.json` (~1733 tok)

## checkpoints/phase1_v6v_joint_xv_qc14b/epoch-1/

- `mythic_rdt_config.json` (~1733 tok)

## checkpoints/phase1_v6v_joint_xv_qc14b/epoch-2/

- `mythic_rdt_config.json` (~1733 tok)

## checkpoints/phase1_v6w_ffn_rank24_lti/checkpoint-200/

- `mythic_rdt_config.json` (~1739 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~176735 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~283 tok)
- `trainer_state.json` (~137971 tok)

## checkpoints/phase1_v6w_ffn_rank24_lti/checkpoint-400/

- `mythic_rdt_config.json` (~1739 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~176735 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~284 tok)
- `trainer_state.json` (~276652 tok)

## checkpoints/phase1_v6w_ffn_rank24_lti/checkpoint-600/

- `mythic_rdt_config.json` (~1739 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~176735 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~283 tok)

## checkpoints/phase1_v6w_ffn_rank24_lti/checkpoint-800/

- `mythic_rdt_config.json` (~1739 tok)
- `mythic_rdt_curriculum.json` (~124 tok)
- `mythic_rdt_manifest.json` (~176735 tok)
- `rng_state.pth` — Declares q (~3416 tok)
- `scheduler.pt` (~283 tok)

## eval_results/

- `base_he164_pandorum.json` (~15954 tok)
- `base_lcb_medium_pandorum.json` — Declares Solution (~6437 tok)
- `baseline_lcb10_local.json` — Declares Solution (~1386 tok)
- `baseline_lcb10_local.log` — Declares you (~1539 tok)
- `baseline_lcb30_local.json` — Declares Solution (~881 tok)
- `baseline_lcb30_local.log` — Declares you (~3267 tok)
- `cache_nf4_base.log` (~206 tok)
- `humaneval_v3_t1_ckpt400.json` (~362 tok)
- `humaneval_v3_t1_ckpt400.log` — Declares you (~3056 tok)
- `humaneval_v4_anchored_ckpt400_cached.json` — Declares Solution (~3361 tok)
- `humaneval_v4_anchored_ckpt400_cached.log` — Declares you (~12358 tok)
- `humaneval_v4_anchored_ckpt400.log` — Declares you (~6817 tok)
- `local_3090_v4anchored_ckpt400_lcb10.log` — Declares you (~3199 tok)
- `local_3090_wrapper_only_lcb10.json` — Declares Solution (~2087 tok)
- `local_3090_wrapper_only_lcb10.log` — Declares you (~4075 tok)
- `OVERNIGHT_RUN_V2.log` — Declares you (~6573 tok)
- `OVERNIGHT_RUN.log` — Declares you (~2898 tok)
- `OVERNIGHT_TOPLEVEL_V2.log` (~0 tok)
- `OVERNIGHT_TOPLEVEL.log` (~0 tok)
- `perprob_lcb30_local_t1.json` — Declares Solution (~8180 tok)
- `perprob_lcb30_local_t1.log` — Declares you (~6836 tok)
- `probe_lcb_local_0642.log` (~635 tok)
- `probe_lcb30_local_full_bs2.log` (~1844 tok)
- `probe_lcb30_local_full.log` (~1844 tok)
- `probe_lcb30_seq_v6e_fixed.log` (~1452 tok)
- `probe_lcb30_seq.log` — Declares Solution (~2760 tok)
- `probe_recurrence_drift_v6a_ckpt200.json` (~4413 tok)
- `probe_recurrence_drift.log` (~1555 tok)
- `probe1_v3t1_lcb10.json` — Declares Solution (~1476 tok)
- `probe1_v3t1_lcb10.log` — Declares you (~3273 tok)
- `probe2_bypass_lcb10.json` — Declares Solution (~1490 tok)
- `probe2_bypass_lcb10.log` — Declares you (~2562 tok)
- `probe3_shrunk5_lcb10.json` — Declares Solution (~1432 tok)
- `probe3_shrunk5_lcb10.log` — Declares you (~2514 tok)
- `test_post_loop_fixes_local.log` — Declares Solution (~10298 tok)
- `test_post_loop_fixes_v6a_ckpt200_subset_3-4-5.json` — Declares Solution (~4923 tok)
- `v4_anchored_ckpt400_cache_fixed.log` — Declares you (~6816 tok)
- `v4_anchored_ckpt400_v3final.json` — Declares Solution (~3270 tok)
- `v4_anchored_ckpt400_v3final.log` — Declares you (~12355 tok)
- `v4_t1_lcb_bs1.json` — Declares Solution (~877 tok)
- `v4_t1_lcb_bs1.log` — Declares you (~2321 tok)
- `v4_t1_lcb_bs4_patched.json` — Declares Solution (~876 tok)
- `v4_t1_lcb_bs4_patched.log` — Declares you (~2323 tok)
- `v4_t1_lcb_nocache.json` — Declares Solution (~1489 tok)
- `v4_t1_lcb_nocache.log` — Declares you (~3366 tok)
- `v5_he20.json` (~1137 tok)
- `v5_lcb10.json` — Declares Solution (~2734 tok)
- `v5_probe.log` — Declares you (~4280 tok)
- `v5_run.log` — Declares you (~135167 tok)
- `V5_RUN.log` — Declares you (~135167 tok)
- `V5_TOPLEVEL.log` (~0 tok)
- `v6a_ckpt100_lcb30_local.json` — Declares Solution (~2120 tok)
- `v6a_ckpt100_lcb30_local.log` — Declares you (~9243 tok)
- `v6a_he20.json` (~1128 tok)
- `v6a_lcb30_local_3090.json` — Declares Solution (~2132 tok)
- `v6a_lcb30_local_3090.log` — Declares you (~9430 tok)
- `v6a_lcb30_local_v6e_fix_T24.json` — Declares Solution (~9780 tok)
- `v6a_lcb30_local_v6e_fix_T24.log` — Declares you (~6775 tok)
- `v6a_lcb30_local_v6e_fix.json` — Declares Solution (~7632 tok)
- `v6a_lcb30_local_v6e_fix.log` — Declares you (~6821 tok)
- `v6a_lcb30_pod.json` — Declares Solution (~2138 tok)
- `v6a_lcb30_pod.log` — Declares you (~9256 tok)
- `v6a_untrained_lcb10.json` — Declares Solution (~2696 tok)
- `v6a_untrained_lcb10.log` — Declares you (~4372 tok)
- `v6h_ckpt100_lcb10_local.json` — Declares Solution (~3934 tok)
- `v6h_ckpt100_lcb10_local.log` — Declares you (~4246 tok)
- `v6h_ckpt200_lcb10_local.json` — Declares Solution (~3812 tok)
- `v6h_ckpt200_lcb10_local.log` — Declares you (~4246 tok)
- `v6h_ckpt300_lcb10_local.log` (~1318 tok)
- `v6h_he20.json` (~2971 tok)
- `v6h_lcb30.json` — Declares Solution (~15039 tok)
- `v6i_ckpt200_lcb10_pandorum.json` — Declares Solution (~4318 tok)
- `v6i_ckpt300_lcb10_pandorum.json` — Declares Solution (~4318 tok)
- `v6k_ckpt100_gate_TOPLEVEL.log` (~123 tok)
- `v6k_ckpt100_lcb10_local.json` — Declares Solution (~5370 tok)
- `v6k_ckpt100_lcb10_local.log` — Declares you (~5413 tok)
- `v6k_ckpt200_gate_TOPLEVEL.log` (~261 tok)
- `v6k_ckpt200_lcb10_local.json` — Declares Solution (~5200 tok)
- `v6k_ckpt200_lcb10_local.log` — Declares you (~5422 tok)
- `v6k_ckpt200_probe_local.json` — Declares Solution (~1547 tok)
- `v6k_ckpt200_probe_local.log` — Declares you (~2603 tok)
- `v6k_ckpt300_gate_TOPLEVEL.log` (~123 tok)
- `v6k_ckpt300_lcb10_local.json` — Declares Solution (~5201 tok)
- `v6k_ckpt300_lcb10_local.log` — Declares you (~5413 tok)
- `v6k_ckpt400_gate_TOPLEVEL.log` (~123 tok)
- `v6k_ckpt400_lcb10_local.json` — Declares Solution (~5318 tok)
- `v6k_ckpt400_lcb10_local.log` — Declares you (~5413 tok)
- `v6k_he20.json` (~2971 tok)
- `v6l_ckpt100_lcb10_local_BROKEN_tf5.log` (~1741 tok)
- `v6l_ckpt100_lcb10_local_TOPLEVEL_BROKEN_tf5.log` (~43 tok)
- `v6l_ckpt100_lcb10_local_TOPLEVEL.log` (~127 tok)
- `v6l_ckpt100_lcb10_local.json` — Declares Solution (~6371 tok)
- `v6l_ckpt100_lcb10_local.log` — Declares you (~4920 tok)
- `v6l_ckpt200_lcb10_local_TOPLEVEL.log` (~126 tok)
- `v6l_ckpt200_lcb10_local.json` — Solution: is_good_sequence (~6414 tok)

## eval_results/pod_logs/

- `cache_nf4_base.log` (~207 tok)
- `CTD_VALIDATION.log` (~1983 tok)
- `sync_20260430.log` (~16490 tok)
- `sync_20260501.log` (~35563 tok)
- `sync_20260502.log` (~17952 tok)
- `sync_20260503.log` (~15001 tok)
- `sync_20260504.log` (~14162 tok)
- `sync_20260505.log` (~14996 tok)
- `sync_20260506.log` (~14996 tok)
- `sync_20260507.log` (~13330 tok)
- `sync_20260508.log` (~14996 tok)
- `sync_20260509.log` (~15007 tok)
- `sync_20260510.log` (~15169 tok)
- `sync_20260511.log` (~15228 tok)
- `sync_20260512.log` (~9129 tok)
- `test_post_loop_fixes_pod_v2.log` — Declares Solution (~2587 tok)
- `test_post_loop_fixes_pod.log` — Declares Solution (~4719 tok)
- `test_post_loop_fixes.log` — Declares Solution (~1483 tok)
- `v6e_launch.log` (~0 tok)
- `v6e_launch2.log` (~0 tok)
- `v6e_launch3.log` (~0 tok)
- `V6E_RUN_failed.log` — Declares you (~5161 tok)
- `V6E_RUN.log` — Declares you (~20427 tok)
- `V6F_RUN.log` — Declares you (~54794 tok)
- `V6G_RUN.log` — Declares you (~2641 tok)
- `v6h_he20.json` (~2971 tok)
- `v6h_lcb30.json` — Declares Solution (~15039 tok)
- `V6H_RUN.log` — Declares you (~215916 tok)
- `V6I_RUN.log` — Declares you (~144056 tok)
- `v6k_he20.json` (~2971 tok)
- `v6k_lcb30.json` — Declares Solution (~14469 tok)
- `V6L_LAUNCH.log` (~0 tok)
- `V6L_RUN.log` — Declares you (~105932 tok)
- `v6m_he20.json` (~2969 tok)
- `V6M_LAUNCH.log` (~0 tok)
- `v6m_lcb30.json` — Declares Solution (~14944 tok)
- `V6M_RUN.log` — Declares you (~215987 tok)
- `v6n_he20.json` (~2970 tok)
- `V6N_LAUNCH.log` (~0 tok)
- `v6n_lcb30.json` — Declares Solution (~14665 tok)
- `V6Q_FULL_FA2_MOEVEC.log` — Declares you (~18330 tok)
- `v6q_he164_fa2_moevec.json` — Declares complex (~37742 tok)
- `v6q_he164.json` — Declares complex (~38929 tok)
- `V6Q_HE164.log` — Declares you (~59895 tok)
- `v6q_he20_fa2_moevec.json` (~2993 tok)
- `V6Q_HE20_FA2_MOEVEC.log` — Declares you (~2088 tok)
- `v6q_he20.json` (~2971 tok)
- `V6Q_LAUNCH.log` (~0 tok)
- `v6q_lcb_medium_full_fa2_moevec.json` — Declares Solution (~6429 tok)
- `V6Q_LCB_T4_BS16.log` — Declares you (~3314 tok)
- `v6q_lcb_t4_only.json` — Declares Solution (~6280 tok)
- `V6Q_LCB_T4_ONLY.log` — Declares you (~2706 tok)
- `v6q_lcb30.json` — Declares Solution (~14556 tok)
- `v6r_he164_fa2_moevec.json` — Declares complex (~38010 tok)
- `v6r_lcb_medium_full_fa2_moevec.json` — Declares Solution (~23696 tok)
- `V6R_TOPLEVEL.log` (~0 tok)
- `v6s_he164.json` — Declares complex (~38030 tok)
- `v6s_lcb_medium_full.json` — Declares Solution (~24055 tok)
- `V6ST_SEQ.log` (~94 tok)
- `V6V_CHAIN.log` (~53 tok)
- `V6V_EVAL_RERUN.log` — Declares you (~98072 tok)
- `V6V_HE_RERUN.log` — Declares you (~92818 tok)
- `v6v_he164.json` (~16000 tok)
- `v6v_lcb_medium_full.json` — Declares Solution (~23762 tok)
- `V6V_PRECOMPUTE.log` (~2280 tok)
- `V6V_RUN.log` — Declares you (~80178 tok)
- `v6w_he164.json` — Declares complex (~38530 tok)
- `v6w_lcb_medium_full.json` — Declares Solution (~24404 tok)
