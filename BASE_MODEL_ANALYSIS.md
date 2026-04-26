# Mythic-RDT — Base Model Analysis

**Decision (2026-04-26):** two-stage strategy.

- **Stage 1: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`** — 16 B / 2.4 B-A coding-specialized DeepSeekMoE, native MLA + shared experts, ~$600 fine-tune. **Headline benchmark = HumanEval pass@1**, base ~81 %, target ≥ 86 % at T=8. Publishable artifact in its own right; this is the **Mythic-Coder** release.
- **Stage 2: `ManniX-ITA/gemma-4-A4B-98e-v3-it`** — 20.8 B / 4 B-A general-reasoning Gemma 4 derivative, GQA + routed-only MoE, ~$2000 fine-tune. Triggered only if Stage 1 succeeds. High-quality general-reasoning release.

Detailed architecture specs in `BASE_DEEPSEEK_CODER_V2_LITE.md` (Stage 1) and `BASE_GEMMA4_98E_V3.md` (Stage 2). This file is the high-level decision record.

---

## Why two stages, in this order

The Gemma 4 path alone is risky: ~$2000 minimum just to find out if retrofit-recurrence works on a real MoE base, with a low ceiling (75 % → 78 % plausible) because the base is already near-saturated for its size class. If it fails, the money is gone with no published artifact.

DS-Coder-V2-Lite-Instruct as Stage 1 is **3× cheaper, 2× faster, architecturally pure (matches OpenMythos's spec verbatim), already strong** (HumanEval @ 81 %), and has a benchmark — HumanEval — where multi-step deliberation is exactly the right kind of compute. RDT amplification on coding is a *more compelling* story than on general reasoning, because hard programming problems benefit precisely from depth recurrence (plan → revise → patch). It's a publishable artifact in its own right *and* a gating de-risk for Stage 2.

| Aspect | Stage 1 (DS-Coder-V2-Lite-Instruct) | Stage 2 (Gemma 4 98e v3) |
|---|---|---|
| Total params | 15.7 B | 20.8 B |
| Active params/token | **2.4 B** | 4.0 B |
| Layers | 27 | 30 |
| Hidden dim | 2048 | 2816 |
| Attention | **MLA** (kv_lora_rank 512) | GQA (head_dim 256) |
| MoE | **64 routed + 2 shared, top-6** | 98 routed, top-8 (no shared) |
| Native context | **128k** | 32k |
| Disk size (bf16) | 31.4 GB | 39 GB |
| Reported headline benchmark | **HumanEval 81.1 %** | GPQA Diamond 75.25 % |
| Architectural fit for OpenMythos | **direct port** | adapted (no MLA, no shared experts) |
| Fine-tune cost (5 B tok) | $600 / 11 d on 2× H100 | $2000 / 11 d on 4× H100 |
| Fine-tune on 4× 3090 (free) | ~25 d feasible | ~50 d barely feasible |
| Realistic outcome at T=8 | HumanEval ~85–88 % (+4–7 pp) | GPQA ~78–82 % (+3–7 pp) |
| Headline | "16 B-storage code model that scales like 50 B at T=8" | "First high-quality general-reasoning RDT on Gemma 4" |

## Why the Coder variant of DS-V2-Lite (not V2-Lite-Chat)

V2-Lite-Chat scores HumanEval 57 %, MMLU 56 %, GPQA ~28 %. DS-Coder-V2-Lite-Instruct (same chassis, +6 T code tokens) scores HumanEval 81 %, GSM8K 86 %, MATH 62 %, MMLU 60 %. Both are 16 B / 2.4 B-A.

For Mythic-RDT specifically:

1. **HumanEval is a near-perfect RDT benchmark.** Hard programming problems require *plan → write → revise* — exactly what depth recurrence buys. Multi-step coding is what RDT was designed for.
2. **Already strong base = published artifact people will actually use.** Mythic-Coder at 86 % HumanEval is genuinely useful at 16 B storage; Mythic-Chat at 40 % GPQA from a 28 % base is academically interesting but not deployed.
3. **128 k context** (V2-Lite-Chat's 32 k extended via continued pretraining) opens long-document code reasoning experiments later.
4. **Same architecture** as V2-Lite-Chat — every OpenMythos-fit advantage is preserved.

## Why DS-Coder-V2-Lite is NOT enough alone (still want Stage 2)

1. **General reasoning generalization signal.** If retrofit-recurrence works on Mythic-Coder *and* Mythic-Gemma4 (different MoE topologies, different domains), that's stronger evidence the technique is a general one. One data point is suggestive; two are convincing.
2. **Different headlines reach different audiences.** Mythic-Coder reaches the "I want a useful 16 B code model" crowd. Mythic-Gemma4 reaches the "I want stronger reasoning on a popular base" crowd. Both have value.
3. **Reuse of phase-1 code for phase-2 conversion.** The recurrence harness, curriculum scripts, and HF custom-code packaging all carry over from Stage 1 → Stage 2. Marginal cost is mostly the GPU bill.

## Other candidates considered and rejected

| Candidate | Why not |
|---|---|
| DeepSeek-V2-Lite-Chat | General-reasoning sibling; superseded by Coder variant for our purposes (lower base scores, no HumanEval headline). Could still be a sister experiment if we want a "Mythic-Chat" variant. |
| DeepSeek-Coder-V2-Instruct (full 236 B) | Too big to fine-tune; even 4-bit is 60+ GB, training won't fit on any reasonable budget. |
| DeepSeek-V2 (236 B / 21 B-A) | Same problem — too big. |
| DeepSeek-V3 (671 B / 37 B-A) | Way too big; 4-bit is ~170 GB. |
| Qwen3-30B-A3B | GQA + routed-only MoE — same architectural mismatch as Gemma 4. No advantage over Gemma. |
| Qwen3-235B-A22B | Too big. |
| Qwen3-Coder-30B-A3B | GQA, no MLA, no shared experts. Architecturally weaker fit for OpenMythos than DS-Coder-V2-Lite. |
| Hunyuan-Large, MiniMax-Text-01, OLMoE-1B-7B | Too big, missing MLA, or missing MoE — all worse fits. |
| Gemma 4 128e original | 118 GB on disk (52 GB text-only). 1.5× larger than 98e v3 with identical 75.25 % GPQA. Use only if Stage 2 fails on 98e v3. |
| Gemma 4 109e v3 | 71.72 % GPQA — strictly dominated by 98e v3. |
| Gemma 4 120e hybrid / 64e / 96e / 16B / 31B / E4B | Various — see prior analysis; all dominated. |

## Stage gating

```
[Stage 0] Architecture skeleton + Phase 0 sanity gates on DS-Coder-V2-Lite-Instruct
   │
   ├── Pass: continue to Stage 1
   └── Fail: revisit recurrent layer / gate init; abort if 3 attempts fail
   │
[Stage 1] DS-Coder-V2-Lite-Instruct curriculum fine-tune (5 B tokens, ~$600)
   │
   ├── Pass criteria: T=8 HumanEval ≥ base + 5 pp; T=1 within 1 pp of base on
   │     HumanEval/MMLU; T=8 LiveCodeBench ≥ base + 4 pp; no mode collapse;
   │     no markdown-fence regression
   │     → Publish ManniX-ITA/Mythic-RDT-Coder-V2-Lite — 1st release
   │
   └── Fail: write up negative result, archive technique, do NOT advance to Stage 2
   │
[Stage 2 — gated on Stage 1 pass] Gemma 4 98e v3 curriculum fine-tune (5 B tokens, ~$2000)
   │
   ├── Pass criteria: T=8 GPQA Diamond ≥ 78 %; T=1 within 2 pp of base; no
   │     channel-token corruption
   │     → Publish ManniX-ITA/Mythic-RDT-Gemma4-26B-A4B-98e — 2nd release
   │
   └── Partial / fail: publish with caveats or document why Gemma is harder than DS
```

## Open log

- **2026-04-26** — Renamed project Mythic-Gemma4 → Mythic-RDT (multi-base scope). Stage 1 = DS-Coder-V2-Lite-Instruct (was: DS-V2-Lite-Chat — switched to Coder variant for stronger HumanEval headline and code-domain RDT story). Stage 2 = Gemma 4 98e v3. Architecture details in `BASE_DEEPSEEK_CODER_V2_LITE.md` and `BASE_GEMMA4_98E_V3.md`.
