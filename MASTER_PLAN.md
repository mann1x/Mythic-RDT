# Mythic-RDT — Master Plan

**Status**: kickoff (2026-04-26).
**Author**: ManniX-ITA + Claude.
**Goal**: convert an existing MoE base into a Recurrent-Depth Transformer following the OpenMythos blueprint, using fine-tuning only — no from-scratch pretraining — and ship as a `transformers`-loadable model.

**Two-stage strategy** (`BASE_MODEL_ANALYSIS.md`):

- **Stage 1: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`** — 15.7 B / 2.4 B-A coding-specialized DeepSeekMoE. Native MLA + shared experts. **Headline benchmark = HumanEval pass@1**, base ~81 %. Target T=8 ≥ 86 %. Spec: `BASE_DEEPSEEK_CODER_V2_LITE.md`. Published as **`ManniX-ITA/Mythic-RDT-Coder-V2-Lite`** (a.k.a. **Mythic-Coder**).
- **Stage 2: `ManniX-ITA/gemma-4-A4B-98e-v3-it`** — 20.8 B / 4 B-A general-reasoning. GQA + routed-only MoE. Headline = GPQA Diamond, base 75.25 %, target T=8 ≥ 78 %. Spec: `BASE_GEMMA4_98E_V3.md`. Triggered only if Stage 1 succeeds. Published as **`ManniX-ITA/Mythic-RDT-Gemma4-26B-A4B-98e`**.

**Method**: OpenMythos as the reference for the recurrence machinery (LTI injection, depth-LoRA, MoE-aware loop), retrofit-recurrence curriculum (arXiv 2511.07384) for the fine-tune procedure.

---

## 1. Vision

A multi-base RDT recipe where the user picks `n_loops` at inference time:

- `n_loops=1` ≈ baseline cost.
- `n_loops=4–8` ≈ stronger reasoning, ~4–8× compute, no extra storage.
- `n_loops=16` ≈ deepest reasoning mode, depth-extrapolated beyond training.

End-users load via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and call `model.generate(..., n_loops=8)`.

---

## 2. Feasibility verdict

**Feasible — Stage 1 is high-confidence, Stage 2 is medium-confidence.**

### Why it's feasible

- The retrofit-recurrence paper (arXiv 2511.07384) demonstrates the exact technique: take a pretrained transformer, freeze most of it, add LTI injection + identity-biased gating + depth-LoRA, fine-tune with a curriculum of increasing recurrence depth.
- OpenMythos is permissively MIT licensed — we lift LTI injection, depth-LoRA, gating modules verbatim.
- Stage 1 base (DS-Coder-V2-Lite-Instruct) is a **direct port target** for OpenMythos: MLA and shared experts are native to the architecture.
- Coding benchmarks (HumanEval, MBPP) reward multi-step deliberation — exactly what depth recurrence buys. Better fit than general reasoning at this size class.

### Why it's hard

- The base's middle layer was trained for one-shot use, not T-loop reuse. Fine-tune must teach the experts to be position-invariant in depth.
- 64-routed-expert MoE with depth-distinct routing is novel at this scale. Default: shared router with rank-16 depth-LoRA conditioned on a depth embedding.
- LTI stability with A/B injection at 16 B–21 B scale and T=16 has no public precedent. Spectral radius constraint helps but is not a guarantee.
- For Stage 2 (Gemma 4): bf16 NaN bug in middle layers (parent project) — must use fp32 RMSNorm in the recurrence path.

### Verdict

Feasible at **academic-grade two-checkpoint release** with a Stage-1 fine-tune budget of ~5 B tokens on 2× H100 (~11 days, ~$600), gating Stage 2 at ~5 B tokens on 4× H100 (~11 days, ~$2000).

The realistic outcome:
- **Stage 1 (Mythic-Coder)**: HumanEval pass@1 81 % → 86 % at T=8, no storage cost increase.
- **Stage 2 (Mythic-Gemma4)**: GPQA Diamond 75 % → 78 % at T=8, no storage cost increase.

---

## 3. Port matrix — what comes from where (per stage)

### Stage 1 (DS-Coder-V2-Lite-Instruct)

| Component | Source | Strategy | Risk |
|---|---|---|---|
| Token embedding + lm_head | DS-Coder | Reuse verbatim, frozen | low |
| Prelude layer (1) | DS-Coder layer 0 (dense FFN) | Reuse verbatim, frozen | low |
| Recurrent block (1 MoE layer) | DS-Coder middle layer (~13) | Reuse weights, train depth-LoRA + LTI on top | **high** — main learning target |
| Coda layer(s) (1–2) | DS-Coder last layer(s) | Reuse verbatim, frozen first phase | low |
| **MLA attention** | DS-Coder | **Reuse verbatim** (native!) | low |
| **Shared experts (2)** | DS-Coder | **Reuse verbatim** (always-on, no LoRA) | low |
| Routed experts (64) | DS-Coder | Reuse verbatim, frozen | low |
| MoE router | DS-Coder + depth-LoRA | Add T-aware rank-16 LoRA | med |
| RMSNorm / rotary | DS-Coder | Reuse, fp32 norms in recurrence path (defensive) | low |
| LTI injection (A, B) | OpenMythos (Parcae) | New params | med |
| Identity-biased gating | Retrofit-recurrence paper | New per-loop scalar | med |
| Per-loop LayerScale | LayerScale literature | Init 1e-4 | low |
| Depth embedding | New | Sinusoidal or learned, fed into router LoRA | low |
| ACT halting | OpenMythos | **DEFERRED** (fixed T for v0) | n/a |
| Tokenizer + chat template | DS-Coder | Reuse verbatim | low |

**Total new trainable params: ~7 M.** Frozen: ~15.7 B.

### Stage 2 (Gemma 4 98e v3)

Same as above except:

- **MLA**: not present in Gemma 4 → keep GQA.
- **Shared experts**: not present in Gemma 4 → skip.
- **Recurrent layer**: Gemma 4 middle layer (~15), avoiding bf16-broken layers 11–29 — pick from `../scripts/expert_neuron_v4.json`.
- **Reasoning channel tokens**: Gemma 4 emits `<|channel>thought` tokens; recurrence must not corrupt them.

---

## 4. Architecture spec — Mythic-RDT v0

```
input_ids → embed → [Prelude: base layer 0 (and maybe 1) frozen]
                → e (encoded input, kept around)
                → h_0 = Prelude(embed)
                → for t in 0..T-1:
                    inj = A·h_t + B·e                   # LTI injection (A, B learnable)
                    block_out = RecurrentBlock(h_t, e)  # base middle layer w/ depth-LoRA
                    g = sigmoid(gate_t)                  # identity-biased gate
                    ls = layerscale_t                   # per-loop LayerScale
                    h_{t+1} = h_t + ls · g · (inj + block_out)
                → [Coda: base last 1-2 layers frozen]
                → norm + lm_head → logits
```

Key constants for first build (both stages, unless noted):

- `prelude_layers`: 1 for Stage 1 (only layer 0 is dense), 2 for Stage 2 (Gemma)
- `coda_layers`: 1 (Stage 1), 2 (Stage 2)
- `recurrent_layer_idx`: 13 (Stage 1, middle of 27 layers), TBD-from-`expert_neuron_v4.json` (Stage 2)
- `train_loop_iters`: 8
- `max_loop_iters`: 16 (inference cap, depth-extrapolatable)
- `depth_lora_rank`: 8 for Q/K/V/O (and the MLA-specific projections); 16 for router LoRA
- `lti_init`: `log_A ~ Uniform(0.01, 0.1)`, `B = zeros + tiny noise`
- `gate_init`: bias = -3 (sigmoid ≈ 0.047 at start)
- `layerscale_init`: 1e-4
- `halting_strategy`: `"fixed"` (ACT later)

---

## 5. Phased roadmap

### Phase 0 — Feasibility prototype (1 week, runs on Stage 1 base)

**Goal**: prove the wrapper runs and doesn't immediately diverge.

- [ ] **Fetch Stage 1 base from HF**: `huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --local-dir base/DeepSeek-Coder-V2-Lite-Instruct --local-dir-use-symlinks False`. Verify total size ~31.4 GB.
- [ ] Decide vendoring strategy: subclass `DeepseekV2ForCausalLM` from upstream `modeling_deepseek.py` (option a), or vendor the modeling code into `src/mythic_rdt/` (option b). Default: (a) for v0, revisit before publishing.
- [ ] Create the empty Python package skeleton (`src/mythic_rdt/`, `pyproject.toml`).
- [ ] Build minimal `MythicRDTConfig` extending `DeepseekV2Config` (Stage 1 config subclass: `MythicRDTDeepseekV2Config`).
- [ ] Build minimal `MythicRDTDeepseekV2ForCausalLM` that loads DS-Coder-V2-Lite weights and runs at T=1 with `gate=0` (identity passthrough through chosen middle layer).
- [ ] **Sanity check**: at T=1 with `gate=0`, output must match running base's middle layer once on the prelude output. Bit-exact in fp32. **Hard gate.**
- [ ] Pick `recurrent_layer_idx`: probe layers 10/13/16 untrained at T=1/T=4/T=8 on 100 prompts (mix of HumanEval-style + FineWeb-Edu prose). Choose cleanest behavior.
- [ ] Decision gate: if T=8 untrained drops > 50 % PPL or produces gibberish on > 20 % of prompts, the architecture choice (which middle layer, what gate init) needs revisiting before phase 1.

### Phase 1 — Stage 1 surgery and conversion (1 week)

- [ ] Conversion script `scripts/convert_dscoder_to_mythic.py base_model_path output_path` that:
  1. Loads DS-Coder-V2-Lite-Instruct weights via `safetensors`.
  2. Strips middle layers (keeps prelude + 1 recurrent + coda).
  3. Initializes LTI / gating / LayerScale / depth-LoRA to safe defaults.
  4. Saves as a Mythic-RDT-Coder checkpoint with `auto_map` registered.
- [ ] Verify: loaded model runs `model.generate(...)` at T=1, T=4, T=8 without crashes.
- [ ] First eval: HumanEval `--limit 20` at T=1 vs base — should be near-parity (~80 % since at T=1 we're effectively a 3-layer slice of DS-Coder). If wildly off, the surgery is wrong.
- [ ] **Critical**: HumanEval eval must use `local-completions` + `/v1/completions` (raw text), NOT `local-chat-completions` — chat mode wraps in markdown fences which break the scorer. See parent project bug-015.

### Phase 2 — Stage 1 curriculum fine-tune (2–4 weeks, longest phase)

- [ ] Compute target: 2× H100 vast.ai (~$600) for ~5 B tokens, OR 4× 3090 on solidpc (~25 days free).
- [ ] Dataset mix (50/50): FineWeb-Edu (general prose preservation) + The-Stack-V2 / open-instruct-code (code preservation). Tokenizer = DS-Coder's. Plus an instruct slice (Magpie-Coder, OpenHermes-Coder).
- [ ] Curriculum (token splits approximate):
  1. **Warm-up** (10 % of budget): freeze everything except LayerScale + gates. Train at T=2. Goal: gates open from 0 to nonzero without diverging.
  2. **Recurrence-2** (20 %): unfreeze A, B, depth-LoRA[t=0,1]. Train at T=2, mixed with T=1 forward sometimes.
  3. **Recurrence-4** (25 %): T=2,4 mixed.
  4. **Recurrence-8** (30 %): T=4,8 mixed.
  5. **Anneal to target T=16** (15 %): T=8,16 mixed.
- [ ] Loss: standard CE on next-token + small auxiliary KL between T=1 output and original DS-Coder output (on a held-out batch) to discourage T=1 drift. Plus DeepSeek's MoE aux load-balance loss (α₁ = 0.001, raise to 0.005 if utilization collapses).
- [ ] Validation: held-out FineWeb-Edu PPL + HumanEval `--limit 20` at every checkpoint.
- [ ] Stop criterion: T=8 HumanEval-20 ≥ base, and T=1 PPL within 1 % of base PPL on FineWeb-Edu val.

### Phase 3 — Stage 1 full eval + iteration (1 week)

- [ ] Full HumanEval pass@1 at T=1, T=4, T=8, T=16. Reference target: base ~81 %.
- [ ] HumanEval+ pass@1 at T=1, T=8.
- [ ] MBPP+ pass@1 at T=1, T=8.
- [ ] LiveCodeBench at T=1, T=8.
- [ ] GSM8K at T=1, T=8 (math sanity).
- [ ] MMLU subset at T=1 (general-knowledge drift check).
- [ ] **Mandatory** sample sanity check on every score: empty / markdown-fence / <5-char junk count next to pass@1.
- [ ] If T=8 HumanEval < base + 5 pp, root-cause and re-curriculum.

### Phase 4 — Stage 1 custom-code transformers package (1 week)

- [ ] Move modeling code to repo root: `configuration_mythic_rdt.py`, `modeling_mythic_rdt.py` (HF custom-code convention).
- [ ] Add `auto_map` to config so `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` works.
- [ ] Vendor or import DeepSeek's MLA + DeepseekMoE modules cleanly (decide between subclass-on-import vs vendored copy).
- [ ] Test load on a fresh machine: `pip install transformers && python -c "..."` works.
- [ ] GGUF export: out of scope for v0 (llama.cpp doesn't support recurrent depth). Document this clearly in README.

### Phase 5 — Stage 1 release (3 days)

- [ ] HF model repo: `ManniX-ITA/Mythic-RDT-Coder-V2-Lite` (BF16 + custom code).
- [ ] README with: architecture, how to load, recommended `n_loops` for tasks, eval table (HumanEval/HumanEval+/MBPP+/LiveCodeBench/GSM8K/MMLU at T=1,4,8,16), reproducibility section, citations (DeepSeek-Coder-V2 + OpenMythos + Parcae + Retrofit-Recurrence + Gemma 4 inheritance via parent project).
- [ ] Optional: 4-bit and 8-bit bnb variants.
- [ ] Blog post / X thread.
- [ ] Issue thread for community feedback.

### Stage gate after Phase 5 — decide on Stage 2

- **Pass criteria**: T=8 HumanEval ≥ base + 5 pp; T=1 within 1 pp of base on HumanEval/MMLU; T=8 LiveCodeBench ≥ base + 4 pp; no mode collapse; no markdown-fence regression.
- If pass: advance to Stage 2 (Gemma 4 98e v3) — recipe is mostly portable; budget ~$2000.
- If fail: write up negative result, archive technique, do NOT spend Stage 2 budget.

### Phase 6 — Stage 2 (Gemma 4 98e v3), gated on Stage 1

Mirror of Phases 0–5 but on Gemma 4 base. Reuse:

- Recurrence harness code (LTI, gates, LayerScale, depth-LoRA, curriculum loop) — verbatim, just point at the Gemma 4 layer.
- Curriculum recipe — same.
- Eval scripts — modify llama-server invocation to add `--reasoning-format deepseek --reasoning-budget 8192` (Gemma 4 specific).

Gemma-specific work:

- [ ] Pick `recurrent_layer_idx` from `../scripts/expert_neuron_v4.json` (avoid bf16-broken layers).
- [ ] Address chat-template gap (HF 98e-v3-it template is 4 KB larger than the local intermediate — see `BASE_GEMMA4_98E_V3.md`).
- [ ] Skip MLA + shared-experts code paths (Gemma 4 has neither).
- [ ] Verify `<|channel>thought` reasoning tokens survive recurrence.
- [ ] Eval suite: GPQA Diamond + HumanEval + MMLU subset (HumanEval can use `local-chat-completions` for Gemma, since Gemma's reasoning template is well-behaved with the right flags).

### Phase 7 — Polish (deferred, post-release)

- ACT halting head (PonderNet-style) — train halting predictor on a frozen base.
- MLA conversion for Gemma 4 (Stage 2 retrofit) — small fine-tune to compress KV.
- Apply same recipe to Qwen3-30B-A3B (different MoE topology, same family as Gemma).
- Mythic-RDT-Chat (DS-V2-Lite-Chat base, general-reasoning sister to Mythic-Coder).
- Long-context experiments using DS-Coder's 128k native window.

---

## 6. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| One MoE layer reused T=16 times mode-collapses regardless of fine-tune budget | high | Phase 0 sanity gate. If untrained T=8 is total junk, fall back to a 2-layer recurrent block |
| LTI A/B at 16-21 B scale don't stabilize even with spectral constraint | med | LayerScale + gate init below 0.05 + curriculum. Worst case: drop LTI, keep gate-only retrofit |
| HumanEval scorer fails silently due to markdown fences (parent bug-015) | med | Use `local-completions` + `/v1/completions` for HumanEval; sanity-check `samples_*.jsonl` for fences |
| MoE router can't handle depth-distinct routing without rebalancing | med | Phase 1 sanity: log expert utilization across loop iterations. If <30 % experts ever fire, raise α₁ aux loss to 0.005 |
| DS-Coder's chat template doesn't survive recurrence | med | Phase 0 generation sanity check on chat-formatted prompts |
| Fine-tune budget too small to recover quality | med | Start with 1 B token pilot, project quality curve, scale up only if curve looks healthy |
| `transformers` custom-code path breaks on new transformers releases | low | Pin compatible transformers version range. Test against transformers 5.4, 5.5, 5.6 |
| Code-specialization degrades under prose-only fine-tune | med | Use 50/50 prose/code mix; never train on prose alone |
| bf16 NaN in norms (Gemma 4 stage only) | low | fp32 RMSNorms in recurrence path. Already known fix. |

---

## 7. Open questions

1. **Stage 1 base?** **Decided: DS-Coder-V2-Lite-Instruct.** See `BASE_MODEL_ANALYSIS.md`. Headline: HumanEval 81 % base, target 86 % at T=8.
2. **Stage 2 base?** **Decided: Gemma 4 98e v3** (gated on Stage 1).
3. **Single-layer vs 2-layer recurrent block?** OpenMythos uses single. Test single first; revisit 2-layer in phase 6 if needed.
4. **Shared router vs T independent routers?** Default: shared router with depth-LoRA conditioned on depth embedding. Cheaper.
5. **Halting strategy for v0?** Decided: fixed T. (ACT in phase 7.)
6. **Compute target for Stage 1?** TBD — user choice between 4× 3090 on solidpc (free, ~25 days) vs 2× H100 vast.ai pod (~$600, ~11 days).
7. **Vendor DeepSeek modeling code or import via `trust_remote_code`?** Default: import for v0; vendor before publishing.
8. **License of derivative?** OpenMythos MIT, Gemma 4 Gemma terms, DeepSeek model agreement (commercial OK). Stage 1 release uses DeepSeek model agreement + OpenMythos attribution. Stage 2 uses Gemma terms + OpenMythos attribution.

---

## 8. Non-goals (explicitly out of scope for v0)

- From-scratch training.
- ACT halting.
- GGUF / llama.cpp support (PyTorch only at first).
- Beating Claude Mythos in absolute terms — we're proving the *technique* works.
- MLA conversion on Gemma 4 (deferred).
- Long-context (>16k) fine-tuning in v0; native 128k is for downstream apps.

---

## 9. References

- OpenMythos: https://github.com/kyegomez/OpenMythos (MIT)
- "Teaching Pretrained LMs to Think Deeper with Retrofitted Recurrence" — arXiv 2511.07384
- "Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization" — arXiv 2603.21676
- LayerScale: Touvron et al., "Going deeper with Image Transformers" (CaiT, 2021)
- Parcae: Prairie et al., 2026 — spectral radius / LTI stability
- DeepSeek-V2 paper: arXiv 2405.04434
- DeepSeek-Coder-V2 paper: https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/paper.pdf
- Gemma 4: `google/gemma-4-26B-A4B-it`
- Parent project's expert-drop work: `../scripts/expert_neuron_v4.json`, `../google/gemma-4-A4B-109e/`

---

## 10. Suggested first-session tasks (for the new session that picks this up)

1. Read this `MASTER_PLAN.md` end to end.
2. Read `CLAUDE.md` for environment / convention rules.
3. Read `BASE_DEEPSEEK_CODER_V2_LITE.md` for Stage 1 architecture details.
4. Skim DeepSeek-Coder-V2's `modeling_deepseek.py` (~78 KB) to understand MLA + DeepseekMoE block structure.
5. Skim OpenMythos's `open_mythos/main.py` and `open_mythos/recurrence.py` — copy LTI injection class skeleton.
6. Skim arXiv 2511.07384 PDF for the curriculum recipe.
7. `huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --local-dir base/DeepSeek-Coder-V2-Lite-Instruct --local-dir-use-symlinks False`.
8. Create `pyproject.toml` + `src/mythic_rdt/__init__.py`.
9. Write `src/mythic_rdt/configuration.py` (extends DeepseekV2Config).
10. Write a minimal `src/mythic_rdt/modeling.py` that loads DS-Coder-V2-Lite-Instruct and runs at T=1 with gate=0 (identity passthrough).
11. **Phase 0 sanity gate**: T=1 output must be bit-exact with running the chosen middle layer once. Don't proceed without this.

---

## Open log (append as we go)

- **2026-04-26** — Project kickoff as Mythic-Gemma4. Skeleton folder + CLAUDE.md + MASTER_PLAN.md + BASE_MODEL_ANALYSIS.md created.
- **2026-04-26** — Renamed Mythic-Gemma4 → Mythic-RDT (multi-base scope). Stage 1 = DeepSeek-Coder-V2-Lite-Instruct (replaces earlier DS-V2-Lite-Chat plan after user pointed out the Coder variant gives a stronger HumanEval headline). Stage 2 = Gemma 4 98e v3.
