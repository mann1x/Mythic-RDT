# Mythic-Gemma4 — Master Plan

**Status**: kickoff (2026-04-26).
**Author**: ManniX-ITA + Claude.
**Goal**: convert Gemma 4 26B-A4B (MoE, top-8 routing) into a Recurrent-Depth Transformer following the OpenMythos blueprint, using fine-tuning only — no from-scratch pretraining — and ship as a `transformers`-loadable model.

**Base model**: **`ManniX-ITA/gemma-4-A4B-98e-v3-it`** (downloaded fresh from HF — see `BASE_MODEL_ANALYSIS.md`). 30 layers, 98 experts/layer, top-8, 20.8 B params, GPQA Diamond 75.25% (matches 128e original). The local `../google/gemma-4-A4B-98e-hybrid/` intermediate is **not bit-identical** to the published artifact; we use the published one. Fallback: `gemma-4-26B-A4B-it` (128e).

**Method**: OpenMythos as the reference for the recurrence machinery (LTI injection, depth-LoRA, MoE-aware loop), retrofit-recurrence curriculum (arXiv 2511.07384) for the fine-tune procedure.

---

## 1. Vision

A single Mythic-Gemma4 26B-A4B checkpoint where the user picks `n_loops` at inference time:

- `n_loops=1` ≈ Gemma 4 baseline cost.
- `n_loops=4–8` ≈ stronger reasoning, ~4–8× compute, no extra storage.
- `n_loops=16` ≈ deepest reasoning mode, depth-extrapolated beyond training.

Compute scales with depth, **storage stays at 26B**, and the same MoE A4B = 4B active per token property is preserved. End-users load via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and call `model.generate(..., n_loops=8)`.

---

## 2. Feasibility verdict

**Feasible — with caveats.**

### Why it's feasible

- The retrofit-recurrence paper (arXiv 2511.07384, "Teaching Pretrained LMs to Think Deeper with Retrofitted Recurrence") demonstrates the exact technique we need: take a pretrained transformer, freeze most of it, add LTI injection + identity-biased gating + depth-LoRA, fine-tune with a curriculum of increasing recurrence depth. They show preservation of base performance plus better math at matched compute — on GPT-2 / OPT / Llama scale.
- OpenMythos is permissively licensed (MIT) — we can lift the LTI injection, depth-LoRA, and gating modules verbatim.
- Gemma 4's structure is RDT-compatible: pre-norm, residual, GQA, MoE FFN — drop-in compatible with a recurrent-block harness.
- 26B-A4B's compute profile is *especially* well suited to RDT: each loop activates 4B params, so T=8 ≈ 32B FLOPs at one-time cost = 26B storage.

### Why it's hard

- Gemma 4's middle layer was trained to be applied **once at one depth position**, not **T times in a loop**. Without retraining, the experts will drift / loop / mode-collapse. Fine-tune *must* recover this.
- 128-expert MoE with depth-distinct routing is an unsolved engineering problem: do we share the router across loop iterations (with a depth-LoRA), or duplicate routers per loop step (T× routers)? Neither is in the OpenMythos paper at this scale.
- Gemma 4's reasoning behavior depends on `--reasoning-format deepseek --reasoning-budget 8192` at inference — we need to verify the recurrent variant doesn't break the channel-token emission.
- LTI stability with A/B injection at 26B scale and T=16 has no public precedent. Spectral radius constraint helps, but bf16 norm bug (parent project, layers 11-29 produce NaN/inf in bf16 norm) means we need fp32 norms in the recurrence path.

### Verdict

Feasible at **academic-grade single-checkpoint release** with a fine-tune budget of ~5–20B tokens on ~2× H100 / 4× 3090 over 2–4 weeks. **Not** feasible if the user expects to surpass Claude Mythos in absolute terms — that needs petabyte-scale pretraining we don't have.

The realistic outcome is: "Gemma 4 26B-A4B at T=1, +5–15% on hard reasoning at T=8, no storage cost increase, custom-code HF release."

---

## 3. Port matrix — what comes from where

| Component                          | Source                             | Strategy                                                        | Risk |
|------------------------------------|------------------------------------|-----------------------------------------------------------------|------|
| Token embedding + lm_head          | Gemma 4                            | Reuse verbatim, frozen                                          | low |
| Prelude layers (1–2)               | Gemma 4 layers 0–1                 | Reuse verbatim, frozen first phase                              | low |
| Recurrent block (1 layer)          | Gemma 4 middle layer (~layer 20)   | Reuse weights, train depth-LoRA + LTI on top                    | **high** — main learning target |
| Coda layers (1–2)                  | Gemma 4 last 1–2 layers            | Reuse verbatim, frozen first phase                              | low |
| GQA attention                      | Gemma 4                            | Reuse verbatim                                                  | low |
| MoE experts (128, top-8)           | Gemma 4                            | Reuse verbatim, frozen                                          | low |
| MoE router                         | Gemma 4 + depth-LoRA               | Add T independent rank-8 LoRAs (or 1 shared LoRA with depth emb) | med |
| RMSNorm / rotary                   | Gemma 4                            | Reuse verbatim, **upgrade norms to fp32 in recurrence path**    | low |
| LTI injection (A, B)               | OpenMythos (Parcae)                | New params, init `A = Diag(-exp(log_A))`, `B ≈ 0`, learnable    | med |
| Identity-biased gating             | Retrofit-recurrence paper          | New per-loop scalar gate, init to ~0 (sigmoid bias = -3)        | med |
| Per-loop LayerScale                | LayerScale literature              | Init 1e-4, learnable                                            | low |
| Depth embedding (T position)       | New                                | Sinusoidal or learned, fed into router LoRA                     | low |
| MLA attention                      | OpenMythos                         | **NOT PORTED** — keep Gemma 4 GQA                               | n/a |
| ACT halting head                   | OpenMythos                         | **DEFERRED** to phase 2 — first release uses fixed T            | n/a |
| Tokenizer                          | Gemma 4                            | Reuse verbatim                                                  | low |
| Chat template + reasoning format   | Gemma 4                            | Reuse, verify it survives recurrence                            | med |

**The single learnable hot spot is the recurrent block + LTI/gating/router-LoRA.** Total new trainable parameters: ~50–200M (depending on rank choice). Frozen: 25.8B+.

---

## 4. Architecture spec — Mythic-Gemma4 v0

```
input_ids → embed → [Prelude: Gemma4 layer 0,1 frozen]
                → e (encoded input, kept around)
                → h_0 = Prelude(embed)
                → for t in 0..T-1:
                    inj = A·h_t + B·e                   # LTI injection (A, B learnable)
                    block_out = RecurrentBlock(h_t, e)  # Gemma4 middle layer w/ depth-LoRA
                    g = sigmoid(gate_t)                  # identity-biased gate, init ≈ 0
                    ls = layerscale_t                   # init 1e-4
                    h_{t+1} = h_t + ls · g · (inj + block_out)
                → [Coda: Gemma4 last 2 layers frozen]
                → norm + lm_head → logits
```

Key constants for first build:

- `prelude_layers = 2`
- `coda_layers = 2`
- `recurrent_layer_idx = floor(L/2)` where L = total Gemma 4 layers (probably 20 of 40)
- `train_loop_iters = 8` (curriculum below)
- `max_loop_iters = 16` (inference cap, depth-extrapolatable)
- `depth_lora_rank = 8` for Q/K/V/O; rank 16 for router LoRA
- `lti_init`: log_A ~ Uniform(0.01, 0.1), B = zeros + tiny noise
- `gate_init`: bias = -3 (sigmoid ≈ 0.047 at start)
- `layerscale_init`: 1e-4
- `halting_strategy = "fixed"` (ACT later)

**Note on which middle layer:** the parent project's expert-drop work (`../scripts/expert_neuron_v4.json`) has per-layer contribution data for Gemma 4 — pick a layer that's "middle" and "boring" (not one of the layers 11-29 with bf16 NaN issues). Phase 0 task: read that data and pick.

---

## 5. Phased roadmap

### Phase 0 — Feasibility prototype (1 week)

**Goal**: prove the wrapper runs and doesn't immediately diverge.

- [ ] **Fetch canonical base from HF**: `huggingface-cli download ManniX-ITA/gemma-4-A4B-98e-v3-it --local-dir base/gemma-4-A4B-98e-v3-it --local-dir-use-symlinks False`. Verify SHA256s match the table in `BASE_MODEL_ANALYSIS.md`. Do NOT use the local `../google/gemma-4-A4B-98e-hybrid/` intermediate (different bytes).
- [ ] Create the empty Python package skeleton (`src/mythic_gemma4/`, `pyproject.toml`).
- [ ] Init local git, push to GitHub repo `Mythic-Gemma4` once skeleton compiles. **Defer until user explicitly asks** — do not push without confirmation.
- [ ] Pick `recurrent_layer_idx` from `../scripts/expert_neuron_v4.json` (avoid bf16-broken layers).
- [ ] Build minimal `MythicGemma4Config` extending `Gemma4Config`.
- [ ] Build `MythicGemma4ForCausalLM` that loads Gemma 4 weights and adds LTI/gating/LayerScale (all init values that make T=1 *exactly* equivalent to passing through the chosen layer once).
- [ ] **Sanity check**: at T=1 with `gate=0`, output must match running Gemma 4's middle layer once on the prelude output. Bit-exact in fp32.
- [ ] Run a short forward pass at T=4, T=8, T=16 — record perplexity drift, output sanity (no infs/NaNs, no mode collapse on 100 prompts from FineWeb-Edu sample).
- [ ] Decision gate: if T=8 untrained drops > 50% PPL or produces gibberish on > 20% of prompts, the architecture choice (which middle layer, what gate init) needs revisiting before phase 1.

### Phase 1 — Surgery and conversion (1 week)

- [ ] Conversion script `scripts/convert_gemma4_to_mythic.py base_model_path output_path` that:
  1. Loads Gemma 4 26B-A4B weights via `safetensors`.
  2. Strips middle layers (keeps prelude + 1 recurrent + coda).
  3. Initializes LTI / gating / LayerScale / depth-LoRA to the safe defaults.
  4. Saves as a Mythic-Gemma4 checkpoint.
- [ ] Verify: loaded model runs `model.generate(...)` at T=1, T=4, T=8 without crashes.
- [ ] Verify: `safetensors` shards don't exceed 5 GB each.
- [ ] First eval: GPQA Diamond `--limit 20` at T=1 vs Gemma 4 base — should be near-parity (~70-75%) since at T=1 we're effectively a 4-layer Gemma slice. If wildly off, the surgery is wrong.

### Phase 2 — Curriculum fine-tune (2–4 weeks, longest phase)

- [ ] Decide compute target: 2× H100 (rented Vast.ai pod) for ~5B tokens, or 4× 3090 (solidpc + remote) for ~2B tokens. Cost estimate: $300–$1500.
- [ ] Dataset: FineWeb-Edu 10B sample + a slice of OpenHermes / Magpie for instruction-following preservation. Tokenizer must match Gemma 4. Target ratio 80/20 web/instruct.
- [ ] Curriculum (token splits approximate):
  1. **Warm-up** (10% of budget): freeze everything except LayerScale + gates. Train at T=2. Goal: gates open from 0 to nonzero without diverging.
  2. **Recurrence-2** (20%): unfreeze A, B, depth-LoRA[t=0,1]. Train at T=2, mixed with T=1 forward sometimes.
  3. **Recurrence-4** (25%): T=2,4 mixed.
  4. **Recurrence-8** (30%): T=4,8 mixed.
  5. **Anneal to target T=16** (15%): T=8,16 mixed.
- [ ] Loss: standard CE on next-token + small auxiliary KL between T=1 output and original Gemma 4 output (on a held-out batch) to discourage T=1 drift.
- [ ] Validation: held-out FineWeb-Edu PPL + GPQA `--limit 20` at every checkpoint.
- [ ] Stop criterion: T=8 GPQA-20 ≥ Gemma 4 base, and T=1 PPL within 1% of Gemma 4 baseline PPL on FineWeb-Edu val.

### Phase 3 — Full eval + iteration (1 week)

- [ ] Full GPQA Diamond (198q) at T=1, T=4, T=8, T=16. Reference target: Gemma 4 base = 75.25%.
- [ ] HumanEval pass@1 at T=1, T=8.
- [ ] MMLU-Pro subset.
- [ ] Sanity: sample 100 generations, check for repetition / mode collapse / channel-token corruption.
- [ ] If any T<8 result is below base by >2%, root-cause and re-curriculum.

### Phase 4 — Custom-code transformers package (1 week)

- [ ] Move modeling code to root: `configuration_mythic_gemma4.py`, `modeling_mythic_gemma4.py` (HF custom-code convention).
- [ ] Add `auto_map` to config so `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` works.
- [ ] Add `MythicGemma4Config.register_for_auto_class()` and `MythicGemma4ForCausalLM.register_for_auto_class("AutoModelForCausalLM")` calls in module init.
- [ ] Test load on a fresh machine: `pip install transformers && python -c "..."` works.
- [ ] GGUF export research: llama.cpp does NOT support recurrent depth — document that GGUF is out of scope for v0 (loop must run in PyTorch). Could add later via custom kernel.

### Phase 5 — Release (3 days)

- [ ] HF model repo: `ManniX-ITA/Mythic-Gemma4-26B-A4B` (BF16 + custom code).
- [ ] README with: architecture, how to load, recommended `n_loops` for tasks, eval table, reproducibility section, citations (Gemma 4 + OpenMythos + Parcae + Retrofit-Recurrence).
- [ ] Optional: 4-bit and 8-bit bitsandbytes variants (these work even though GGUF doesn't, since recurrence runs in PyTorch).
- [ ] Blog post / X thread.
- [ ] Issue thread for community feedback.

### Phase 6 — Polish (deferred, post-release)

- ACT halting head — train the halting predictor on a frozen base.
- MLA conversion experiment — small fine-tune to compress KV.
- Apply same recipe to other Gemma 4 sizes if Google releases more.
- Mythic-Gemma4-pruned: combine with the parent project's 109e expert drop → 109-expert RDT.

---

## 6. Risks and mitigations

| Risk                                                                                                                       | Severity | Mitigation |
|----------------------------------------------------------------------------------------------------------------------------|----------|------------|
| One Gemma 4 middle layer reused T=16 times mode-collapses regardless of fine-tune budget                                    | high     | Phase 0 sanity gate. If untrained T=8 is total junk, fall back to a 2-layer recurrent block (more params, less depth scaling) |
| LTI A/B at 26B scale don't stabilize even with spectral constraint                                                         | med      | LayerScale + gate init below 0.05 + curriculum. Worst case: drop LTI, keep gate-only retrofit (`h_{t+1} = h_t + g·Block(h_t)`) |
| Gemma 4 reasoning channel tokens (`<|channel>thought`) corrupt during recurrence                                           | med      | Phase 0 generation sanity. If broken, add channel-aware loss term in fine-tune |
| Fine-tune budget too small to recover quality                                                                              | med      | Start with 1-2B token pilot, project quality curve, scale up only if curve looks healthy |
| `transformers` custom-code path breaks on new transformers releases                                                        | low      | Pin compatible transformers version range in repo. Test against transformers 5.4, 5.5, 5.6. |
| MoE router can't handle depth-distinct routing without rebalancing experts                                                 | med      | Phase 1 sanity: log expert utilization across loop iterations. If <30% experts ever fire, redistribute via an aux load-balance loss |
| bf16 NaN in norms (parent project bug)                                                                                     | low      | Use fp32 in all RMSNorms inside the recurrent block. Already known fix. |

---

## 7. Open questions (resolve before/during phase 0)

1. **Base model choice?** **Decided: `gemma-4-A4B-98e-hybrid` (98e v3) as primary, 128e as fallback.** See `BASE_MODEL_ANALYSIS.md` for full reasoning. Headline: 98e v3 has same GPQA (75.25%) as 128e, 23% less storage, and we own the weights.
2. **Single-layer vs 2-layer recurrent block?** OpenMythos uses single. Some literature uses 2-layer. Test both in phase 0.
3. **Shared router across loops vs T independent routers?** Storage cost of T routers is small (router is small), but T independent routers = 8× more new params to train. Default: shared with depth-LoRA.
4. **Halting strategy for v0?** Decided: fixed T. (ACT in phase 6.)
5. **Compute target?** Pending user decision (vast.ai 2× H100 pod ~$500-800/week vs solidpc + remote 3090s ~free but slower).
6. **License of derivative?** OpenMythos is MIT (permissive), Gemma 4 has Google's Gemma terms (allow derivatives, restrict harmful use). Output: Gemma terms + attribution to OpenMythos. Verify with Google's Gemma license text.

---

## 8. Non-goals (explicitly out of scope for v0)

- From-scratch training.
- MLA attention.
- Training data curation beyond standard FineWeb-Edu + a small instruct mix.
- ACT halting.
- GGUF / llama.cpp support (PyTorch only at first).
- Beating Claude Mythos in absolute terms — we're proving the *technique* works on Gemma 4.

---

## 9. References

- OpenMythos: https://github.com/kyegomez/OpenMythos (MIT)
- "Teaching Pretrained LMs to Think Deeper with Retrofitted Recurrence" — arXiv 2511.07384
- "Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization" — arXiv 2603.21676
- LayerScale: Touvron et al., "Going deeper with Image Transformers" (CaiT, 2021)
- Parcae: Prairie et al., 2026 — spectral radius / LTI stability
- AttnLRP: Achtibat et al., ICML 2024 (relevant if we ever want LRP-aware fine-tune)
- Gemma 4: `google/gemma-4-26B-A4B-it`
- Parent project's expert-drop work for layer choice: `../scripts/expert_neuron_v4.json`, `../google/gemma-4-A4B-109e/`

---

## 10. Suggested first-session tasks (for the new session that picks this up)

1. Read this `MASTER_PLAN.md` end to end.
2. Read `CLAUDE.md` for environment / convention rules.
3. Read `../scripts/expert_neuron_v4.json` summary — pick `recurrent_layer_idx` candidate (a "boring" middle layer, not 11-29 if those have bf16 issues).
4. Read OpenMythos's `open_mythos/main.py` and `open_mythos/recurrence.py` — copy LTI injection class skeleton.
5. Skim arXiv 2511.07384 PDF for the curriculum recipe.
6. Create `pyproject.toml` + `src/mythic_gemma4/__init__.py`.
7. Write `src/mythic_gemma4/configuration.py` (extends Gemma4Config).
8. Write a minimal `src/mythic_gemma4/modeling.py` that loads Gemma 4 26B-A4B and runs at T=1 (no LTI, no gates — just identity passthrough).
9. **Phase 0 sanity gate**: T=1 output must be bit-exact with running the chosen middle layer once. Don't proceed without this.

---

## Open log (append as we go)

- **2026-04-26** — Project kickoff. Skeleton folder + CLAUDE.md + MASTER_PLAN.md created. No code yet.
