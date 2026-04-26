# Mythic-Gemma4 — Base Model Analysis

Decision: **which Gemma 4 variant becomes the foundation for the Mythic-Gemma4 RDT?**

## Candidates on this server

| Variant                          | Local path                                        | Storage (bf16) | Layers | Experts/layer | GPQA Diamond (Q6_K, full) | Notes |
|----------------------------------|---------------------------------------------------|----------------|--------|---------------|---------------------------|-------|
| **gemma-4-26B-A4B-it (128e)**    | `../google/gemma-4-26B-A4B-it/`                   | 118 GB*        | 30     | 128           | **75.25%**                | original; 118 GB incl. multimodal heads, text-only ~52 GB |
| **gemma-4-A4B-98e-hybrid (98e v3)** | `../google/gemma-4-A4B-98e-hybrid/`            | 39 GB          | 30     | 98            | **75.25%**                | published as `ManniX-ITA/gemma-4-A4B-98e-v3-it`; **same score as 128e**, 23.4% MoE capacity removed |
| gemma-4-A4B-109e (drop)          | `../google/gemma-4-A4B-109e/`                     | 42 GB          | 30     | 109           | 71.72%                    | first published HF model; -3.5 pp vs original |
| gemma-4-A4B-109e-v3              | `../google/gemma-4-A4B-109e-v3/`                  | 42 GB          | 30     | 109           | 71.72%                    | clean teacher-force map version of 109e |
| gemma-4-A4B-120e-hybrid          | `../google/gemma-4-A4B-120e-hybrid/`              | 46 GB          | 30     | 120           | unknown                   | larger hybrid drop, less aggressive |
| gemma-4-A4B-64e                  | `../google/gemma-4-A4B-64e/`                      | ~30 GB         | 30     | 64            | 0.00% (broken)            | aggressive drop, broke the model |
| gemma-4-A4B-96e                  | `../google/gemma-4-A4B-96e/`                      | ~38 GB         | 30     | 96            | unknown                   | early sibling to 98e, superseded |
| gemma-4-16B-it                   | `../google/gemma-4-16B-it/`                       | 32 GB          | ?      | dense (no MoE)| n/a                       | smaller dense Gemma 4 variant |
| gemma-4-31B-it                   | `../google/gemma-4-31B-it/`                       | 189 GB*        | ?      | ? (likely MoE)| unknown                   | larger Gemma 4 |
| Gemma 4 E4B                      | (not on disk)                                     | small          | ?      | dense         | 57.07%                    | Google's small variant; too weak |

*sizes include multimodal encoders for the official releases.

Architecture for all A4B variants (verified from `text_config`):
- `hidden_size: 2816`, `intermediate_size: 2112`
- `head_dim: 256`, `global_head_dim: 512`, `attention_k_eq_v: true` (K and V share weights)
- `num_hidden_layers: 30`, `layer_types`: mixed `sliding_attention` / `full_attention`
- `dtype: bfloat16`, `final_logit_softcapping: 30.0`, `enable_moe_block: true`

---

## Comparative analysis

### 128e (original)

**Pros**
- Reference model, no surprises.
- Highest absolute headroom — every uncompressed expert is available for the recurrent block to draw on.
- Most prior art: every published Gemma 4 result, eval methodology, and llama-server config has been validated on this exact base.

**Cons**
- 118 GB on disk, ~52 GB text-only after stripping multimodal heads. Fine-tune storage cost ≈ 1.5–2× larger than 98e-v3 throughout the curriculum.
- Full 128 experts × 30 layers means the recurrent layer carries the full MoE block even though we only reuse one layer's experts. Wasted disk + memory budget for capacity we're not using.
- No quality margin to recover via recurrence — must out-perform the 75.25% baseline purely from compute scaling.

### 98e-hybrid (= "98e v3", primary candidate)

**Pros**
- **Same 75.25% GPQA Diamond as 128e**, despite removing 23.4% of MoE capacity. This is a documented Pareto improvement, not theoretical.
- Smaller storage (~39 GB on disk, ~20.8B params bf16) → cheaper fine-tune memory, faster shard loads, smaller HF release.
- Already pruned with the broader `expert_neuron_v4.json` analysis (all task categories, not GPQA-specific) — generalization is preserved.
- Published and verified: `ManniX-ITA/gemma-4-A4B-98e-v3-it`. The community can reproduce.
- Owned weights: no download cost, instant fine-tune start.
- Smaller expert population per layer (98 vs 128) means depth-distinct router LoRA has fewer outputs to coordinate — slightly easier learning target.
- The hybrid drop kept different expert subsets per layer, so the chosen recurrent layer's 98 experts are already a "useful 98", not a worst-case subset.

**Cons**
- The recurrent block reuses one layer's MoE T times. That layer has 98 experts now, instead of 128. If recurrence happens to stress capacity, 30 missing experts could matter more than they did at T=1.
- Pruning was performed under a single-shot inference assumption. Whether the dropped experts would have been useful under T-loop reuse is untested — there's a small chance a pruned expert is exactly the "deep reasoning" expert we'd want available at T=8.
- Less margin: if Mythic-98e fine-tune underperforms, the floor is closer.

### 109e v3 (drop variant)

**Pros**
- Larger expert population than 98e (109 vs 98).
- Already published, with a clean teacher-force map.

**Cons**
- **Loses 3.5 pp on GPQA vs both 128e and 98e-v3** (71.72%). Starting from a degraded baseline is worse than starting from 98e-v3 which has zero quality loss.
- No reason to choose this when 98e-v3 dominates on both quality and storage.

### 120e-hybrid

**Pros**
- Less aggressive drop, closer to 128e capacity.

**Cons**
- GPQA score never logged in `Reference results`. Without a verified score it's a leap of faith.
- Storage is between 98e and 128e with unclear quality position. Probably no advantage over either.
- Skip unless 98e-v3 fine-tune fails and we need a "more capacity" fallback.

### 16B / 31B / E4B

- **16B-it (dense)**: no MoE — wrong architectural family, defeats the A4B-MoE design. Skip.
- **31B-it**: bigger than 26B, but no GPQA score and harder to fine-tune. Skip unless 26B variants all fail.
- **E4B**: 57% GPQA is too low — even with T=8 recurrence we'd struggle to hit 70%. Skip.

---

## Recommendation

**Primary base: `gemma-4-A4B-98e-hybrid` (published as `ManniX-ITA/gemma-4-A4B-98e-v3-it`).**

Reasoning:

1. **Zero quality loss.** GPQA Diamond 75.25% matches the 128e original — there's no "degraded base" risk. The whole conversation about pruned-base quality recovery becomes moot.
2. **Smaller fine-tune target.** 39 GB on disk vs 118 GB (or 52 GB text-only) — every gradient step is cheaper, every checkpoint write is faster, every HF release is more downloadable.
3. **The pruning is orthogonal to recurrence.** 98e v3 removed the lowest-contribution experts under single-shot inference. Recurrence adds compute scaling. These are independent levers; combining them preserves both wins.
4. **The story is interesting:** "Mythic-Gemma4-26B-A4B-98e — 20.8B params, matches Gemma 4 base at T=1, surpasses it at T=4–8 on hard reasoning, no extra storage." That's a tighter narrative than the 128e variant.
5. **We own the weights and the pipeline.** Already on disk, already evaluated, already published — instant phase-0 start.

**Fallback base: `gemma-4-26B-A4B-it` (128e original).**

Use 128e if:
- Phase-0 sanity gate fails on 98e-v3 (untrained T=8 produces total junk).
- Phase-2 fine-tune curriculum cannot recover quality on 98e-v3 even after 3+ B tokens.
- We discover that pruned experts at the chosen recurrent layer are critical for T-loop reuse.

In that fallback, all the architectural decisions (recurrent layer index, LTI init, gate init, depth-LoRA rank, curriculum) are identical — only the base swap.

**Explicitly NOT used:** 109e, 120e, 64e, 96e, 16B, 31B, E4B, 31B. Each fails on either quality, architecture mismatch, or both.

---

## Local ↔ HF integrity check — RESULT: **NOT BIT-IDENTICAL**

Local `../google/gemma-4-A4B-98e-hybrid/` is **not** the same artifact as the published `ManniX-ITA/gemma-4-A4B-98e-v3-it`. File **sizes** match for every weight shard, but the **SHA256 hashes differ on all 9 shards**, and three metadata files differ in size as well.

### Weight shards (sizes match, content differs)

| File | HF size | Local size | HF SHA256 | Local SHA256 | Match |
|---|---|---|---|---|---|
| model-00001 | 5,296,424,138 | 5,296,424,138 | `abac5294…66e05bc` | `f7223f63…18846026` | ❌ |
| model-00002 | 5,077,280,712 | 5,077,280,712 | `d96d8f61…3c1f9bc` | `c00dfae2…01d1acb` | ❌ |
| model-00003 | 5,077,280,712 | 5,077,280,712 | `fa5cd1ce…fbd221e` | `7f53bbc4…0a06583e` | ❌ |
| model-00004 | 5,084,721,096 | 5,084,721,096 | `96ae1721…ec717709` | `ce43c969…17eee6a4` | ❌ |
| model-00005 | 4,688,717,600 | 4,688,717,600 | `8205b4bb…d53aa8f2` | `80cefe95…d76e90cec` | ❌ |
| model-00006 | 4,688,717,592 | 4,688,717,592 | `9deef0dd…2726583` | `6f350918…07673e6` | ❌ |
| model-00007 | 5,077,280,592 | 5,077,280,592 | `8372534c…e9c790f32` | `52fc9242…f58d86412f` | ❌ |
| model-00008 | 5,025,251,546 | 5,025,251,546 | `9cb02c1f…956b4748` | `0b3cf270…1e6670f0` | ❌ |
| model-00009 | 885,957,616 | 885,957,616 | `45a63287…666e770` | `7b52fe18…fe5994f` | ❌ |

### Metadata files (sizes also differ on 3)

| File | HF size | Local size | Δ | Likely meaning |
|---|---|---|---|---|
| chat_template.jinja | **16,448** | **12,045** | **+4,403 on HF** | HF has an updated chat template (large delta — likely added/changed sections) |
| tokenizer_config.json | **2,095** | **2,068** | +27 on HF | small config tweak |
| expert_drop_metadata.json | 40,247 | 40,266 | -19 on HF | minor reorder/timestamp diff |
| model.safetensors.index.json | 103,158 | 103,158 | 0 | likely identical |
| config.json | 3,813 | 3,813 | 0 | likely identical |
| generation_config.json | 208 | 208 | 0 | likely identical |
| processor_config.json | 1,689 | 1,689 | 0 | likely identical |
| tokenizer.json | 32,169,626 | 32,169,626 | 0 | likely identical (vocab) |

### Files present on HF but missing locally

`.gitattributes`, `README.md`, `expert_drop.py` — repo-metadata only, but `expert_drop.py` is the methodology code we should preserve as project artifact.

### What the difference probably means

The **same logical model** was rebuilt and re-saved during the publish step (parent project task #35 was "98e v3: rebuild HF weights + full GGUF pack + publish on pod"). Any of the following will produce identical sizes but different SHA256:

- Different `safetensors` save order or metadata block.
- Slightly different per-tensor dtype path (e.g. cast through fp32→bf16 vs direct bf16).
- Different `expert_drop.py` revision producing the same shapes but different float bytes (e.g. router-row reordering after expert pruning).
- Different `safe_serialization` framework version writing the metadata block.

The `chat_template.jinja` size delta is the most concrete user-facing difference — the published model has a newer template (likely the corrected reasoning-format template used in the eval that produced 75.25%).

### Recommendation: use the canonical HF version, not the local copy

Fetch the published artifact fresh into the project folder:

```bash
mkdir -p Mythic-Gemma4/base
huggingface-cli download ManniX-ITA/gemma-4-A4B-98e-v3-it \
    --local-dir Mythic-Gemma4/base/gemma-4-A4B-98e-v3-it \
    --local-dir-use-symlinks False
```

Then verify:

```bash
cd Mythic-Gemma4/base/gemma-4-A4B-98e-v3-it
sha256sum model-*.safetensors > LOCAL_SHA256
# compare to the table above — must match exactly
```

This is the artifact that produces the documented 75.25% GPQA. Mythic-Gemma4 should build on the published version, not the unpublished intermediate.

**Keep the local `gemma-4-A4B-98e-hybrid/` as-is** — don't delete or overwrite. It's a useful backup and the source of the published rebuild.

### Investigation deferred (low priority)

Why does the local intermediate differ from the published rebuild? Worth a 30-min look at some point, but **not** blocking Mythic-Gemma4. The published model is canonical.

## Open log

- **2026-04-26** — Decision: primary base = 98e-v3 (`gemma-4-A4B-98e-hybrid` locally). Fallback = 128e original. SHA256 verification pending (background hash job).
