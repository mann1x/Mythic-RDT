# Mythic-RDT Stage 2 Base — Gemma 4 26B-A4B 98e v3

Reference for Stage 2 of Mythic-RDT. Used **only after** Stage 1 (DS-V2-Lite) demonstrates the retrofit-recurrence technique works on a real MoE base. Stage 1 details: `BASE_DEEPSEEK_V2_LITE.md`.

## Source

- HF model: [`ManniX-ITA/gemma-4-A4B-98e-v3-it`](https://huggingface.co/ManniX-ITA/gemma-4-A4B-98e-v3-it) (our published artifact)
- Built from: `google/gemma-4-26B-A4B-it` (128e, 30 layers) by dropping the 30 lowest-contributing experts per layer
- Local intermediate (NOT bit-identical to HF — see SHA audit below): `../google/gemma-4-A4B-98e-hybrid/`

## High-level

| Field | Value |
|---|---|
| Total params | 20.8 B |
| Active params/token | 4.0 B |
| Hidden dim | 2816 |
| Layers | 30 |
| Experts/layer | 98 (was 128, dropped 30 lowest-contribution) |
| Top-k routing | 8 |
| Vocab | ~262k |
| Native context | 32k (Gemma 4 official) |
| Attention | GQA, head_dim=256, global_head_dim=512, `attention_k_eq_v=true` |
| Shared experts | **none** (Gemma 4 doesn't have them) |
| Dtype | bfloat16 |
| Disk size | 39 GB |
| **GPQA Diamond** | **75.25%** (matches 128e original) |

## Why DS-V2-Lite first, Gemma 4 98e v3 later

| Aspect | DS-V2-Lite (Stage 1) | Gemma 4 98e v3 (Stage 2) |
|---|---|---|
| Architectural fit for OpenMythos | ✅ native (MLA + shared experts) | ❌ requires bolt-on / scope reduction |
| Fine-tune cost (5 B tokens) | ~$600 / 11 days on 2× H100 | ~$1500 / 22 days on 2× H100 |
| Base GPQA Diamond | ~28% | 75.25% |
| Recurrence headroom | **large** (28 → 40+ plausible) | small (75 → 78 plausible) |
| Risk if technique fails | low (cheap, fast) | high (expensive, slow) |

Stage 1 is the **cheap technique-validation** with publishable artifact. Stage 2 is the **high-quality release** justified by Stage 1 success. If Stage 1 fails, Stage 2 is cancelled — no expensive Gemma 4 GPU time wasted.

## Architectural mismatch with OpenMythos

| OpenMythos component | Gemma 4 98e v3 status | Strategy |
|---|---|---|
| MLA (Multi-Latent Attention) | **MISSING** — Gemma 4 uses GQA | Keep GQA. MLA conversion needs pretrain budget we don't have. |
| Shared experts | **MISSING** — Gemma 4 has only routed experts | Skip. Don't try to add shared experts; would change param count and require re-pretraining the routing dynamics. |
| 32k context | ≅ DS-V2-Lite | OK |
| MoE topology | 98 routed, top-8 vs DS's 64 routed + 2 shared, top-6 | Different but workable; depth-LoRA handles the routing differences. |
| RoPE | similar to DS | OK |
| RMSNorm | similar | **fp32 in recurrence path** (parent project bf16 NaN bug) |

The Stage-2 Mythic on Gemma is therefore "OpenMythos-inspired" rather than "OpenMythos-direct". The retrofit-recurrence machinery (LTI + gates + LayerScale + depth-LoRA) still applies; the OpenMythos-specific MLA/shared-experts pieces are dropped.

## Reuse potential from Stage 1

If Stage 1 (DS-V2-Lite) succeeds, the Stage-2 conversion reuses:
- **Recurrence harness code** (LTI, gates, LayerScale, depth-LoRA, curriculum loop) — verbatim, just point at the Gemma 4 layer instead.
- **Curriculum recipe** (T=2 → 4 → 8 → 16 with mixed-T sampling) — same.
- **Eval scripts** (mostly; Gemma needs `--reasoning-format deepseek --reasoning-budget 8192` re-added to llama-server invocation).
- **HF custom-code packaging pattern** — `MythicRDTConfig` / `MythicRDTForCausalLM` already established; just register a Gemma-flavored subclass.

The only Gemma-specific work is:
- Pick `recurrent_layer_idx` from `../scripts/expert_neuron_v4.json` (avoid bf16-broken middle layers).
- Re-add `--reasoning-format` flags to eval scripts.
- Address the chat-template gap (HF 98e-v3-it template is 4 KB larger than the local intermediate — see SHA audit below).

## Local ↔ HF integrity audit (preserved from earlier analysis)

The local `../google/gemma-4-A4B-98e-hybrid/` intermediate is **not bit-identical** to the published HF artifact. All 9 safetensors shards have identical sizes but **different SHA256 hashes**, and three metadata files differ in size:

| File | HF size | Local size | Notes |
|---|---|---|---|
| chat_template.jinja | 16,448 | 12,045 | **HF has a newer/larger template** |
| tokenizer_config.json | 2,095 | 2,068 | minor |
| expert_drop_metadata.json | 40,247 | 40,266 | minor |

Cause: parent project task #35 ("rebuild + publish on pod") re-saved through a different code path. Same logical model, different bytes.

**Stage 2 must use the HF version**, not the local intermediate:

```bash
huggingface-cli download ManniX-ITA/gemma-4-A4B-98e-v3-it \
    --local-dir base/gemma-4-A4B-98e-v3-it \
    --local-dir-use-symlinks False
```

Verify all 9 shards match these SHA256s after download:

```
shard 1: abac529478c80b02f8a9fd1cfb23df0a6d3014293f626f32c517c9a2466e05bc
shard 2: d96d8f6158ede8c5f8f83523d39eee9ad7a3e1704ec86eb8ff8b3ca663c1f9bc
shard 3: fa5cd1cef1b492d29b41e7d700ff96b3f68376c36fc04e0691def3a41fbd221e
shard 4: 96ae172125579b372a1ad5b1f49f1335e41eee9212e55e6dd49d3b73ec717709
shard 5: 8205b4bb448e497abfce41139da87e43f5ea60e1bba725ce0fda9098d53aa8f2
shard 6: 9deef0ddfb231d3abeadbc2bbfdf77a0baa7ab08dd7908838619c2fc72726583
shard 7: 8372534c8927bb15603ab573fcabe510f2f20bbc853ab17485b7158e9c790f32
shard 8: 9cb02c1f28b2075c1967c1d37fe9d474547072a5c381022d9086940c956b4748
shard 9: 45a63287acf16d1bcd673586382aa653ee9ceaf1b8f21962ed19b59e5666e770
```

Keep the local intermediate as backup; do not delete or overwrite.

## Memory budget for fine-tune (Stage 2)

| Setup | bf16 base | Activations (T=8 ckpt, batch 1) | Trainable+opt | **Total VRAM** | Fits on |
|---|---|---|---|---|---|
| Naive bf16 | 41 GB | ~250 MB | ~50 MB | **~42 GB** | 1× A100/H100 80GB ✅; 48GB tight |
| 4-bit base (bnb NF4) | 11 GB | ~250 MB | ~50 MB | **~12 GB** | 1× 3090 ✅; 1× 4090 ✅ |
| 4-bit base + DDP × 4× 3090 | 11 GB / GPU | ~1 GB | ~50 MB | ~13 GB / GPU | **solidpc 4× 3090 ✅** |

GQA's larger KV cache means fewer prompts per batch than DS-V2-Lite at the same VRAM. Plan for batch=1 and gradient accumulation.

## Throughput estimate (Stage 2 fine-tune)

Effective compute at T=8: 2 prelude + 8 × recurrent + 2 coda = 12 effective layers at A4B (4 B active) → ~24 GFLOPs/token forward, ~72 GFLOPs/token train.

| GPUs | tok/s training | tok/day | 5 B tokens (release) |
|---|---|---|---|
| 4× 3090 DDP — 4-bit base | 900–1500 | ~100 M | ~50 days |
| 2× H100 vast.ai | 2000–3000 | ~220 M | ~22 days, ~$1500 |
| 4× H100 vast.ai | 4000–6000 | ~430 M | **~11 days, ~$2000** |

Stage 2 release target: 5 B tokens on 4× H100, ~$2000.

## Stage-2 success criteria

- T=1 GPQA Diamond ≥ 73% (within 2 pp of base 75.25%).
- T=8 GPQA Diamond ≥ 78% (3+ pp lift over base).
- T=16 GPQA Diamond ≥ T=8.
- T=1 HumanEval / MMLU within 2 pp of base.
- No mode collapse, no chat-template corruption, no `<|channel>` token corruption.

If those hold: publish as `ManniX-ITA/Mythic-RDT-Gemma4-26B-A4B-98e`. Otherwise document and archive.
