# Mythic-RDT Stage 1 Base — DeepSeek-Coder-V2-Lite-Instruct

Reference for Stage 1 of Mythic-RDT. **Mythic-Coder** is the coding-specialized RDT — depth recurrence applied to a code-strong base, where the headline benchmark is HumanEval / MBPP, not GPQA. Stage 2 (Gemma 4 98e v3) is documented in `BASE_GEMMA4_98E_V3.md`.

## Why the Coder variant over plain V2-Lite-Chat

DS-Coder-V2-Lite-Instruct is the same chassis as DS-V2-Lite-Chat (continued pretraining from a V2-Lite intermediate checkpoint with +6 T code tokens), but it's already at **HumanEval ~81 %** baseline — a coding model that holds its own against models 5–10× its size. RDT-amplified coding is a more compelling demonstration than RDT-amplified general reasoning on a small base, because hard programming problems benefit *exactly* from multi-step deliberation, which is what depth recurrence buys.

The architecture is identical to V2-Lite, so all the OpenMythos-fit advantages (native MLA, native shared experts) carry over. Only the training data differs.

## Source

- HF model: [`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)
- Paper: [DeepSeek-Coder-V2 paper](https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/paper.pdf)
- Continued from: DeepSeek-V2 intermediate checkpoint + 6 T additional code tokens (covers 338 programming languages, vs 86 in DeepSeek-Coder V1).
- License: Code MIT, Model = DeepSeek Model Agreement (commercial use allowed; restrictions on harmful use)
- Total disk: **31.4 GB** (4 shards, bf16)

## High-level

| Field | Value |
|---|---|
| Total params | 15.7 B |
| Active params/token | **2.4 B** |
| Hidden dim | 2048 |
| Layers | 27 |
| Vocab | 102,400 |
| **Context length** | **128k** (extended from V2-Lite's 32k via continued pretraining) |
| Dtype on disk | bfloat16 |
| Disk size | 31.4 GB |
| Custom code | yes — `modeling_deepseek.py`, `configuration_deepseek.py`, `tokenization_deepseek_fast.py` (load with `trust_remote_code=True`) |

## Multi-Latent Attention (MLA)

Same MLA spec as V2-Lite. KV cache is ~6× smaller than GQA at long context, which makes 128k-context fine-tune feasible on hardware that can't touch GQA at the same context.

| Field | Value |
|---|---|
| Num attention heads | 16 |
| Per-head dim (V) | 128 |
| Q compression rank | not applied in Lite (only in V2 full) |
| **KV compression rank (`kv_lora_rank`)** | **512** |
| Decoupled-RoPE Q per-head dim (`qk_rope_head_dim`) | 64 |
| Non-RoPE Q/K per-head dim (`qk_nope_head_dim`) | 128 |
| V per-head dim (`v_head_dim`) | 128 |

KV cache footprint per token ≈ 576 floats. At 32k context that's ~2 GB; at 128k it's ~8 GB (still very manageable; GQA equivalent would be ~32 GB).

## DeepSeekMoE (routed + shared)

Same as V2-Lite — the OpenMythos topology, native.

| Field | Value |
|---|---|
| Layer 0 FFN | dense (no MoE) |
| Layers 1–26 FFN | MoE (26 of 27 layers) |
| Routed experts (`n_routed_experts`) | **64** |
| Shared experts (`n_shared_experts`) | **2** (always active) |
| Top-k routing (`num_experts_per_tok`) | **6** routed |
| Expert FFN intermediate dim | 1408 |
| Aux loss coefficient | α₁ = 0.001 |

## Reported benchmarks (DS-Coder-V2-Lite-Instruct)

| Benchmark | Score | Notes |
|---|---|---|
| **HumanEval pass@1** | **81.1 %** | strongest open model in size class |
| HumanEval+ pass@1 | 75.6 % |  |
| MBPP+ pass@1 | 68.8 % |  |
| LiveCodeBench (2024) | ~24 % |  |
| BigCodeBench-Hard | ~18 % |  |
| GSM8K | 86.4 % | math reasoning still strong |
| MATH | 61.8 % |  |
| MMLU | 60.1 % | general knowledge slightly above V2-Lite-Chat |
| GPQA Diamond | not commonly reported (estimate 25–35 %) | not the headline benchmark |

Numbers are author-reported; we'll re-run on solidpc with the canonical lm-eval pipeline before quoting in the Mythic-RDT release.

## Architectural fit for OpenMythos RDT — same as V2-Lite

| OpenMythos requirement | Native to DS-Coder-V2-Lite? | Notes |
|---|---|---|
| Three-stage shape | ✅ structural | layer 0 (dense) = prelude; one of layers 1–26 = recurrent; last layer(s) = coda |
| Shared-weight recurrence | ✅ trivial | reuse one transformer block T times |
| MLA | ✅ **native** | already implemented |
| DeepSeekMoE (routed + shared) | ✅ **native** | already implemented |
| Long context (≥32k) | ✅ **128k native** | farther than we need for v0; lets us do long-context recurrence experiments later |
| Depth-distinct routing | ➕ add via depth-LoRA on router | rank-16 LoRA × T iterations |
| LTI injection (A, B) | ➕ add (Parcae) | new params |
| Identity-biased gating | ➕ add (retrofit-recurrence) | new per-loop scalar |
| Per-loop LayerScale | ➕ add | new per-loop scalar |

Total trainable additions: ~7 M params. Frozen: ~15.7 B.

## Recurrent layer choice (Phase 0 task)

- **Avoid layer 1** (right after the dense FFN, likely transitional) and **layer 26** (last MoE, likely transitional).
- **Middle range candidates**: layers 12–16. Default: **layer 13** (floor(27/2) for 0-indexed).
- **Phase 0 probe**: pick 3 candidate layers (e.g., 10, 13, 16), run T=1/T=4/T=8 untrained on 100 prompts (mix of HumanEval-style code prompts and FineWeb-Edu prose), choose the one with cleanest behavior. Code prompts matter here because the base is code-specialized and we want to preserve that strength.

## Memory budget for fine-tune (Stage 1)

| Setup | bf16 base | KV cache @ 4k seq, batch 4 | Activations (T=8 ckpt) | Trainable+opt | **Total VRAM** | Fits on |
|---|---|---|---|---|---|---|
| Naive bf16 | 31 GB | 1.2 GB | ~150 MB | ~30 MB | **~33 GB** | 1× A100/H100 80GB ✅; 1× RTX 6000 Ada 48GB ✅; **NOT 24GB** |
| 4-bit base (bnb NF4) | 8 GB | 1.2 GB | ~150 MB | ~30 MB | **~10 GB** | **1× 3090 ✅; 1× 4090 ✅** |
| 4-bit base + DDP × 4× 3090 | 8 GB / GPU | 1.2 GB | ~150 MB | ~30 MB | ~10 GB / GPU | **solidpc 4× 3090 ✅** |

MLA's tiny KV footprint means we can run **batch 4 at 4k seq** on a single 3090 with 4-bit base. That's a real practical advantage for fine-tune iteration speed.

## Throughput estimate (Stage 1 fine-tune)

Effective compute per training token at T=8: 1 dense prelude + 8 × MoE recurrent + 1 MoE coda = **10 effective layers** at 2.4 B-A → ~14 GFLOPs/token forward, ~42 GFLOPs/token train.

| GPUs | tok/s training (estimate ±30%) | tok/day | 1 B tokens (pilot) | 5 B tokens (release) |
|---|---|---|---|---|
| 1× 3090 (24 GB) — 4-bit base | 500–800 | ~50 M | ~20 days | ~100 days |
| **4× 3090 DDP — 4-bit base** | **1800–3000** | **~200 M** | **~5 days** | **~25 days** |
| 1× H100 80GB | 2200–3500 | ~250 M | ~4 days | ~20 days |
| 2× H100 vast.ai | 4000–6000 | ~450 M | ~2.2 days | **~11 days** |
| 4× H100 vast.ai | 7500–11000 | ~850 M | ~1.2 days | **~6 days** |

**Stage 1 v0 release target**: 5 B tokens on 2× H100 (~$600) or 4× 3090 on solidpc (~3-4 weeks free).

## Files needed

```bash
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --local-dir base/DeepSeek-Coder-V2-Lite-Instruct \
    --local-dir-use-symlinks False
```

Expected files (total 31.4 GB):

| File | Size |
|---|---|
| config.json | 1.5 KB |
| configuration_deepseek.py | 10.3 KB |
| modeling_deepseek.py | 78.7 KB |
| tokenization_deepseek_fast.py | 1.4 KB |
| tokenizer.json | 4.6 MB |
| tokenizer_config.json | 1.3 KB |
| generation_config.json | 0.2 KB |
| model-00001-of-000004.safetensors | 8.6 GB |
| model-00002-of-000004.safetensors | 8.6 GB |
| model-00003-of-000004.safetensors | 8.6 GB |
| model-00004-of-000004.safetensors | 5.6 GB |
| model.safetensors.index.json | 480 KB |

Verify SHA256 after download (will be added to `LOCAL_SHA256` once we've fetched once).

## Specific gotchas for DS-Coder-V2-Lite-Instruct

- **Custom code via `trust_remote_code=True`**: the HF repo ships `modeling_deepseek.py`, `configuration_deepseek.py`, `tokenization_deepseek_fast.py`. We can either (a) keep `trust_remote_code=True` and import from the downloaded base directory, or (b) vendor the modeling code into our `src/mythic_rdt/` package. Vendoring is cleaner long-term but means tracking upstream patches.
- **Aux loss for load balance**: keep DeepSeek's `α₁ = 0.001`. With recurrent reuse, raise to 0.005 if expert utilization collapses (<30 % experts ever active across the loop).
- **Routed-expert bias correction**: re-tune during fine-tune (very few params, big effect).
- **Shared experts**: always-on, fire every loop iteration. **Don't add LoRA to them** — they already participate in every step.
- **No `--reasoning-format` magic**: standard `local-chat-completions` works. The "deepseek" reasoning format flag is a Gemma 4 quirk, not a DeepSeek model property.
- **GPQA cache key**: use `--use_cache <workdir>/<bench>_cache/dscoderv2lite` to avoid cache collisions.
- **Long context**: 128k native. For v0 fine-tune we use 4k–16k sequences (compute budget); the 128k headroom is for downstream apps.
- **MoE on layer 1 onwards**: layer 0 is dense FFN. Don't pick layer 0 as recurrent block; the routing dynamic isn't there.

## Quick-look config.json (canonical fields)

```json
{
  "model_type": "deepseek_v2",
  "hidden_size": 2048,
  "intermediate_size": 10944,
  "moe_intermediate_size": 1408,
  "num_hidden_layers": 27,
  "num_attention_heads": 16,
  "num_key_value_heads": 16,
  "n_shared_experts": 2,
  "n_routed_experts": 64,
  "num_experts_per_tok": 6,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1,
  "kv_lora_rank": 512,
  "q_lora_rank": null,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "rope_theta": 10000.0,
  "rope_scaling": {"type": "yarn", "factor": 40, ...},
  "vocab_size": 102400,
  "max_position_embeddings": 163840,
  "auto_map": {"AutoConfig": "configuration_deepseek.DeepseekV2Config",
               "AutoModel": "modeling_deepseek.DeepseekV2Model",
               "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM"}
}
```

## Stage-1 success criteria (Mythic-Coder)

Headline benchmark is **HumanEval pass@1** because the base is code-specialized. Targets:

- T=1 HumanEval pass@1 ≥ base − 1 pp (within noise; should be near-identical to base).
- **T=8 HumanEval pass@1 ≥ base + 5 pp** (i.e., target ≥ 86 % from a base of ~81 %).
- T=8 MBPP+ ≥ base + 3 pp.
- T=8 LiveCodeBench ≥ base + 4 pp (this is where multi-step deliberation should pay off the most).
- T=1 MMLU ≤ base + 1.5 pp drift.
- T=1 GSM8K within 2 pp of base.
- No mode collapse, no markdown-fence regression in HumanEval samples (parent project bug-015 — chat-mode wrapping in ` ``` ` makes scorer eval to 0 % even on correct solutions).

If those hold: publish **`ManniX-ITA/Mythic-RDT-Coder-V2-Lite`**. The headline pitch:

> "A 16 B / 2.4 B-active code model that scales like a 50 B model when you turn the depth knob up. Same storage as DS-Coder-V2-Lite-Instruct; pass@1 climbs 5+ points on HumanEval at T=8 with no extra parameters."

That's a publishable contribution and an artifact people will actually use, because **HumanEval@86 % at 16 B storage is genuinely useful** — much more so than a small-model RDT at 40 % GPQA would be. The recurrence story is concrete, the benchmark gain is meaningful, and the model is small enough to be deployed.

## Mythic-Coder-specific anti-patterns to avoid

- **Don't** evaluate via `local-chat-completions` for HumanEval. Parent project bug-015: chat mode wraps generations in ` ```python ` fences, scorer fails at `exec(prompt+gen)`, reports 0 % even when the code is correct. Use `local-completions` + `/v1/completions` (raw text completion) for HumanEval. Confirm by inspecting `samples_humaneval_*.jsonl` for fences before trusting any score.
- **Don't** train without a code-prose mix. If we only train on FineWeb-Edu, the coding strength will degrade. Use a 50/50 mix: FineWeb-Edu for general prose preservation + The-Stack-V2 / CodeNet / open-instruct-code for code preservation.
- **Don't** assume DeepSeek's chat template is identical to V2-Lite-Chat. The Coder variant has its own template; verify with `tokenizer.apply_chat_template` round-trip before evaluating.
