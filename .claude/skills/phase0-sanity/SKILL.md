---
name: phase0-sanity
description: Runs the Phase 0 decision gate from MASTER_PLAN.md §5 — verifies the Mythic-Gemma4 RDT wrapper is bit-exact with base Gemma 4 at T=1/gate=0, and that T=4/8/16 untrained loops produce no NaN/inf and <20% gibberish on 100 FineWeb-Edu prompts. Use when user says 'phase 0', 'sanity check', 'decision gate', 'verify wrapper', 'before fine-tune', or 'pick recurrent_layer_idx'. Picks recurrent_layer_idx from ../scripts/expert_neuron_v4.json avoiding bf16-broken layers 11-29. Do NOT use for fine-tuning, eval runs (GPQA/HumanEval), or quantization — those are Phase 1+. Do NOT skip this gate before any Phase 1 work.
paths:
  - experiments/00_phase0_sanity/**
  - src/mythic_gemma4/**
  - MASTER_PLAN.md
---
# Phase 0 Sanity / Decision Gate

Wraps Gemma 4 26B-A4B as Mythic-Gemma4 RDT and proves the wrapper is correct **before any fine-tune compute is spent**. Two hard gates from `MASTER_PLAN.md` §5:

- **Gate A (bit-exact)**: T=1 with gate logit = -∞ (effective gate=0) must produce logits identical to base Gemma 4, fp32, max-abs-diff < 1e-5 on 32 prompts.
- **Gate B (stability)**: T∈{4,8,16} untrained must have zero NaN/inf and <20% gibberish on 100 FineWeb-Edu prompts.

Failing either gate means the wrapper is wrong. **Do not proceed to Phase 1 fine-tuning.**

## Critical

1. **Never run on `/tmp`.** Weights, sample dumps, logits → a numbered phase-0 sanity dir under `experiments/` on persistent disk. /tmp is 64 GB tmpfs and will trash the run.
2. **Use `lightseek` conda env**: `conda activate lightseek`. Transformers 5.5.0 is required for Gemma 4 tokenizer. Do NOT touch the `vllm` env.
3. **Base model = the canonical `ManniX-ITA/gemma-4-A4B-98e-v3-it` HF download** (not `../google/gemma-4-A4B-98e-hybrid/` — bytes differ; see `BASE_MODEL_ANALYSIS.md`). Verify the base config is loadable before running gates.
4. **fp32 for Gate A only.** Gate A is bit-exact comparison; bf16 round-off makes it impossible. Load both base and wrapper as `torch_dtype=torch.float32` for this gate. Gate B may use bf16.
5. **Avoid bf16-broken layer band 11-29** when picking `recurrent_layer_idx`. These layers showed bf16 numerical issues in the parent project (`../scripts/expert_neuron_v4.json`). Pick from outside this band.
6. **Identity-bias the gate to -∞ for Gate A.** Set `gate_logit_init = -1e9` (or use a `gate_disabled=True` config flag). With sigmoid(gate) ≈ 0, the recurrent contribution vanishes and the model must equal base.
7. **Read `MASTER_PLAN.md` §5 first.** It is the source of truth for the gate definitions; if the numbers below differ from the plan, the plan wins.
8. **Log to the phase-0 sanity dir's notes file** with hypothesis, command, observed numbers, verdict (PASS/FAIL). Required by OpenWolf conventions.

## Instructions

### Step 1 — Pick `recurrent_layer_idx`

This step's output is the integer that every later step uses.

1. Read the per-layer contribution data:
   ```bash
   python -c "import json; d=json.load(open('../scripts/expert_neuron_v4.json')); print(sorted([(i, l.get('mean_contribution', 0)) for i,l in enumerate(d['layers'])], key=lambda x: -x[1])[:20])"
   ```
2. From the top-contribution layers, **exclude indices 11-29** (bf16-broken band). Prefer a layer in the 30-50 range; that's the empirical sweet spot for Gemma 4 mid-stack recurrence.
3. Record the pick into the phase-0 sanity dir's `recurrent_layer_idx.txt` (single integer).

**Verify before proceeding:** `cat <phase-0-sanity-dir>/recurrent_layer_idx.txt` shows a single integer outside [11, 29].

### Step 2 — Smallest viable wrapper

Write a minimal `MythicGemma4ForCausalLM` that does only what Gate A needs. No LTI, no LoRA, no LayerScale yet — those are Phase 1.

Location: project's modeling module (skeleton; full version comes later):

- `MythicGemma4Config(Gemma4Config)` with fields: `recurrent_layer_idx: int`, `loop_iters: int = 1`, `gate_logit_init: float = -1e9`.
- `MythicGemma4ForCausalLM(Gemma4ForCausalLM)` overriding `forward`. The override:
  1. Runs the base layer stack normally.
  2. After layer `recurrent_layer_idx`, runs that single layer `loop_iters - 1` extra times with `h_{t+1} = h_t + sigmoid(gate_logit) * (layer(h_t) - h_t)`.
  3. With `gate_logit_init = -1e9`, sigmoid ≈ 0, so output equals base.
- Register: `AutoConfig.register("mythic_gemma4", MythicGemma4Config)` and `AutoModelForCausalLM.register(MythicGemma4Config, MythicGemma4ForCausalLM)`.

**Verify** with an import smoke test:

```bash
conda activate lightseek
python -c "from mythic_gemma4 import MythicGemma4ForCausalLM; print('ok')"
```

### Step 3 — Gate A: bit-exact T=1, gate=0

Write `gate_a_bitexact.py` inside the phase-0 sanity dir. It must:

1. Load 32 short prompts (≤64 tokens) from FineWeb-Edu. Cache the token IDs to `prompts_32.pt` in the same dir.
2. Load base in fp32: `Gemma4ForCausalLM.from_pretrained('<base hf path>', torch_dtype=torch.float32, device_map='cuda:0')`.
3. Load wrapper in fp32 with same weights, `loop_iters=1`, `gate_logit_init=-1e9`, `recurrent_layer_idx=<N from Step 1>`.
4. For each prompt: forward both, compute `(logits_wrapper - logits_base).abs().max()`.
5. PASS if `max_diff < 1e-5` for all 32 prompts. Print min/median/max.

Run:
```bash
conda activate lightseek
cd /srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/Mythic-Gemma4
python <phase-0-sanity-dir>/gate_a_bitexact.py 2>&1 | tee <phase-0-sanity-dir>/gate_a.log
```

**Verify:** Last line of the gate A log reads `GATE A: PASS (max_diff=<x>, x<1e-5)`. If FAIL, **stop** — debug the wrapper, do not run Gate B.

### Step 4 — Gate B: T∈{4,8,16} stability, untrained

Write `gate_b_stability.py` in the phase-0 sanity dir. Uses output of Step 1, runs after Step 3 passes. May use bf16 (large T in fp32 OOMs the 3090).

1. Load 100 FineWeb-Edu prompts (16-128 tokens each). Cache to `prompts_100.pt` in the same dir.
2. For each `T in [4, 8, 16]`:
   - Load wrapper bf16 with `loop_iters=T`, `gate_logit_init=0.0` (sigmoid ≈ 0.5 — actually engages the loop, untrained).
   - Generate 64 new tokens per prompt, greedy.
   - Count: (a) any NaN/inf in logits (`torch.isfinite(logits).all()`), (b) gibberish — defined as: <30% of generated tokens are in the top 10000 most frequent tokens of the tokenizer, OR a 4-gram repeats >5 times in 64 tokens.
3. PASS criteria per T: NaN/inf count = 0 AND gibberish_rate < 0.20.

Run:
```bash
python <phase-0-sanity-dir>/gate_b_stability.py 2>&1 | tee <phase-0-sanity-dir>/gate_b.log
```

**Verify:** the gate B log shows `T=4 PASS`, `T=8 PASS`, `T=16 PASS`. If T=16 fails but T=4/8 pass, that is still a soft pass — note in the dir's hypothesis-and-result file and proceed cautiously.

### Step 5 — Record verdict

Append to the phase-0 sanity dir's hypothesis-and-result file:

- Hypothesis: wrapper is correct for chosen `recurrent_layer_idx=<N>`.
- Gate A result: PASS/FAIL, max_diff number, # prompts.
- Gate B result: PASS/FAIL per T, NaN counts, gibberish rates.
- Verdict: GO / NO-GO for Phase 1.
- Append one line to `../.wolf/memory.md`: `| HH:MM | phase0 sanity | <phase-0-sanity-dir>/ | <verdict> | ~tokens |`.

**Verify:** `grep -E 'GATE [AB]: (PASS|FAIL)' <phase-0-sanity-dir>/<notes-file>` shows two lines.

## Examples

### Example 1 — Fresh run

User says: *"Run the phase 0 sanity check before we start fine-tuning."*

Actions:
1. `conda activate lightseek` and confirm `python -c 'import transformers; print(transformers.__version__)'` is `5.5.0`.
2. Step 1: read `../scripts/expert_neuron_v4.json`, top contributors outside [11,29] are layers {38, 42, 47}. Pick **42**, write to `recurrent_layer_idx.txt` in the phase-0 sanity dir.
3. Step 2: write minimal wrapper, smoke-test import.
4. Step 3: run Gate A with `recurrent_layer_idx=42`, fp32. Result: `max_diff=3.2e-7` → PASS.
5. Step 4: run Gate B, bf16. Results: `T=4` 0 NaN, 4% gibberish; `T=8` 0 NaN, 11% gibberish; `T=16` 0 NaN, 17% gibberish → all PASS.
6. Step 5: notes-file verdict GO. Memory line appended.

Result: Phase 1 fine-tuning is unblocked. `recurrent_layer_idx=42` is fixed for downstream scripts.

### Example 2 — Gate A fails

User says: *"Verify the wrapper."*

Gate A reports `max_diff=0.018` for all 32 prompts. **Stop. Do not run Gate B.** Likely causes (in order): wrong layer index off-by-one, residual added twice in `forward`, gate logit not actually gating the residual, or hidden-state dtype cast somewhere mid-loop. Fix wrapper, re-run only Gate A. Log the bug to `../.wolf/buglog.json`.

## Common Issues

- **`max_diff ≈ 1e-3` in Gate A, not 1e-7**: you loaded one of the two models in bf16. Re-check both `from_pretrained` calls have `torch_dtype=torch.float32`. Bit-exact comparison in bf16 is impossible.
- **`max_diff ≈ 0.5` in Gate A**: the gate is not actually disabled. Confirm `sigmoid(-1e9) == 0.0` in your code path and that it multiplies the *delta*, not the absolute residual. With `loop_iters=1` your loop body should execute zero extra iterations — verify with a print.
- **`CUDA out of memory` during Gate B at T=16**: drop generation length to 32 tokens, or batch size to 1, or cast the prelude+coda to bf16 and only the recurrent layer to fp32. Do NOT skip T=16 silently — record the OOM in the notes file and rerun with reduced budget.
- **`AttributeError: 'Gemma4Config' object has no attribute 'recurrent_layer_idx'`**: `MythicGemma4Config` is not registered, or the saved config was the base one. Re-save the config: `MythicGemma4Config(...).save_pretrained(...)` and reload.
- **Tokenizer mismatch / `KeyError: '<|...|>'`**: wrong env. `conda activate lightseek` (transformers 5.5.0). The `vllm` env has an older transformers and will fail on Gemma 4 special tokens.
- **Gibberish rate >50% at T=4 untrained**: this can be legitimate — untrained loops can drift. But if T=4 is >50% gibberish while T=1 is fine, the LTI/identity-bias initialization is wrong: the gate at `gate_logit=0.0` (sigmoid=0.5) is too aggressive. Re-init with `gate_logit_init=-2.0` (sigmoid≈0.12) for Gate B and document the change in the notes file.
- **`FileNotFoundError: ../scripts/expert_neuron_v4.json`**: the parent project file moved. Run `ls ../scripts/expert_neuron_v4*.json` and pick the latest; update Step 1 path. Do NOT pick a random layer — the bf16-broken band exclusion depends on this data.
- **Gate A passes, Gate B has NaN at T=8 only**: numerical blow-up in mid-stack. This is the bf16-broken band biting you — verify your `recurrent_layer_idx` is truly outside [11,29]. If it is, the layer still has a marginal numeric issue; pick a different top-contributor layer and re-run both gates.
