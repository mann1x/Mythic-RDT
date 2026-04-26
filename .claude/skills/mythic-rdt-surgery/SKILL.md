---
name: mythic-rdt-surgery
description: Builds the Mythic-Gemma4 recurrent block (LTI injection, identity-biased gating, depth-LoRA, per-iteration LayerScale) per MASTER_PLAN.md §4. Initializes values so T=1 is bit-exact with single-pass middle layer through Gemma 4. Use when user says 'build the wrapper', 'add LTI', 'implement gating', 'create MythicGemma4ForCausalLM', 'phase 1 surgery', 'wire the recurrent block', 'init Parcae A', or 'add depth-LoRA'. Includes Parcae A=Diag(-exp(log_A)) parameterization, fp32 norms inside the loop, identity-biased gate, and HF AutoConfig/AutoModel registration. Do NOT use for from-scratch training, MLA conversion, ACT halting head training, or Wanda/SVD/DERN compression — those are explicitly out of scope per MASTER_PLAN.md and parent CLAUDE.md.
---
# Mythic-RDT Surgery — Phase 1 Wrapper

Builds the recurrent-depth wrapper around Gemma 4 26B-A4B per `MASTER_PLAN.md` §4. The output is two custom-code modules (the config and the modeling file) plus a registration shim that load via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

## Critical

1. **Read `MASTER_PLAN.md` first.** It contains the port/no-port matrix and the chosen `recurrent_layer_idx`. Do not invent layer indices — pull them from `../scripts/expert_neuron_v4.json` per-layer contribution data, as referenced in the plan.
2. **Base model is the canonical `ManniX-ITA/gemma-4-A4B-98e-v3-it` HF download**, NOT `../google/gemma-4-A4B-98e-hybrid/`. SHA256s differ — see `BASE_MODEL_ANALYSIS.md`. Any script that hardcodes the wrong path is a bug.
3. **T=1 must be bit-exact with single-pass Gemma 4 through the chosen middle layer.** This is the non-negotiable correctness gate before any fine-tune. Concretely:
   - LayerScale γ = `lti_init` (default `1e-4`) — small, not zero, but the gate must dominate.
   - Gate logit bias initialized so `sigmoid(b_g) ≈ 0` (use `-8.0`).
   - LTI matrices A, B initialized so the injection is ≈ 0 at init: `log_A` drawn from `N(0, 0.02)` (so A ≈ -1 → fast decay); B is `N(0, 0.02)`.
   - Depth-LoRA `B` matrices initialized to **zero** (standard LoRA init). `A` matrices Kaiming-uniform.
   - Result: `h_{t+1} = h_t + sigmoid(b_g) * γ * [LTI + RecurrentBlock(...)]` ≈ `h_t` at T=1 init. Verify before proceeding.
4. **Spectral radius constraint on A is mandatory.** Always parameterize as `A = -torch.exp(log_A)` (Parcae form) — never store A directly. This guarantees ρ(A) < 1 by construction. Comment this in the code.
5. **Norms inside the recurrent loop run in fp32.** Cast `h` to fp32 before the RMSNorm inside the loop body, cast back to model dtype after. Loop at bf16 with fp32 norms only — anything else risks NaN at T≥4.
6. **Never write checkpoints to `/tmp`.** Outputs go to the user-specified `--output` path on persistent disk. /tmp is tmpfs (64 GB RAM).
7. **Use `lightseek` conda env** (`conda activate lightseek`) — transformers 5.5.0 has the Gemma 4 tokenizer. Never touch the `vllm` env.

## Instructions

### Step 1 — Pick `recurrent_layer_idx` from contribution data

Read `../scripts/expert_neuron_v4.json` and select the layer index per `MASTER_PLAN.md` §4 criterion (highest-contribution middle layer, NOT prelude, NOT coda). Record the choice in the experiment notes.

```python
import json
with open("../scripts/expert_neuron_v4.json") as f:
    data = json.load(f)
# pick per the plan's criterion — do not invent
```

**Verify**: the chosen index is in the middle third of the model's 36 layers (Gemma 4 26B-A4B has 36) and matches the value committed in `MASTER_PLAN.md`. Do not proceed otherwise.

### Step 2 — Write the config module at repo root

Subclass `Gemma4Config`. Add exactly these fields (names are non-negotiable, used by HF custom-code path):

```python
from transformers import Gemma4Config

class MythicGemma4Config(Gemma4Config):
    model_type = "mythic_gemma4"

    def __init__(
        self,
        prelude_layers: list[int] | None = None,
        coda_layers: list[int] | None = None,
        recurrent_layer_idx: int = 18,
        max_loop_iters: int = 16,
        train_loop_iters: int = 1,
        lti_init: float = 1e-4,
        depth_lora_rank: int = 8,
        halting_strategy: str = "fixed",  # phase-1: fixed only; ACT is phase-2
        gate_bias_init: float = -8.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prelude_layers = prelude_layers
        self.coda_layers = coda_layers
        self.recurrent_layer_idx = recurrent_layer_idx
        self.max_loop_iters = max_loop_iters
        self.train_loop_iters = train_loop_iters
        self.lti_init = lti_init
        self.depth_lora_rank = depth_lora_rank
        self.halting_strategy = halting_strategy
        self.gate_bias_init = gate_bias_init
```

**Verify** with a quick import smoke test:

```bash
conda activate lightseek
python -c "from configuration_mythic_gemma4 import MythicGemma4Config; print(MythicGemma4Config())"
```

Do not proceed otherwise.

### Step 3 — Write the modeling module at repo root

This file holds: `LTIInjection`, `DepthLoRA`, `RecurrentBlock`, `MythicGemma4Model`, `MythicGemma4ForCausalLM`. Skeleton (fill bodies in this order):

```python
import torch
import torch.nn as nn
from transformers import Gemma4ForCausalLM, Gemma4Model
from .configuration_mythic_gemma4 import MythicGemma4Config


class LTIInjection(nn.Module):
    """Parcae-form linear time-invariant injection. ρ(A) < 1 by construction."""
    def __init__(self, hidden_size: int):
        super().__init__()
        # log_A → A = -exp(log_A); guarantees A ∈ (-∞, 0) so ρ(A) < 1
        self.log_A = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.B = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.log_A)              # spectral radius constraint
        return A * h + torch.matmul(e, self.B.T)


class DepthLoRA(nn.Module):
    """T distinct LoRA sets on Q/K/V/O. B init zero (standard LoRA)."""
    def __init__(self, hidden_size: int, rank: int, n_iters: int):
        super().__init__()
        self.A = nn.ParameterList([
            nn.Parameter(torch.empty(rank, hidden_size)) for _ in range(n_iters)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, rank)) for _ in range(n_iters)
        ])
        for a in self.A:
            nn.init.kaiming_uniform_(a, a=5 ** 0.5)

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return x @ self.A[t].T @ self.B[t].T


class MythicGemma4ForCausalLM(Gemma4ForCausalLM):
    config_class = MythicGemma4Config

    def __init__(self, config: MythicGemma4Config):
        super().__init__(config)
        h = config.hidden_size
        self.lti = LTIInjection(h)
        self.gate_bias = nn.Parameter(torch.tensor(config.gate_bias_init))
        # per-iteration LayerScale, init small
        self.layerscale = nn.Parameter(torch.full((config.max_loop_iters, h), config.lti_init))
        self.depth_lora = DepthLoRA(h, config.depth_lora_rank, config.max_loop_iters)

    def _recurrent_step(self, h, e, t):
        # fp32 norm inside loop
        h_fp32 = h.float()
        normed = self.model.layers[self.config.recurrent_layer_idx].input_layernorm(h_fp32)
        normed = normed.to(h.dtype)
        block_out = self.model.layers[self.config.recurrent_layer_idx](normed)[0]
        block_out = block_out + self.depth_lora(block_out, t)
        injection = self.lti(h, e) + block_out
        gate = torch.sigmoid(self.gate_bias)
        return h + gate * self.layerscale[t] * injection

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Run prelude → loop the middle layer T times → coda → lm_head
        # T = self.config.train_loop_iters in train mode; self.config.max_loop_iters at eval
        ...
```

**Verify**: at init, with T=1 and `gate_bias=-8.0`, `sigmoid(b_g) ≈ 3.4e-4` and γ ≈ 1e-4 — combined factor ≈ 3.4e-8, so the injection is effectively gated off. The LTI's B-projection is also small (0.02 scale). Run a single-token forward on the wrapped model and the base Gemma 4 with the same input; cosine similarity of final logits must be ≥ 0.9999. If not, the wrapper is broken — do not proceed.

### Step 4 — Register for `trust_remote_code` loading

At the bottom of the modeling module:

```python
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("mythic_gemma4", MythicGemma4Config)
AutoModelForCausalLM.register(MythicGemma4Config, MythicGemma4ForCausalLM)
```

Add `auto_map` to the saved config:

```python
config.auto_map = {
    "AutoConfig": "configuration_mythic_gemma4.MythicGemma4Config",
    "AutoModelForCausalLM": "modeling_mythic_gemma4.MythicGemma4ForCausalLM",
}
```

**Verify**: after `model.save_pretrained(<output dir>)`, reload with `AutoModelForCausalLM.from_pretrained(<output dir>, trust_remote_code=True)` in a fresh Python process. Both modeling files must be present in the saved dir. Do not proceed otherwise.

### Step 5 — Write the conversion script

`scripts/convert_to_mythic.py` — args: `--base <hf base path> --output <output path> --recurrent-layer-idx N`. Loads base, builds `MythicGemma4Config` from `base.config.to_dict()` plus the RDT fields, instantiates `MythicGemma4ForCausalLM(config)`, copies over the base state_dict (the new params start at their init values), saves with `auto_map`.

```bash
conda activate lightseek
python scripts/convert_to_mythic.py --base <hf base path> --output <output dir> --recurrent-layer-idx 18
```

**Verify**: `ls <output dir>` shows the model config, the two custom-code modeling files, and weight shards. No path under `/tmp`. Run T=1 sanity probe (Step 3 verify, but on the saved checkpoint).

### Step 6 — Add experiment probe

Under `experiments/`, create a numbered probe directory containing a hypothesis-and-result note (hypothesis: T=1 init is bit-exact-equivalent to base) and a `run.py` that compares logits cosine on 64 random inputs. Result must be ≥ 0.9999 mean cosine. Commit the result inline in the note.

## Examples

**User says**: "build the wrapper, recurrent layer 18, max iters 16"

**Actions**:
1. Read `MASTER_PLAN.md` §4 and `../scripts/expert_neuron_v4.json` — confirm layer 18 is the chosen middle-third high-contribution layer.
2. Write the config module with `recurrent_layer_idx=18, max_loop_iters=16, train_loop_iters=1`.
3. Write the modeling module with `LTIInjection`, `DepthLoRA`, `MythicGemma4ForCausalLM` per Step 3 skeleton. Parcae form: `A = -torch.exp(log_A)`. Gate bias init `-8.0`. LayerScale init `1e-4`. Depth-LoRA `B` matrices zero.
4. Add `AutoConfig.register` / `AutoModelForCausalLM.register` + `auto_map`.
5. Run T=1 cosine probe under a numbered `experiments/` subdir — confirm ≥ 0.9999.
6. Save to the output dir. Reload with `trust_remote_code=True` in a fresh process and re-run the probe.

**Result**: Wrapped model loads via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`, T=1 forward is numerically equivalent to base Gemma 4, all new params at safe init values ready for curriculum fine-tune.

## Common Issues

- **"T=1 cosine < 0.9999 vs base"**: gate or LayerScale not small enough at init, or LTI B projection not near-zero. Check: `model.gate_bias.item()` should be `-8.0`; `model.layerscale[0].abs().mean()` should be `≈ 1e-4`; `model.lti.B.std()` should be `≈ 0.02`. Also verify the recurrent block is being run **exactly once** at T=1 — an off-by-one in the loop range will run it twice.
- **"NaN in recurrent loop at T≥4"**: norms running in bf16 inside the loop. Fix: cast `h` to fp32 before `input_layernorm`, cast back after. Loop body stays bf16, only the norm is fp32.
- **"`KeyError: 'mythic_gemma4'` on reload"**: `auto_map` not in saved `config.json`, or `AutoConfig.register(...)` not at module top-level of the modeling module. Re-save with the registration calls executed at import time.
- **"Spectral radius > 1 after a few train steps"**: code stored A directly instead of `log_A`. Fix: `A` must be a derived quantity computed as `-torch.exp(self.log_A)` every forward; never make `A` itself an `nn.Parameter`.
- **"`ImportError: cannot import name 'Gemma4Config'`"**: wrong env. `conda activate lightseek` (transformers 5.5.0 has Gemma 4). Do NOT use `vllm` env.
- **"Saved checkpoint missing modeling files"**: HF `save_pretrained` does not auto-copy custom code unless `auto_map` is set on the config object **before** the save call. Set it on `model.config.auto_map = {...}` then save.
- **"Depth-LoRA doing something at T=1 init"**: `B` matrices not zero. Standard LoRA init is `A` random (Kaiming), `B` zero — verify `model.depth_lora.B[0].abs().sum() == 0` after construction.
- **"Wrote intermediate to /tmp and lost it on reboot"**: re-run with `--output` on persistent disk. `/tmp` is tmpfs, 64 GB RAM, wiped on reboot. Parent `CLAUDE.md` rule.
