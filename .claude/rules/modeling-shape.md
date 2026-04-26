---
paths:
  - src/mythic_gemma4/**
  - modeling_mythic_gemma4.py
  - configuration_mythic_gemma4.py
---

# HF custom-code shape (Mythic-Gemma4)

The model MUST load via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. Anything that breaks this shape is a bug.

## Class hierarchy

- `MythicGemma4Config(Gemma4Config)` — adds RDT fields: `prelude_layers`, `coda_layers`, `recurrent_layer_idx`, `max_loop_iters`, `train_loop_iters`, `lti_init`, `depth_lora_rank`, `halting_strategy`.
- `MythicGemma4ForCausalLM` — subclasses or composes `Gemma4ForCausalLM`. Recurrence implemented inside `forward()`.
- Register via `AutoConfig.register("mythic_gemma4", MythicGemma4Config)` and `AutoModelForCausalLM.register(MythicGemma4Config, MythicGemma4ForCausalLM)`.
- Use `register_for_auto_class()` on both so HF custom-code path picks them up.

## Recurrence shape (per `MASTER_PLAN.md` §4)

```
h_0 = Prelude(embed)
for t in 0..T-1:
    inj = A·h_t + B·e               # LTI injection
    block_out = RecurrentBlock(h_t, e)  # Gemma4 mid-layer + depth-LoRA[t]
    g = sigmoid(gate_t)             # init bias = -3 → g ≈ 0.047
    ls = layerscale_t               # init 1e-4
    h_{t+1} = h_t + ls·g·(inj + block_out)
logits = lm_head(norm(Coda(h_T)))
```

## Stability requirements

- `A := Diag(-exp(log_A))` parameterization → ρ(A) < 1 guaranteed (Parcae).
- All RMSNorms in the recurrence path use **fp32** (parent project bf16 NaN bug, layers 11-29).
- At T=1 with `gate=0`, output must be bit-exact with running the chosen middle layer once. Phase 0 gate.

## Won't include

MLA, ACT halting head (v0), from-scratch training paths, MoE re-layout. See `MASTER_PLAN.md` §3 port matrix.
