"""Diagnose: does the position_ids patch fix bs>1 padding+cache?

Generates the same LCB prompt at bs=1 and bs=2 (padded with a shorter
prompt), with KV cache enabled. If patch is correct, both produce the
same completion. If different, padding+cache is still broken.
"""
import sys, os, json
sys.path.insert(0, "/workspace/mythic-rdt/src")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from mythic_rdt.modeling import (
    MythicRDTDeepseekV2Config, MythicRDTDeepseekV2ForCausalLM,
)

BASE = "/workspace/mythic-rdt/base/DeepSeek-Coder-V2-Lite-Instruct"
CKPT = "/workspace/mythic-rdt/checkpoints/phase1_v4_anchored"

print("[diag] loading base nf4...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE, trust_remote_code=True, quantization_config=bnb,
    torch_dtype=torch.bfloat16, device_map="cuda",
)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
print(f"[diag] base loaded. layers={base.config.num_hidden_layers}")

# Wrap
cfg = MythicRDTDeepseekV2Config(
    prelude_layers=4, coda_layers=4, recurrent_layer_idx=10,
    block_layer_indices=list(range(4, 22)), block_mode=True,
    max_loop_iters=4, gate_init_bias=0.0, layerscale_clamp_max=1e-2,
    depth_lora_rank=8, lora_targets=("q_proj", "o_proj"),
)
wrapper = MythicRDTDeepseekV2ForCausalLM(cfg, base=base).eval()

# Load trainable
ck = os.path.join(CKPT, sorted([d for d in os.listdir(CKPT) if d.startswith("checkpoint-")])[-1])
trainable = torch.load(os.path.join(ck, "mythic_rdt_trainable.pt"), map_location="cpu")
wrapper.load_state_dict(trainable, strict=False)
wrapper = wrapper.to("cuda")
print(f"[diag] wrapper loaded from {ck}")

# Build a real LCB-like prompt + a short padding companion
long_prompt = ("<｜begin▁of▁sentence｜>User: Solve the following Python coding problem. "
               "Respond with ONLY the completed Solution class in a Python markdown block, no explanations.\n\n"
               "You are given an array of integers nums of size 3.\n"
               "Return the maximum possible number whose binary representation can be formed "
               "by concatenating the binary representations of all elements in some order.\n\n"
               "Example: Input: nums = [1,2,3]  Output: 30\n"
               "Constraints: nums.length == 3; 1 <= nums[i] <= 127.\n\n"
               "class Solution:\n    def maxGoodNumber(self, nums: List[int]) -> int:\n\nAssistant:")
short_prompt = "<｜begin▁of▁sentence｜>User: Hello\n\nAssistant:"

@torch.no_grad()
def gen(prompts, T=1, gen_tokens=64):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids = enc.input_ids.cuda(); attn = enc.attention_mask.cuda()
    print(f"[diag]   batch shape={ids.shape}  attn cumsum-1 last col per seq={(attn.long().cumsum(-1)-1)[:,-1].tolist()}")
    out = wrapper(ids, attention_mask=attn, T=T, use_cache=True, return_dict=True)
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1)
    cur_mask = torch.cat([attn, torch.ones_like(next_id).unsqueeze(1)], dim=1)
    generated = next_id.unsqueeze(1)
    for _ in range(gen_tokens - 1):
        out = wrapper(next_id.unsqueeze(1), attention_mask=cur_mask, T=T,
                      past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id).unsqueeze(1)], dim=1)
        generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
    return [tok.decode(generated[i].tolist(), skip_special_tokens=True) for i in range(generated.size(0))]

print("\n[diag] === bs=1 (no padding) ===")
out_bs1 = gen([long_prompt], T=1)
print(f"[diag]   bs=1 completion[:200]: {out_bs1[0][:200]!r}")

print("\n[diag] === bs=2, long+short (long is left-padded? short is) ===")
out_bs2 = gen([short_prompt, long_prompt], T=1)
print(f"[diag]   bs=2 short completion[:200]: {out_bs2[0][:200]!r}")
print(f"[diag]   bs=2 long  completion[:200]: {out_bs2[1][:200]!r}")

print("\n[diag] === bs=2 long-vs-bs1-long match? ===")
match = out_bs1[0] == out_bs2[1]
print(f"[diag]   exact match: {match}")
if not match:
    # Find first divergence
    for i, (a, b) in enumerate(zip(out_bs1[0], out_bs2[1])):
        if a != b:
            print(f"[diag]   first divergence at char {i}: bs1={a!r} vs bs2={b!r}")
            print(f"[diag]   bs1 head[i-30:i+30]={out_bs1[0][max(0,i-30):i+30]!r}")
            print(f"[diag]   bs2 head[i-30:i+30]={out_bs2[1][max(0,i-30):i+30]!r}")
            break
    else:
        print(f"[diag]   one is prefix of other; lens bs1={len(out_bs1[0])} bs2={len(out_bs2[1])}")
