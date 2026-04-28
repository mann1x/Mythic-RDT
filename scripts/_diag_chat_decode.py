"""One-off diagnostic: render chat prompt, run base.generate, dump raw output.

Goal: figure out why DS-Coder-V2-Lite-Instruct emits multilingual gibberish
even with the chat template applied + BOS in place. We need to see the actual
token IDs, the per-token decode, and the full string-decode of the assistant
turn.
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "base/DeepSeek-Coder-V2-Lite-Instruct"

t = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=True)
print("tokenizer class:", type(t).__name__)
print("bos_token:", repr(t.bos_token), "id:", t.bos_token_id)
print("eos_token:", repr(t.eos_token), "id:", t.eos_token_id)
print("pad_token:", repr(t.pad_token), "id:", t.pad_token_id)

m = AutoModelForCausalLM.from_pretrained(
    BASE,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True,
)
m.eval()

prompt = 'def add(a, b):\n    """Return a+b."""'
content = (
    "Please complete the following Python function. "
    "Respond with ONLY the completed code in a Python markdown block.\n\n"
    "```python\n" + prompt + "\n```"
)
msg = [{"role": "user", "content": content}]

chat = t.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
print("=== rendered chat ===")
print(repr(chat))

ids = t(chat, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
print("=== input ids ===")
print("len:", ids.shape[1])
print("first 30:", ids[0][:30].tolist())
print("decoded[:1] tok by tok (first 30):")
for tid in ids[0][:30].tolist():
    print(f"  {tid:>6} -> {t.decode([tid])!r}")

print("=== generating use_cache=False (matches smoke script) ===")
with torch.no_grad():
    out = m.generate(
        input_ids=ids,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=t.eos_token_id,
        use_cache=False,
    )
gen_ids = out[0, ids.shape[1]:].tolist()
print("gen ids:", gen_ids[:50])
print("gen decoded skip-special:", repr(t.decode(gen_ids, skip_special_tokens=True)))
print("gen decoded keep-special:", repr(t.decode(gen_ids, skip_special_tokens=False)))
