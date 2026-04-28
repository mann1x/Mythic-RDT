"""Compare four base.generate configurations on a single HumanEval prompt:
  A. raw prompt + use_cache=True   (the natural way; will hit RoPE bug if any)
  B. raw prompt + use_cache=False  (the smoke-script way)
  C. chat prompt + use_cache=True
  D. chat prompt + use_cache=False
"""
import sys, traceback
import torch
from transformers import AutoModelForCausalLM

BASE = "base/DeepSeek-Coder-V2-Lite-Instruct"

sys.path.insert(0, BASE)
from tokenization_deepseek_fast import DeepseekTokenizerFast

t = DeepseekTokenizerFast.from_pretrained(BASE)

import transformers
_TF_MAJOR = int(transformers.__version__.split(".")[0])
_dtype_kw = {"dtype": torch.bfloat16} if _TF_MAJOR >= 5 else {"torch_dtype": torch.bfloat16}
m = AutoModelForCausalLM.from_pretrained(
    BASE, trust_remote_code=True,
    device_map="cuda", low_cpu_mem_usage=True,
    **_dtype_kw,
)
m.eval()

raw_prompt = (
    "from typing import List\n\n\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    "    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n"
    "    given threshold.\n"
    "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
    "    False\n"
    "    \"\"\"\n"
)
chat_prompt = t.apply_chat_template(
    [{"role":"user","content":"Please complete the following Python function. "
                              "Respond with ONLY the code in a Python block.\n\n```python\n"
                              + raw_prompt + "```"}],
    tokenize=False, add_generation_prompt=True,
)

# raw needs BOS prepended manually (raw tokenizer has add_bos_token=True so encode adds it)
def gen(prompt, use_cache, label, add_special):
    print(f"\n=== {label}  use_cache={use_cache}  add_special={add_special} ===")
    enc = t(prompt, return_tensors="pt", add_special_tokens=add_special)
    ids = enc.input_ids.to("cuda")
    am = enc.attention_mask.to("cuda")
    print(f"input_ids[0][:5] = {ids[0][:5].tolist()}  (BOS={t.bos_token_id})")
    print(f"input_ids[0][-5:] = {ids[0][-5:].tolist()}  len={ids.shape[1]}")
    try:
        with torch.no_grad():
            out = m.generate(
                input_ids=ids, attention_mask=am,
                max_new_tokens=80, do_sample=False,
                pad_token_id=t.eos_token_id, use_cache=use_cache,
            )
        gen_ids = out[0, ids.shape[1]:].tolist()
        print(f"first 10 gen ids: {gen_ids[:10]}")
        print(f"decoded:\n{t.decode(gen_ids, skip_special_tokens=True)!r}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

gen(raw_prompt, True, "A. raw + cache",  True)
gen(raw_prompt, False, "B. raw + no cache",  True)
gen(chat_prompt, True, "C. chat + cache",  False)  # template already has BOS
gen(chat_prompt, False, "D. chat + no cache", False)
