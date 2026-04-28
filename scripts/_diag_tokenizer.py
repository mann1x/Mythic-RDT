"""Tokenizer roundtrip + decode method comparison."""
from transformers import AutoTokenizer

BASE = "base/DeepSeek-Coder-V2-Lite-Instruct"

for use_fast in (True, False):
    print(f"\n========== use_fast={use_fast} ==========")
    t = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=use_fast)
    print("class:", type(t).__name__)
    s = "Please complete the following Python function. Hello world."
    ids = t.encode(s, add_special_tokens=False)
    print(f"encoded {len(ids)} tokens:", ids)
    toks = t.convert_ids_to_tokens(ids)
    print("convert_ids_to_tokens:", toks)
    print("decode (full string):", repr(t.decode(ids, skip_special_tokens=True)))
    if hasattr(t, "convert_tokens_to_string"):
        try:
            print("convert_tokens_to_string:", repr(t.convert_tokens_to_string(toks)))
        except Exception as e:
            print("convert_tokens_to_string failed:", e)

    # Round-trip a Chinese fragment to see byte-level handling
    s2 = "本身就是 hello"
    ids2 = t.encode(s2, add_special_tokens=False)
    print(f"chinese encode -> {ids2}")
    print(f"chinese decode -> {t.decode(ids2, skip_special_tokens=True)!r}")
