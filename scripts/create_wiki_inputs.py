import json
from datasets import load_dataset
from transformers import AutoTokenizer

N = 64
TARGET = 256 - 1  # 255 tokens, leaving room for BOS

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', local_files_only=True)

ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)

results = []
for article in ds:
    text = article['text']
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:TARGET]
    if len(ids) < TARGET:
        ids += [tokenizer.eos_token_id] * (TARGET - len(ids))
    decoded = tokenizer.decode(ids)
    check = tokenizer.encode(decoded, add_special_tokens=False)
    print(f'[{len(results)+1}/{N}] "{article["title"]}" — {len(check)} tokens')
    results.append(decoded)
    if len(results) == N:
        break

with open('wiki_inputs.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nwiki_inputs.json written ({len(results)} prompts).')
