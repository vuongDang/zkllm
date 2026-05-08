import os
import sys

model_card = "meta-llama/Llama-2-7b-hf"
if len(sys.argv) < 2:
    print("Error: please provide a HuggingFace token", file=sys.stderr)
    sys.exit(1)
token = sys.argv[1]
    

from transformers import AutoTokenizer, AutoModelForCausalLM
# download the models
try:
    tokenizer = AutoTokenizer.from_pretrained(model_card, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_card,
        device_map="auto",
        token=token
        # quantization_config=quantization_config
    )
except RuntimeError:
    pass