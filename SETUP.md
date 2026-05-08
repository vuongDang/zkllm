# zkLLM Setup Guide

## Hardware

- **GPU**: NVIDIA A100 (sm_80). Driver must support CUDA 12.x.

## Steps

### 1. Fix Makefile for A100

In `Makefile`, set:
```
ARCH := sm_80
```
The original repo defaults to `sm_86` (RTX A6000).
`sm_80` for (A100).

### 2. Create conda environment

```bash
conda create -n zkllm-env python=3.11
conda activate zkllm-env
conda install cuda -c nvidia/label/cuda-12.1.0
```

### 3. Install Python packages

```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.35.2
pip install accelerate
```

**Why these specific versions:**
- `torch 2.5.1+cu121`: newer torch (e.g. 2.11) requires CUDA 13.0 driver which the A100 here does not have
- `transformers 4.35.2`: last version before LLaMA attention was refactored — newer versions removed `layer.self_attn.num_heads` and `layer.self_attn.rotary_emb`
- `accelerate`: required by `from_pretrained` with `device_map="auto"`

### 4. Download model

```bash
# Downloads to default HF cache (~/.cache/huggingface/hub/)
python download-models.py "huggingface_token"
```

### 5. Preprocess weights (one-time per model)

```bash
python llama-ppgen.py 7 16      # generates public parameters
python llama-commit.py 7 16     # commits model weights
```

### 6. Build CUDA binaries

```bash
make all
```

### 7. Run

```bash
python full_run.py "Your prompt here" --max_new_tokens 10
```