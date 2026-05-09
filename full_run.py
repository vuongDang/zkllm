import os
import sys
import json
import argparse
import math
import torch
import numpy as np

import torch.nn.functional as F
import fileio_utils
from fileio_utils import load_int, save_int, to_int64, to_float, fromto_int64
from transformers import AutoTokenizer, AutoModelForCausalLM

SCALING_LOG = 16
VALUE_LOGSF  = 16
ACCU_LOGSF   = 20
SWIGLU_TABLE = 'swiglu-table.bin'


def padded_seq_len(seq_len, embed_dim, scaling_factor=1 << 16):
    # Constraints from ZK proof components:
    # 1. seq_len * embed_dim must be a power of 2 and divisible by scaling_factor (rescaling proofs)
    # 2. seq_len^2 must be divisible by scaling_factor (zkSoftmax tLookupRangeMapping with bs={1<<16,1<<16,1<<16})
    p = 1
    while p < seq_len or (p * embed_dim) % scaling_factor != 0 or (p * p) % scaling_factor != 0:
        p <<= 1
    return p


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def setup(model_size=7):
    model_card = f'meta-llama/Llama-2-{model_size}b-hf'
    workdir    = f'./zkllm-workdir/Llama-2-{model_size}b'
    tokenizer = AutoTokenizer.from_pretrained(model_card, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only=True)
    model.eval()
    embed_dim  = model.model.layers[0].self_attn.q_proj.in_features
    hidden_dim = model.model.layers[0].mlp.up_proj.out_features
    num_layers = len(model.model.layers)
    ensure_swiglu()
    return tokenizer, model, embed_dim, hidden_dim, num_layers, workdir

def run_cmd(cmd, msg=None):
    if os.system(cmd) != 0:
        if msg:
            print(msg)
        raise RuntimeError(f'Command failed: {cmd}')

def ensure_swiglu():
    if not os.path.exists(SWIGLU_TABLE):
        Xs = torch.arange(-(1 << 9), 1 << 9, step=1 / (1 << 12), device=0)
        save_int(Xs * torch.sigmoid(Xs), 1 << 16, SWIGLU_TABLE)


def run_layer(layer, layer_idx, seq_len, embed_dim, hidden_dim, workdir, input_file, output_file):
    lp        = f'layer-{layer_idx}'
    attn_in   = f'_tmp_attn_in_{layer_idx}.bin'
    attn_out  = f'_tmp_attn_out_{layer_idx}.bin'
    post_attn = f'_tmp_post_attn_{layer_idx}.bin'
    ffn_in    = f'_tmp_ffn_in_{layer_idx}.bin'
    ffn_out   = f'_tmp_ffn_out_{layer_idx}.bin'

    # 1. Input RMSNorm
    X = torch.tensor(np.fromfile(input_file, dtype=np.int32).reshape(seq_len, embed_dim),
                     dtype=torch.float64, device=0) / (1 << SCALING_LOG)
    rms_inv = 1 / torch.sqrt(torch.mean(X ** 2, dim=1) + layer.input_layernorm.variance_epsilon)
    save_int(rms_inv.float(), 1 << SCALING_LOG, 'rms_inv_temp.bin')
    run_cmd(f'./rmsnorm input {input_file} {seq_len} {embed_dim} {workdir} {lp} {attn_in}')
    os.remove('rms_inv_temp.bin')

    # 2. Self-attention
    run_cmd(f'./self-attn linear {attn_in} {seq_len} {embed_dim} {workdir} {lp} {attn_out}')

    Q = load_int('temp_Q.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    K = load_int('temp_K.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)
    V = load_int('temp_V.bin').reshape(seq_len, embed_dim).float() / (1 << VALUE_LOGSF)

    Q = Q.view(seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1)
    K = K.view(seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1)
    V = V.view(seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1)

    layer.self_attn.rotary_emb.to(0)
    cos, sin = layer.self_attn.rotary_emb(Q.float(), seq_len=seq_len)
    cos = cos.squeeze(0).squeeze(0)  # [seq_len, head_dim]
    sin = sin.squeeze(0).squeeze(0)
    Q = (Q * cos.unsqueeze(0) + rotate_half(Q) * sin.unsqueeze(0)).to(torch.float64)
    K = (K * cos.unsqueeze(0) + rotate_half(K) * sin.unsqueeze(0)).to(torch.float64)

    A = to_int64(Q @ K.transpose(-2, -1), ACCU_LOGSF)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=0, dtype=bool), diagonal=1)
    A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
    shift = math.sqrt(layer.self_attn.head_dim) * torch.log(
        (torch.exp(to_float(A, ACCU_LOGSF) / math.sqrt(layer.self_attn.head_dim)) * ~mask).sum(dim=-1, keepdim=True)
    )
    A -= to_int64(shift, ACCU_LOGSF)
    attn_weights = torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(layer.self_attn.head_dim)).float() * ~mask
    attn_v = fromto_int64(attn_weights @ V, VALUE_LOGSF)
    save_int(attn_v.transpose(0, 1).contiguous().view(seq_len, embed_dim).float(), 1 << VALUE_LOGSF, 'temp_attn_out.bin')

    run_cmd(f'./self-attn attn {attn_in} {seq_len} {embed_dim} {workdir} {lp} {attn_out}')
    run_cmd('rm -f ./temp_Q.bin ./temp_K.bin ./temp_V.bin ./temp_attn_out.bin')

    # Apply output projection (o_proj) — not proved by the binary, done in Python
    attn_v_flat = attn_v.float().transpose(0, 1).contiguous().view(seq_len, embed_dim).cpu()
    with torch.no_grad():
        attn_out_tensor = layer.self_attn.o_proj(attn_v_flat)
    save_int(attn_out_tensor, 1 << SCALING_LOG, attn_out)

    # 3. Skip connection (attention)
    run_cmd(f'./skip-connection {input_file} {attn_out} {post_attn}')

    # 4. Post-attention RMSNorm
    X2 = torch.tensor(np.fromfile(post_attn, dtype=np.int32).reshape(seq_len, embed_dim),
                      dtype=torch.float64, device=0) / (1 << SCALING_LOG)
    rms_inv2 = 1 / torch.sqrt(torch.mean(X2 ** 2, dim=1) + layer.post_attention_layernorm.variance_epsilon)
    save_int(rms_inv2.float(), 1 << SCALING_LOG, 'rms_inv_temp.bin')
    run_cmd(f'./rmsnorm post_attention {post_attn} {seq_len} {embed_dim} {workdir} {lp} {ffn_in}')
    os.remove('rms_inv_temp.bin')

    # 5. FFN
    run_cmd(f'./ffn {ffn_in} {seq_len} {embed_dim} {hidden_dim} {workdir} {lp} {ffn_out}')

    # 6. Skip connection (FFN)
    run_cmd(f'./skip-connection {post_attn} {ffn_out} {output_file}')

    for f in [attn_in, attn_out, post_attn, ffn_in, ffn_out]:
        if os.path.exists(f):
            os.remove(f)


def forward_pass(token_ids, model, embed_dim, hidden_dim, num_layers, workdir):
    seq_len = token_ids.shape[1]
    padded = padded_seq_len(seq_len, embed_dim)

    with torch.no_grad():
        embedding = model.model.embed_tokens(token_ids).squeeze(0).float()
    if padded > seq_len:
        embedding = torch.cat([embedding, torch.zeros(padded - seq_len, embed_dim)], dim=0)
    save_int(embedding, 1 << SCALING_LOG, '_embed.bin')

    for i in range(num_layers):
        inp = '_embed.bin' if i == 0 else f'_layer_out_{i-1}.bin'
        out = f'_layer_out_{i}.bin'
        run_layer(model.model.layers[i], i, padded, embed_dim, hidden_dim, workdir, inp, out)
        if i > 0 and os.path.exists(inp):
            os.remove(inp)

    final_file = f'_layer_out_{num_layers-1}.bin'
    hidden = torch.tensor(
        np.fromfile(final_file, dtype=np.int32).reshape(padded, embed_dim),
        dtype=torch.float32
    ) / (1 << SCALING_LOG)

    os.remove('_embed.bin')
    os.remove(final_file)

    # only the last real token's hidden state is meaningful for next-token prediction
    with torch.no_grad():
        hidden = model.model.norm(hidden[seq_len - 1].unsqueeze(0))
        logits = model.lm_head(hidden)

    return logits[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='zkLLM multi-token generation with LLaMA-2')
    parser.add_argument('input_file', type=str, help='JSON file containing a list of prompts')
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7)
    parser.add_argument('--max_new_tokens', type=int, default=1)
    args = parser.parse_args()

    run_cmd('make all', "Build failed")

    tokenizer, model, embed_dim, hidden_dim, num_layers, workdir = setup()

    with open(args.input_file) as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        print('Error: JSON file must contain a list of strings'); sys.exit(1)

    print(f'\nRunning inference for {len(prompts)} prompt(s)...')
    results = []
    for idx, text in enumerate(prompts):
        print(f'\n--- Input {idx + 1}/{len(prompts)} ---')
        token_ids = tokenizer(text, return_tensors='pt')['input_ids']
        print(f'Prompt ({token_ids.shape[1]} tokens): {text}')
        print('Generating: ', end='', flush=True)

        zk_logits = None
        generated = []
        original_token_ids = token_ids
        for step in range(args.max_new_tokens):
            print(f'[step {step+1}]', end=' ', flush=True)
            zk_logits = forward_pass(token_ids, model, embed_dim, hidden_dim, num_layers, workdir)

            next_id = zk_logits.argmax().item()
            if next_id == tokenizer.eos_token_id:
                print('<eos>')
                break

            generated.append(next_id)
            token_ids = torch.cat([token_ids, torch.tensor([[next_id]])], dim=1)
            print(tokenizer.decode([next_id]), end='', flush=True)

        output_text = tokenizer.decode(generated)

        print(f'\n\nFull output: {text}{output_text}')

        # Compare with base model
        model.cpu()
        with torch.no_grad():
            base_output = model(original_token_ids)  

            base_logits = base_output.logits[0, -1]
            kl = F.kl_div(F.log_softmax(zk_logits, dim=-1), F.softmax(base_logits, dim=-1), reduction='sum').item()
            
            results.append({
                'input': text,
                'zk_output': output_text,
                'base_output': tokenizer.decode([base_logits.argmax().item()]),
                'kl_div': kl,
                'zk_logits': zk_logits.tolist() if zk_logits is not None else [],
                'base_logits': base_logits.tolist(),
            })
        
    with open('output.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults written to output.json')
