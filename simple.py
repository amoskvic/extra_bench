#!/usr/bin/env python3
"""
Place holder for MAMBA SSM performance script
"""

import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-2.8b")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=2048)
parser.add_argument("--genlen", type=int, default=1) #TTFT
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=0.9)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.2)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--dtype", type=str, default='float16')
parser.add_argument("--do_compile", type=bool, default=False)
parser.add_argument("--perf_log", type=str, default='perf_log')
parser.add_argument("--warmup_steps", type=int, default=10)
parser.add_argument("--repeats", type=int, default=10)
args = parser.parse_args()


device = "cuda"

if args.dtype.lower() == 'float16':
    args.dtype = torch.float16
elif args.dtype.lower() == 'bfloat16':
    args.dtype = torch.bfloat16
elif args.dtype.lower() == 'float32':
    args.dtype = torch.float32
else:
    raise ValueError(f'Unsupported dtype {args.dtype}')

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=args.dtype)

model.eval()

input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device=device)
attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

max_length = input_ids.shape[1] + args.genlen
cg = False

torch.random.manual_seed(0)

total = 0

with torch.no_grad():

    if args.do_compile:
        model_cur = torch.compile(model, mode="max-autotune")
    else:
        model_cur = model

    
    fn = lambda: model_cur.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=cg,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=True,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            repetition_penalty=args.repetition_penalty,
        )

    
    for i in range(args.warmup_steps):
        out = fn()

    
    for i in range(args.repeats):
        torch.cuda.empty_cache()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()

        start.record()
        out = fn()
        torch.cuda.synchronize()
        end.synchronize()
        end.record()
        
        torch.cuda.synchronize()
        result = start.elapsed_time(end)
        total += result


print(f"TTFT:{total / args.repeats}")
