
import math
import os
import typing
import torch.nn.functional as F

import entmax
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoTokenizer, TrainerCallback, \
    LlamaConfig, Cache
from datasets import load_dataset
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

def perform_attn(query, key, value, input_shape, attn_mask,scale, softmax_fn, o_proj, training, dropout_p, is_causal=True, enable_gqa=False):
    # [batch, num_heads, seq_len, head_dim]
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device='mps')
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(device='mps')
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = softmax_fn(attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

perform_attn(torch.ones([1, 32, 512, 64]).to(device='mps'), torch.ones([1, 8, 512, 64]).to(device='mps'), torch.ones([1, 8, 512, 64]).to(device='mps'), torch.Size(torch.tensor([1, 512])),
            None, 0.125, lambda x: torch.nn.functional.softmax(x, dim=-1), torch.nn.Linear(2048, 2048).to(device='mps'),
             True,
             0.2, True, True)