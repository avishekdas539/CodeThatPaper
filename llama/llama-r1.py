import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import tiktoken
import streamlit as st

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = tiktoken.encoding_for_model('gpt-2')
special_token_list = ['<|im_start|>', '<|im_end|>', "user", "reasoning", 'assistant', '<|PAD|>']
sp_tokens = {token : model.n_vocab+i for i, token in enumerate(special_token_list)}
sp_tokens.update(model._special_tokens)
model = tiktoken.Encoding(
    name="p50k_with_custom",
    pat_str=model._pat_str,
    mergeable_ranks=model._mergeable_ranks,
    special_tokens=sp_tokens
)


def encode(text, append_eot = False):
    tokens = model.encode(text, allowed_special = set(model._special_tokens.keys())) # forcefully allowing every special tokens
    if append_eot == True:
        tokens.append(50256)
    return tokens

def decode(tokens : list[int]):
    return model.decode(tokens)
@dataclass
class ModelArgs:
    vocab_size : int = model.n_vocab
    max_seq_len : int = 1280
    model_dim : int = 768
    padding_idx : int = 50262
    num_hidden_layers : int = 6
    intermediate_dim: int = 768
    n_kv_heads: int = 4
    n_head: int = 8
    rms_norm_eps : float = 1e-6
    bias : bool = False
    lr : float = 8e-4

class ReasoningDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i]['template']

    def __len__(self):
        return self.data.num_rows
def collate_fn(batch : list[str], max_seq_len : int = 1024, pad_id : int = 50262):
    batch_tokens = []
    for text in batch:
        tokens = encode(text, True)[:max_seq_len+1]
        token_len = len(tokens)
        pad_len = max(0, max_seq_len+1-token_len)
        if pad_len:
            tokens = tokens + [pad_id] * pad_len
        batch_tokens.append(tokens)
    return torch.tensor(batch_tokens, dtype=torch.long).to(device)
        
def calculate_mask(batch_x : torch.Tensor, pad_id : int = 50262):
    B, T = batch_x.shape
    causal_mask = torch.tril(torch.ones(1, T, T)).to(device)
    pad_mask = (batch_x!=pad_id).to(device)
    key_mask = pad_mask[:, None, :] # B, 1, T
    query_mask = pad_mask[:, :, None] # B, T, 1
    final_mask = causal_mask  * key_mask * query_mask
    return final_mask.to(device)

# batch[3].tolist()[800:]
# batch.shape
# mask = calculate_mask(batch)
# mask.shape
# plt.imshow(mask[1].detach().cpu().numpy())
class RMSNorm(nn.Module):
    def __init__(self, dim : int, eps : float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(data=torch.ones(dim))
    def forward(self, x : torch.Tensor):
        variance = x.pow(2).mean(dim=-1, keepdims=True).type_as(x)
        return x * torch.rsqrt(variance+self.eps) * self.weight # rsqrt is same as 1/sqrt
class RoPE(nn.Module):
    def __init__(self, max_seq_len : int = 1024, d : int = 256, k : float = 10000.0, device : str = 'cpu'):
        super().__init__()
        self.d = d
        self.max_seq_len = max_seq_len
        self.device=device
        freqs, sin, cos = self.precompute_freqs(k=k)
        self.register_buffer('freqs', freqs.to(device))
        self.register_buffer('sin', sin.to(device))
        self.register_buffer('cos', cos.to(device))

    @torch.no_grad()
    def precompute_freqs(self, k : float = 10000.0):
        theta = 1/(k**(torch.arange(0, self.d, 2.0)/self.d))
        pos = torch.arange(self.max_seq_len).unsqueeze(1)
        freqs = pos*theta
        cos = torch.cos(freqs).to(self.device)
        sin = torch.sin(freqs).to(self.device)
        # print(theta.shape, pos.shape, freqs.shape)
        # print(theta, pos, freqs, sin, cos)
        return freqs, sin, cos
    def apply_rope(self, x : torch.Tensor):
        """Assumes x to be B, H, T,D"""
        B, H, T, D = x.shape
        x_reshaped = x.view(*x.shape[:-1], self.d//2, 2)
        x1 = x_reshaped[...,0]
        x2 = x_reshaped[...,1]

        cos = self.cos[:T, ...]
        sin = self.sin[:T, ...]
        stacked = torch.stack([x1 * cos - x2 * sin, 
                              x1 * sin + x2 * cos], dim=-1) # stack on last dimension
        out = stacked.view(x.shape)
        return out
    def forward(self, x : torch.Tensor):
        return self.apply_rope(x)
class LlamaMLP(nn.Module):
    def __init__(self, dim : int = 256, intermediate_dim : int = 256, bias : bool = True):
        super(LlamaMLP, self).__init__()
        self.d = dim
        self.intermediate_dim = intermediate_dim
        self.gate = nn.Linear(dim, intermediate_dim, bias=bias)
        self.up = nn.Linear(dim, intermediate_dim, bias=bias)
        self.down = nn.Linear(intermediate_dim, dim, bias=bias)
        self.activation_fn = F.silu

    def forward(self, x : torch.Tensor):
        # SwigLU(q, b) = SiLU(a) * b
        # final layer is W*(SwiGLU(x)) = W * (SiLU(x) * (W*x))
        return self.down(self.activation_fn(self.gate(x)) * self.up(x))
def repeat_kv(module : nn.Module, x : torch.Tensor, n_reps : int):
    B, H, T, D = x.shape
    if n_reps == 1:
        return x
    else:
        return x[:, :, None, :, :].expand(B, H, n_reps, T, D).reshape(B, H*n_reps, T, D)
class LlamaAttention(nn.Module):
    def __init__(self, dim : int = 256, n_kv_heads : int = 4, n_head : int = 8, max_seq_len : int = 1024):
        super(LlamaAttention, self).__init__()
        self.dim = dim
        self.n_kv_heads = n_kv_heads
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.w_q = nn.Linear(dim, self.head_dim * self.n_head, bias=False)
        self.w_k = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
        self.w_v = nn.Linear(dim, self.head_dim * self.n_kv_heads, bias=False)
        self.w_o = nn.Linear(self.head_dim * self.n_head, dim, bias=False)
        self.rotary_embedding = RoPE(max_seq_len = max_seq_len, d = self.head_dim)
    
    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        """mask is filled with -inf at the position where the attn to be ignored
        x is of shape, B, T, D
        mask is of shape B, T, T"""
        B, T, D = x.shape
        
        # make all as shape B, H, T, head_dim
        Q = self.w_q(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        K = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2) # on top of this repeatations are needed
        V = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2) # on top of this repeatations are needed

        Q = self.rotary_embedding(Q)
        K = self.rotary_embedding(K)
        
        n_reps = self.n_head // self.n_kv_heads
        K = repeat_kv(self, K, n_reps)
        V = repeat_kv(self, V, n_reps) 

        attn_scores = torch.matmul(Q, K.transpose(2,3)) / math.sqrt(self.head_dim) # B, H, T, D * B, H, D, T => B, H, T, T
        if mask is not None:
            mask = mask[:, :T, :T] # forcefully making of time steps equal to x
            attn_scores += attn_scores.masked_fill(mask.unsqueeze(1)==0, -1e9)
        
        attn_scores = F.softmax(attn_scores.float(), dim=-1)
        attn_output = torch.matmul(attn_scores, V) # B, H, T, T * B, H, T, D => B, H, T, D
        
        attn_output = attn_output.transpose(1,2)
        attn_output = attn_output.reshape(B, T, -1).contiguous() # reshape back to B, T, D from B, H, T, D
        
        attn_output = self.w_o(attn_output)
        return attn_output, attn_scores
# attn = LlamaAttention()
def count_params(module):
    total = 0
    for p in module.parameters():
        v = 1
        for d in p.shape:
            v *= d
        total += v
    return total
# x = torch.rand(32, 1024, 256)
# y, att_s = attn(x)
# y.shape, att_s.shape
class LlamaDecoder(nn.Module):
    def __init__(self, hidden_dim : int = 256, intermediate_dim : int = 256, n_kv_heads : int = 4, n_head : int = 8, max_seq_len : int = 1024):
        super(LlamaDecoder, self).__init__()
        self.rms_norm = RMSNorm(dim = hidden_dim) # eps needed
        self.self_attn = LlamaAttention(dim = hidden_dim, n_kv_heads = n_kv_heads, n_head = n_head, max_seq_len = max_seq_len)
        self.mlp = LlamaMLP(dim = hidden_dim, intermediate_dim = intermediate_dim) # bias needed
        self.hidden_dim = hidden_dim
        self.n_kv_heads = n_kv_heads
        self.n_head = n_head

    def forward(self, hidden_states : torch.Tensor, mask : torch.Tensor = None):
        """hidden_steps of shape B, T, D"""
        # print(torch.isnan(hidden_states).any())
        state = self.rms_norm(hidden_states)
        # print(torch.isnan(state).any())
        attn_output, attn_scores = self.self_attn(state, mask)
        # print(torch.isnan(attn_output).any())

        hidden_states = hidden_states + attn_output
        # print(torch.isnan(hidden_states).any())
        

        state = self.rms_norm(hidden_states)
        # print(torch.isnan(state).any())
        
        state = self.mlp(state)
        # print(torch.isnan(state).any())

        hidden_states = hidden_states + state
        # print(torch.isnan(hidden_states).any())
        # print('========================================')
        return hidden_states
# decoder = LlamaDecoder()
# count_params(decoder)
# y = decoder(x)
# y.shape
class Llama(nn.Module):
    def __init__(self, config : ModelArgs):
        super(Llama, self).__init__()
        self.config = config
        self.embedding_table = nn.Embedding(num_embeddings = config.vocab_size, embedding_dim = config.model_dim, padding_idx = config.padding_idx)
        self.decoder_layers = nn.ModuleList([
                                                LlamaDecoder(hidden_dim = config.model_dim, intermediate_dim = config.intermediate_dim,
                                                          n_kv_heads = config.n_kv_heads, n_head = config.n_head, max_seq_len = config.max_seq_len)
                                                for _ in range(config.num_hidden_layers)
                                            ])
        self.rms_norm = RMSNorm(dim = config.model_dim, eps = config.rms_norm_eps)
        self.mlp = LlamaMLP(dim = config.model_dim, intermediate_dim = config.intermediate_dim, bias = config.bias)
        self.proj_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        """X is in shape B, T
        mask is in shape B, T, T"""
        state = self.embedding_table(x)
        for dec_layer in self.decoder_layers:
            state = dec_layer(state, mask)
        
        state = self.rms_norm(state)
        state = self.mlp(state)
        state = self.proj_head(state)
        return state
    
    def apply_rep_penalty(self, logits : torch.Tensor, gen_tokens : torch.Tensor, penalty : float):
        """logits => 1, D and gen_tokens 1, T"""
        for token_id in gen_tokens[0].tolist():
            if logits[0][token_id]>0:
                logits[0][token_id] /= penalty
            else:
                logits[0][token_id] *= penalty
        return logits
    
    def generate(self, x : torch.Tensor, max_token : int = 1024, temp = 0.7, rep_penalty : float = 1.5):
        """x is of shape B, T"""
        for _ in range(max_token):
            x = x[:, -self.config.max_seq_len:]
            mask = calculate_mask(x, pad_id = 50262)
            state = self.embedding_table(x)
            for dec_layer in self.decoder_layers:
                state = dec_layer(state, mask)
            
            state = self.rms_norm(state)
            
            last_step_pred = state[:, -1, :] # B, 1, Vocab_size
            last_step_pred = self.mlp(last_step_pred)
            last_step_pred = self.proj_head(last_step_pred)
            if temp==0:
                idx_next = torch.argmax(last_step_pred, dim=-1, keepdim=True)
            else:
                probs = F.softmax(last_step_pred/temp, dim=-1)
                if rep_penalty>0:
                    probs = self.apply_rep_penalty(probs, x, penalty=rep_penalty)
                idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1) # (B, T+1)
            yield idx_next

llama_model = Llama(config = ModelArgs).to(device)
print(f"Model size: {count_params(llama_model)/10**6}M parameters")


def answer(llama_model, question, max_token=1280, temp = 0.7, rep_penalty = 1.5,  end = '|'):
    f_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>reasoning"
    tokens = encode(f_text)
    print(f_text, end='')
    llama_model.eval()
    for idx in llama_model.generate(torch.tensor([tokens], dtype=torch.long, device=device), max_token=1280, temp = temp, rep_penalty=rep_penalty):
        token = decode(idx[0].tolist())
        if token == '<|endoftext|>':
            break
        print(token, end=end)

state_dict = torch.load('model_15.pth', map_location=torch.device(device)) # Load to CPU
llama_model_ = Llama(config = ModelArgs).to(device)
print(llama_model_.load_state_dict(state_dict))

x = input('Ask question: ')
while x!='exit':
    answer(llama_model_, x, end='', temp=0.5, rep_penalty=1.1)