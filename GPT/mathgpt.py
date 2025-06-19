import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import json




batch_size = 16
block_size = 1024*3//4
max_iter = 75000
eval_interval = 500
lr = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 256
n_heads = 4
n_layers = 4
dropout = 0.2




tokenizer = tiktoken.encoding_for_model('gpt-2')
special_token_list = ['<QUESTION>', '</QUESTION>', "<ANSWER>", "</ANSWER>", '<|PAD|>']
sp_tokens = {token : tokenizer.n_vocab+i for i, token in enumerate(special_token_list)}
sp_tokens.update(tokenizer._special_tokens)
tokenizer_ = tiktoken.Encoding(
    name="p50k_with_custom",
    pat_str=tokenizer._pat_str,
    mergeable_ranks=tokenizer._mergeable_ranks,
    special_tokens=sp_tokens
)


def encode(text, append_eot = False):
    tokens = tokenizer_.encode(text)
    if append_eot == True:
        tokens.append(50256)
    return tokens

def decode(tokens : list[int]):
    return tokenizer_.decode(tokens)







class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.wq = nn.Linear(n_embed, head_size, bias=False)
        self.wk = nn.Linear(n_embed, head_size, bias=False)
        self.wv = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        affinity = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        affinity = affinity.masked_fill(self.tril[:T, :T]==0, float('-inf')) # fill the future time steps
        affinity = F.softmax(affinity, dim=-1)
        affinity = self.dropout(affinity)

        output = affinity @ v # (B, T, T) @ (B, T, hs) = (B, T, hs)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size*n_heads, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(inplace=True),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.projection(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.mha = MultiHeadAttention(n_head, head_size)
        self.ffd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = self.mha(self.ln1(x)) + x
        x = self.ffd(self.ln2(x)) + x
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(tokenizer_.n_vocab, n_embed) # V, E
        self.positional_embed = nn.Embedding(block_size, n_embed) # T, E
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        self.lnm = nn.LayerNorm(n_embed)
        self.proj = nn.Linear(n_embed, tokenizer_.n_vocab)

        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, inputs, targets = None):
        B, T = inputs.shape
        tok_emb = self.token_embedding(inputs)
        pos_emb = self.positional_embed(torch.arange(T, device = device))

        x = tok_emb + pos_emb # B, T, n_embed

        x = self.blocks(x) # B, T, n_embed

        x = self.lnm(x) # B, T, n_embed

        logits = self.proj(x) # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.reshape(B*T,)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_iterations : int):
        """idx is the current context with B, T dimensions"""
        for _ in range(max_iterations):
            idx_context_limited = idx[:, -block_size:] 
            logits, loss = self(idx_context_limited)
            probs = F.softmax(logits[:, -1, :], dim=-1) # softmax for last time stamp
            # idx_next = torch.argmax(probs, dim=-1, keepdim=True) # B, 1
            # print(idx_next)
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            yield idx_next


model = GPTLanguageModel().to(device)
print(model.load_state_dict(torch.load('smol-math-gpt.pth', map_location=device)))

def solve(question):
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        idxs = model.generate(torch.tensor([tokenizer_.encode(f'<QUESTION>\n{question}\n</QUESTION>', allowed_special={'<QUESTION>', '</QUESTION>'})], device=device), 512)
        for idx in idxs:
            print(tokenizer_.decode([int(idx.item())]), end='')
            if int(idx.item())==50260:
                break
        print('\n================================================================================================================\n')
        # return tokenizer_.decode(idxs[0].tolist())


while True:
    solve(input("Enter your query: "))