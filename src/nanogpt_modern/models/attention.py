import torch
from torch import nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_num = config.n_heads
        self.flash = False
        self.c = config.n_embed_size
        self.c_proj = nn.Linear(self.c, self.c)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.c_atten = nn.Linear(self.c, 3*self.c)
        if not self.flash:
            print("using slow attention compute")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

    
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_atten(x).split(self.c, dim=2)
        q = q.view(B, T, self.head_num, C//self.head_num).transpose(1, 2)
        k = k.view(B, T, self.head_num, C//self.head_num).transpose(1, 2)
        v = v.view(B, T, self.head_num, C//self.head_num).transpose(1, 2)

        att = q @ k.transpose(-2, -1) *  (1.0 / math.sqrt(k.size(-1)))
        attn = att.masked_fill(self.bias[:, :, T, T]==0, float("-inf"))
        attn = attn .softmax(dim=-1)
        attn = attn @ v
        y = attn.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y
