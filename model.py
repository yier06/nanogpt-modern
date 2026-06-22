import torch
from torch import nn
import math

class CausalSelfAttention(nn.module):
    def __init__(self, config):
        super().__init__()
        self.head_num = config.n_heads
        self.flash = False
        self.c_atten = nn.Linear(self.c, 3*self.c)
        if not self.flash:
            print("using slow attention compute")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.blcok_size).view(1, 1, config.blcok_size, config.block_size)))

    
    def forward(self, x):
        B, T, C = x.sh
        q, k, v = self.c_atten(x).split(3, dim=2)
        q = q.view(B, T, self.head_num, C//self.head_num).transpose(1, 2)
        k = k.view(B, T, self.head_num, C//self.head_num).transpose(1, 2)
        v = v.view(B, T, self.head_num, C//self.head_num).tramspose(1, 2)

        att = q @ k.transpose(-2, -1) *  (1 // math.sqrt(k.size(-1)))
        attn = att.mask_fill(self.bias[:, :, T, T]==0, float("-inf"))
        attn = attn .softmax(dim=-1)
        attn = attn @ v
        y = attn.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.module):
    def __init__(self, config):
        super.__init__()
        self.c_fc = nn.linear(config.n_embed_size, 4*config.n_embed_size)
        self.gelu =  nn.gelu()
        self.c_proj = nn.linear(4*config.n_embed_size, config.n_embed_size)
        self.dropout = nn.dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.module):
    def __init__(self, config):
        super.__init__()
        self.ln_1 = nn.layer_norm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.layer_norm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x+ self.mlp(self.ln_2(x))
        return x

class GPT(nn.module):
    def __init__(self, config):
        super.__init__()
        self.tok_emb = nn.embeding(config.vocab_size, config.n_embed_size)
        self.pos_embed = nn.embeding(config.block_size, config.n_embed_size)
        self.transformer = nn.moduleDict(dict(
            wte = nn.embeding(config.vocab_size, config.n_embed_size),
            wpe = nn.embeding(config.block_size, config.n_embed_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.layer_norm(config.n_embed_size)
        ))
        self.lm_head = nn.linear(config.n_embed_size, config.vocab_size)
        self.transformer.wte.weight = self.tok_emb.weight

        # init all weights
        self.apply(self._init_weights)


        
    def forward(self, idx):
        batch_size, seq_length = idx.shape
        tok_embeds = self.transformer.wte(idx)
        
        pos_embeds = self.pos_embed(idx)
        token_embeds = tok_embeds + pos_embeds
        x = token_embeds
        for block in self.blocks:
            x = block(x)
        
        return x


