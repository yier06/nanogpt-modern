import torch
from torch import nn
from .block import Block

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed_size)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embed_size)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed_size),
            wpe = nn.Embedding(config.block_size, config.n_embed_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embed_size)
        ))
        self.lm_head = nn.Linear(config.n_embed_size, config.vocab_size)
        self.transformer.wte.weight = self.tok_emb.weight

        # init all weights
        self.apply(self._init_weights)

    def forward(self, idx):
        b, t = idx.shape
        device = idx.device

        # token embedding：用 idx（token id）
        tok_emb = self.transformer.wte(idx)

        # position embedding：用 0..t-1（位置 id）
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:   # 不是 self.blocks
            x = block(x)

        x = self.transformer.ln_f(x)
        return self.lm_head(x)              # 返回 logits，generation 才能用
        
    # def forward(self, idx):
    #     batch_size, seq_length = idx.shape
    #     tok_embeds = self.transformer.wte(idx)
        
    #     pos_embeds = self.pos_embed(idx)
    #     token_embeds = tok_embeds + pos_embeds
    #     x = token_embeds
    #     for block in self.blocks:
    #         x = block(x)
        
    #     return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



