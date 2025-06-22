# model/embedding.py

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.token_embed(x)

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.pos_embed(positions)

class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(max_len, embed_dim)

    def forward(self, x):
        return self.token_embed(x) + self.pos_embed(x)
