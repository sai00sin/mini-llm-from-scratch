# model/mini_gpt.py

import torch
import torch.nn as nn
from model.embedding import EmbeddingWithPosition
from model.attention import SelfAttention

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, head_size):
        super().__init__()
        self.attn = SelfAttention(embed_dim, head_size)
        self.ffn = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim):
        super().__init__()
        self.embed = EmbeddingWithPosition(vocab_size, max_len, embed_dim)
        self.block = TransformerBlock(embed_dim, head_size=embed_dim)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)  # 出力は語彙数に対応（logits）

    def forward(self, idx):
        x = self.embed(idx)        # (B, T, C)
        x = self.block(x)          # (B, T, C)
        x = self.ln_f(x)           # (B, T, C)
        logits = self.head(x)      # (B, T, vocab_size)
        return logits
