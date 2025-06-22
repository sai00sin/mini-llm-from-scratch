# model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.scale = head_size ** -0.5  # Scaled Dot-Product
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time, Channels

        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Attention Scores: (B, T, T)
        att = q @ k.transpose(-2, -1) * self.scale

        # Causal Mask (未来を見ない)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Weighted sum: (B, T, head_size)
        out = att @ v
        return out
