# attention_test.py（ルートに作成してもOK）

from model.attention import SelfAttention
import torch

B, T, C = 1, 8, 16  # バッチ1, シーケンス長8, 埋め込み次元16
x = torch.randn(B, T, C)

attention = SelfAttention(embed_dim=C, head_size=8)
y = attention(x)

print("Output shape:", y.shape)  # → (1, 8, 8)
