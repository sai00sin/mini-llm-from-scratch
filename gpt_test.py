# gpt_test.py（プロジェクトルートに作成）

from model.mini_gpt import MiniGPT
import torch

vocab_size = 65
max_len = 32
embed_dim = 16

model = MiniGPT(vocab_size, max_len, embed_dim)

# 入力トークン（バッチ1、長さ32）
x = torch.randint(0, vocab_size, (1, max_len))
logits = model(x)

print("Input shape: ", x.shape)         # torch.Size([1, 32])
print("Logits shape:", logits.shape)    # torch.Size([1, 32, 65])
