# embedding_test.py

from model.embedding import EmbeddingWithPosition
import torch

vocab_size = 65       # tokenizer.vocab_size と一致
max_len = 32          # 仮の入力シーケンス長
embed_dim = 16        # 埋め込み次元

# 埋め込みモデルを初期化
model = EmbeddingWithPosition(vocab_size, max_len, embed_dim)

# ダミーのトークン列（batch=1, seq_len=32）
dummy_input = torch.randint(0, vocab_size, (1, max_len))

# 順伝播（forward）
output = model(dummy_input)

print("Input shape :", dummy_input.shape)
print("Output shape:", output.shape)
