# tokenization/tokenizer.py

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])


if __name__ == "__main__":
    with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    sample = "To be or not to be"

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("Sample Input: ", sample)
    print("Encoded:      ", encoded)
    print("Decoded:      ", decoded)
    print("Vocab size:   ", tokenizer.vocab_size)