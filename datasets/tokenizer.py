import numpy as np

class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.id2w = [w.strip() for w in f.readlines()]
        self.id2w = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'] + self.id2w
        self.w2id = {w: id for id, w in enumerate(self.id2w)}
        self.reserved_ids = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.reserved_ids = {x: self.w2id[x] for x in self.reserved_ids}

    def encode(self, s):
        s = [self.w2id[w] if w in self.w2id else self.reserved_ids['<UNK>'] for w in s.split()]
        return np.array(s)

    def decode(self, ids):
        s = ' '.join([self.id2w[id] for id in ids])
        return s