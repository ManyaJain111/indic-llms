import json
import collections
import re
import string

PUNCT = re.escape(string.punctuation + "।॥‘’“”")
WORD_PAT = re.compile(rf" ?[^{PUNCT}\s]+| ?[{PUNCT}]+", re.UNICODE)

class CustomBPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.merges = []  
        self.vocab = {}
        self.byte_to_char = {i: chr(i) for i in range(256)}
        self.char_to_byte = {v: k for k, v in self.byte_to_char.items()}
        self.merge_ranks = {}

    def _word_to_symbols(self, w):
        return [self.byte_to_char[b] for b in w.encode("utf-8")]

    def encode(self, text):
        if not hasattr(self, 'cache'):
            self.cache = {}
        ids = []
        for w in WORD_PAT.findall(text):
            if w not in self.cache:
                symbols = self._word_to_symbols(w)
                while len(symbols) > 1:
                    pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                    ranked = [(self.merge_ranks[p], i) for i, p in enumerate(pairs) if p in self.merge_ranks]
                    if not ranked:
                        break
                    _, i = min(ranked)
                    symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
                self.cache[w] = [self.vocab.get(s, 0) for s in symbols]
            ids.extend(self.cache[w])
        return ids

    def decode(self, ids):
        if getattr(self, 'id_to_tok', None) is None:
            self.id_to_tok = {v: k for k, v in self.vocab.items()}
            
        byte_arr = bytearray()
        for i in ids:
            for ch in self.id_to_tok.get(i, ""):
                try:
                    byte_arr.append(self.char_to_byte[ch])
                except KeyError:
                    # For Character-Level BPEs (like Nepali) that don't map to bytes
                    byte_arr.extend(ch.encode("utf-8"))
        
        # If byte_arr was built from utf-8 bytes (Hindi), decode it.
        # If it was built directly from characters (Nepali), it's already utf-8 bytes so decode still works!
        return bytes(byte_arr).decode("utf-8", errors="replace")

    def load(self, in_path):
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.vocab_size = data.get("vocab_size", len(self.vocab))
        self.merges = [((a, b), a + b) for a, b in data["merges"]]
        self.merge_ranks = {pair: i for i, (pair, _) in enumerate(self.merges)}

class TokenizerWrapper:
    def __init__(self, json_path):
        self.tokenizer = CustomBPETokenizer()
        self.tokenizer.load(json_path)
        
    def encode(self, text):
        return self.tokenizer.encode(text)
        
    def decode(self, ids):
        return self.tokenizer.decode(ids)
        
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
