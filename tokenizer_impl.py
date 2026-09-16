class CustomBPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        self.byte_to_char = {i: chr(i) for i in range(256)}
        self.char_to_byte = {v: k for k, v in self.byte_to_char.items()}
        self.merge_ranks = {}
        self.mode = 'byte'

    def _base_symbols(self, w):
        if self.mode == 'byte':
            return [self.byte_to_char[b] for b in w.encode('utf-8')]
        return list(w)

    def encode(self, text):
        if not hasattr(self, 'cache'):
            self.cache = {}
        ids = []
        for w in WORD_PAT.findall(text):
            if w not in self.cache:
                symbols = self._base_symbols(w)
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
                    byte_arr.extend(ch.encode("utf-8"))
        return bytes(byte_arr).decode("utf-8", errors="replace")

    def load(self, in_path):
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.vocab_size = data.get("vocab_size", len(self.vocab))
        self.merges = [((a, b), a + b) for a, b in data["merges"]]
