import json, collections, re, string

PUNCT = re.escape(string.punctuation + "।॥‘’“”")
WORD_PAT = re.compile(rf" ?[^{PUNCT}\s]+| ?[{PUNCT}]+", re.UNICODE)

class ByteLevelBPETokenizer:
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
        if not hasattr(self, 'cache'): self.cache = {}
        ids = []
        for w in WORD_PAT.findall(text):
            if w not in self.cache:
                symbols = self._word_to_symbols(w)
                while len(symbols) > 1:
                    pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                    ranked = [(self.merge_ranks[p], i) for i, p in enumerate(pairs) if p in self.merge_ranks]
                    if not ranked: break
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
                try: byte_arr.append(self.char_to_byte[ch])
                except KeyError: byte_arr.extend(ch.encode("utf-8"))
        return bytes(byte_arr).decode("utf-8", errors="replace")

    def load(self, data):
        self.vocab = data["vocab"]
        self.vocab_size = data.get("vocab_size", len(self.vocab))
        self.merges = [((a, b), a + b) for a, b in data["merges"]]
        self.merge_ranks = {pair: i for i, (pair, _) in enumerate(self.merges)}


class UnicodeBPETokenizer:
    def __init__(self, vocab_size=37000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.unk_id = None
        self._merge_ranks = {}
        self._id_to_token = {}

    def _encode_symbols(self, text):
        encoded = []
        for word in WORD_PAT.findall(text):
            symbols = list(word)
            while len(symbols) > 1:
                candidates = [
                    (self._merge_ranks[pair], index)
                    for index, pair in enumerate(zip(symbols, symbols[1:]))
                    if pair in self._merge_ranks
                ]
                if not candidates: break
                _, index = min(candidates)
                symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
            encoded.extend(symbols)
        return encoded

    def encode(self, text):
        return [self.vocab.get(symbol, self.unk_id) for symbol in self._encode_symbols(text)]

    def decode(self, token_ids):
        return "".join(self._id_to_token.get(token_id, "<UNK>") for token_id in token_ids)

    def load(self, data):
        self.vocab = data["vocab"]
        self.vocab_size = data["vocab_size"]
        self.unk_id = data.get("unk_id", 0)
        self.merges = [((left, right), left + right) for left, right in data["merges"]]
        self._merge_ranks = {pair: rank for rank, (pair, _) in enumerate(self.merges)}
        self._id_to_token = {token_id: token for token, token_id in self.vocab.items()}

class TokenizerWrapper:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if data.get("format") == "custom_unicode_bpe_v1":
            self.tokenizer = UnicodeBPETokenizer()
        else:
            self.tokenizer = ByteLevelBPETokenizer()
            
        self.tokenizer.load(data)
        
    def encode(self, text):
        return self.tokenizer.encode(text)
        
    def decode(self, ids):
        return self.tokenizer.decode(ids)
        
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

# Test it works on Nepali
tok = TokenizerWrapper("repo/nepali/tokenizer/tokenizer.json")
encoded = tok.encode("तार्किक सम्बन्ध अनुसार, मानौं वीरगन्ज को प्रतिव्यक्ति आय 352 हजार रुपैयाँ छ र भक्तपुर को प्रतिव्यक्ति आय 226 हजार रुपैयाँ छ। बढी प्रतिव्यक्ति आय कसको छ?\\nउत्तर: ")
print("Tokens length:", len(encoded))
print("Decoded string:", repr(tok.decode(encoded)))

