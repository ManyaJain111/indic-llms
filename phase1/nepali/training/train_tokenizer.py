#!/usr/bin/env python3
"""
Byte-level BPE tokenizer trainer, fixed for large (500M+ token) corpora.

Key differences from the naive version:
  - Streams the corpus instead of loading it into memory.
  - Incremental pair-count updates: each merge only touches the words that
    actually contain the merged pair, not the whole vocabulary. This is the
    difference between O(merges * corpus_words) and O(merges * avg_affected)
    -- the former does not finish in reasonable time at this scale.
  - Ordered merge list (list, not dict) so encode() applies merges in the
    order they were learned -- required for correct BPE encoding.
  - Actual encode()/decode() methods.
  - Frequency floor to drop hapax words from pair statistics (they add cost,
    not signal, at 500M-token scale).
"""
import json, re, collections, time, argparse, string, pickle, heapq
from pathlib import Path
from tqdm import tqdm

PUNCT = re.escape(string.punctuation + "।॥‘’“”")
WORD_PAT = re.compile(rf" ?[^{PUNCT}\s]+| ?[{PUNCT}]+", re.UNICODE)


class CustomBPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.merges = []  # ordered list of ((a, b), merged_str), earliest first
        self.vocab = {}
        self.byte_to_char = {i: chr(i) for i in range(256)}
        self.char_to_byte = {v: k for k, v in self.byte_to_char.items()}
        self.merge_ranks = {}

    def _word_to_symbols(self, w):
        return [self.byte_to_char[b] for b in w.encode("utf-8")]

    def train(self, corpus_iter, target_vocab_size=None, min_freq=2):
        target_vocab_size = target_vocab_size or self.vocab_size

        # --- 1. word frequencies (streamed) ---
        word_freqs = collections.Counter()
        for line in tqdm(corpus_iter, desc="Counting words"):
            word_freqs.update(WORD_PAT.findall(line))
        print(f"Unique words: {len(word_freqs):,}")

        words, freqs = [], []
        for w, f in word_freqs.items():
            if f < min_freq:
                continue
            words.append(self._word_to_symbols(w))
            freqs.append(f)
        del word_freqs
        print(f"Words kept after min_freq={min_freq}: {len(words):,}")

        # --- 2. initial pair statistics ---
        pair_counts = collections.defaultdict(int)
        pair_index = collections.defaultdict(set)  # pair -> set of word indices
        for idx, word in enumerate(words):
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                pair_counts[p] += freqs[idx]
                pair_index[p].add(idx)

        # --- 3. incremental merges ---
        num_merges = max(0, target_vocab_size - 256)
        pbar = tqdm(total=num_merges, desc="BPE merges")
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        for m in range(num_merges):
            while heap:
                neg_count, best = heapq.heappop(heap)
                if -neg_count == pair_counts.get(best, 0):
                    break
            else:
                print("No pairs left to merge, stopping early.")
                break
            
            if -neg_count < 2:
                break
                
            merged_tok = best[0] + best[1]
            self.merges.append((best, merged_tok))

            affected = list(pair_index.pop(best, ()))
            pair_counts.pop(best, None)
            changed_pairs = set()

            for idx in affected:
                word, freq = words[idx], freqs[idx]

                # remove old pair contributions
                for i in range(len(word) - 1):
                    p = (word[i], word[i + 1])
                    pair_counts[p] -= freq
                    pair_index[p].discard(idx)
                    changed_pairs.add(p)
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                        pair_index.pop(p, None)

                # apply merge
                new_word, i = [], 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best[0] and word[i + 1] == best[1]:
                        new_word.append(merged_tok)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                words[idx] = new_word

                # add new pair contributions
                for i in range(len(new_word) - 1):
                    p = (new_word[i], new_word[i + 1])
                    pair_counts[p] += freq
                    pair_index[p].add(idx)
                    changed_pairs.add(p)

            for p in changed_pairs:
                if p in pair_counts:
                    heapq.heappush(heap, (-pair_counts[p], p))

            pbar.update(1)
            if (m + 1) % 500 == 0:
                pbar.write(f"Merge {m + 1}/{num_merges}: {best} -> {merged_tok!r}")
        pbar.close()

        # --- 4. final vocab ---
        self.vocab = {chr(i): i for i in range(256)}
        next_id = 256
        for _, merged in self.merges:
            if merged not in self.vocab:
                self.vocab[merged] = next_id
                next_id += 1
        print(f"Final vocab size: {len(self.vocab):,}")
        self.merge_ranks = {pair: i for i, (pair, _) in enumerate(self.merges)}

    def encode(self, text):
        ids = []
        for w in WORD_PAT.findall(text):
            symbols = self._word_to_symbols(w)
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                ranked = [(self.merge_ranks[p], i) for i, p in enumerate(pairs) if p in self.merge_ranks]
                if not ranked:
                    break
                _, i = min(ranked)
                symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
            ids.extend(self.vocab.get(s, -1) for s in symbols)
        return ids

    def decode(self, ids):
        id_to_tok = {v: k for k, v in self.vocab.items()}
        byte_arr = bytearray()
        for i in ids:
            for ch in id_to_tok.get(i, ""):
                byte_arr.append(self.char_to_byte[ch])
        return bytes(byte_arr).decode("utf-8", errors="replace")

    def save(self, out_path):
        data = {
            "vocab_size": self.vocab_size,
            "merges": [[a, b] for (a, b), _ in self.merges],
            "vocab": self.vocab,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_pkl(self, out_path):
        with open(out_path, "wb") as f:
            pickle.dump(self, f)

    def load(self, in_path):
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.vocab_size = data.get("vocab_size", len(self.vocab))
        self.merges = [((a, b), a + b) for a, b in data["merges"]]
        self.merge_ranks = {pair: i for i, (pair, _) in enumerate(self.merges)}


def iter_corpus(paths, max_lines=None):
    n = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if max_lines is not None and n >= max_lines:
                    return
                line = line.strip()
                if line:
                    yield line
                n += 1


def evaluate_tokenizer(tokenizer, test_file, max_lines=None):
    total_tokens = total_words = total_characters = unknown_tokens = 0
    token_frequencies = collections.Counter()
    examples = []
    for line in iter_corpus([test_file], max_lines=max_lines):
        ids = tokenizer.encode(line)
        total_tokens += len(ids)
        total_words += len(line.split())
        total_characters += len(line)
        unknown_tokens += ids.count(-1)
        token_frequencies.update(ids)
        if len(examples) < 3:
            examples.append({"text": line, "token_ids": ids, "decoded": tokenizer.decode(ids)})

    id_to_token = {token_id: token for token, token_id in tokenizer.vocab.items()}
    return {
        "evaluation_split": str(test_file),
        "evaluation_max_lines": max_lines,
        "vocab_size": len(tokenizer.vocab),
        "total_words": total_words,
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "average_characters_per_token": total_characters / total_tokens if total_tokens else 0,
        "average_tokens_per_word": total_tokens / total_words if total_words else 0,
        "unknown_tokens": unknown_tokens,
        "unknown_token_rate": unknown_tokens / total_tokens if total_tokens else 0,
        "top_token_frequencies": [
            {"id": token_id, "token": id_to_token.get(token_id, "<UNK>"), "count": count}
            for token_id, count in token_frequencies.most_common(50)
        ],
        "examples": examples,
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=base_dir / "datasets")
    ap.add_argument("--pattern", default="train.txt", help="glob for training files inside corpus-dir")
    ap.add_argument("--out-dir", type=Path, default=base_dir / "models")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-lines", type=int, default=None, help="debug cap; omit for full corpus")
    ap.add_argument("--test-file", type=Path, default=base_dir / "datasets" / "test.txt")
    ap.add_argument("--model", type=Path, default=base_dir / "models" / "custom_tokenizer.json")
    ap.add_argument("--vocab-output", type=Path, default=base_dir / "models" / "nepali_vocabulary.json")
    ap.add_argument("--report-output", type=Path, default=base_dir / "reports" / "nepali_tokenizer_statistics.json")
    ap.add_argument("--evaluate-only", action="store_true", help="Evaluate an existing model without retraining")
    ap.add_argument("--max-eval-lines", type=int, default=10000, help="Held-out line cap; 0 evaluates all lines")
    args = ap.parse_args()

    # Append vocab size to filenames automatically
    args.model = args.model.with_name(f"{args.model.stem}_{args.vocab_size}{args.model.suffix}")
    args.vocab_output = args.vocab_output.with_name(f"{args.vocab_output.stem}_{args.vocab_size}{args.vocab_output.suffix}")
    args.report_output = args.report_output.with_name(f"{args.report_output.stem}_{args.vocab_size}{args.report_output.suffix}")
    model_pkl = args.model.with_suffix(".pkl")

    tokenizer = CustomBPETokenizer(vocab_size=args.vocab_size)
    if args.evaluate_only:
        tokenizer.load(args.model)
    else:
        files = sorted(args.corpus_dir.glob(args.pattern))
        if not files:
            raise SystemExit(f"No files matching {args.pattern} in {args.corpus_dir}")
        print(f"Training on {len(files)} file(s): {[f.name for f in files]}")
        args.out_dir.mkdir(exist_ok=True, parents=True)
        start = time.time()
        tokenizer.train(iter_corpus(files, max_lines=args.max_lines), min_freq=args.min_freq)
        print(f"Training completed in {time.time() - start:.2f}s")
        tokenizer.save(args.model)
        tokenizer.save_pkl(model_pkl)
        print(f"Saved tokenizer to {args.model} and {model_pkl}")

    args.vocab_output.parent.mkdir(exist_ok=True, parents=True)
    args.vocab_output.write_text(json.dumps(tokenizer.vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    max_eval_lines = None if args.max_eval_lines == 0 else args.max_eval_lines
    metrics = evaluate_tokenizer(tokenizer, args.test_file, max_lines=max_eval_lines)
    args.report_output.parent.mkdir(exist_ok=True, parents=True)
    args.report_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFertility: {metrics['average_tokens_per_word']:.2f}")
    print(f"Unknown-token rate: {metrics['unknown_token_rate']:.4%}")
    print(f"Saved vocabulary to {args.vocab_output}")
    print(f"Saved tokenizer report to {args.report_output}")


if __name__ == "__main__":
    main()
