#!/usr/bin/env python3
"""Train a Unicode character-level BPE tokenizer without external libraries."""

import argparse
import collections
import heapq
import json
import re
import string
import time
from pathlib import Path

from tqdm import tqdm


PUNCTUATION = re.escape(string.punctuation + "।॥‘’“”")
WORD_PATTERN = re.compile(rf" ?[^{PUNCTUATION}\s]+| ?[{PUNCTUATION}]+", re.UNICODE)
MAX_VOCAB_SIZE = 40000


class CustomBPETokenizer:
    def __init__(self, vocab_size=37000):
        if not 1 < vocab_size <= MAX_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be between 2 and {MAX_VOCAB_SIZE:,}")
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.unk_id = None
        self._merge_ranks = {}
        self._id_to_token = {}

    def _refresh_indexes(self):
        self._merge_ranks = {pair: rank for rank, (pair, _) in enumerate(self.merges)}
        self._id_to_token = {token_id: token for token, token_id in self.vocab.items()}

    @staticmethod
    def _merge_pair(symbols, pair, merged_symbol):
        merged, index = [], 0
        while index < len(symbols):
            if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                merged.append(merged_symbol)
                index += 2
            else:
                merged.append(symbols[index])
                index += 1
        return merged

    def train(self, texts, min_frequency=2, max_word_types=200_000):
        word_frequencies = collections.Counter()
        for text in tqdm(texts, desc="Counting words"):
            word_frequencies.update(WORD_PATTERN.findall(text))

        retained = [
            (word, frequency)
            for word, frequency in word_frequencies.most_common(max_word_types)
            if frequency >= min_frequency
        ]
        characters = sorted({character for word, _ in retained for character in word})
        self.vocab = {character: index for index, character in enumerate(characters)}
        self.unk_id = len(self.vocab)
        self.vocab["<UNK>"] = self.unk_id
        print(f"Unique words: {len(word_frequencies):,}; BPE training words: {len(retained):,}")

        words = [list(word) for word, _ in retained]
        frequencies = [frequency for _, frequency in retained]
        pair_counts = collections.defaultdict(int)
        pair_to_words = collections.defaultdict(set)
        for word_id, symbols in enumerate(words):
            for index, pair in enumerate(zip(symbols, symbols[1:])):
                pair_counts[pair] += frequencies[word_id]
                pair_to_words[pair].add(word_id)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        merge_budget = max(0, self.vocab_size - len(self.vocab))
        progress = tqdm(total=merge_budget, desc="Learning BPE merges")

        for _ in range(merge_budget):
            while heap:
                negative_count, best_pair = heapq.heappop(heap)
                if -negative_count == pair_counts.get(best_pair, 0):
                    break
            else:
                break
            if -negative_count < min_frequency:
                break

            merged_symbol = best_pair[0] + best_pair[1]
            self.merges.append((best_pair, merged_symbol))
            affected_word_ids = list(pair_to_words.pop(best_pair, ()))
            pair_counts.pop(best_pair, None)
            changed_pairs = set()

            for word_id in affected_word_ids:
                old_symbols = words[word_id]
                frequency = frequencies[word_id]
                for pair in zip(old_symbols, old_symbols[1:]):
                    pair_counts[pair] -= frequency
                    pair_to_words[pair].discard(word_id)
                    changed_pairs.add(pair)
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                        pair_to_words.pop(pair, None)

                new_symbols = self._merge_pair(old_symbols, best_pair, merged_symbol)
                words[word_id] = new_symbols
                for pair in zip(new_symbols, new_symbols[1:]):
                    pair_counts[pair] += frequency
                    pair_to_words[pair].add(word_id)
                    changed_pairs.add(pair)

            for pair in changed_pairs:
                count = pair_counts.get(pair)
                if count:
                    heapq.heappush(heap, (-count, pair))
            progress.update(1)
        progress.close()

        for _, merged_symbol in self.merges:
            if merged_symbol not in self.vocab:
                self.vocab[merged_symbol] = len(self.vocab)
        self._refresh_indexes()
        print(f"Final vocabulary size: {len(self.vocab):,}")

    def _encode_symbols(self, text):
        encoded = []
        for word in WORD_PATTERN.findall(text):
            symbols = list(word)
            while len(symbols) > 1:
                candidates = [
                    (self._merge_ranks[pair], index)
                    for index, pair in enumerate(zip(symbols, symbols[1:]))
                    if pair in self._merge_ranks
                ]
                if not candidates:
                    break
                _, index = min(candidates)
                symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
            encoded.extend(symbols)
        return encoded

    def encode(self, text):
        return [self.vocab.get(symbol, self.unk_id) for symbol in self._encode_symbols(text)]

    def decode(self, token_ids):
        return "".join(self._id_to_token.get(token_id, "<UNK>") for token_id in token_ids)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "custom_unicode_bpe_v1",
            "vocab_size": len(self.vocab),
            "merges": [[left, right] for (left, right), _ in self.merges],
            "vocab": self.vocab,
            "unk_id": self.unk_id,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_vocabulary(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "custom_unicode_bpe_v1":
            raise ValueError(f"Unsupported tokenizer format: {data.get('format')!r}")
        self.vocab = data["vocab"]
        self.vocab_size = data["vocab_size"]
        self.unk_id = data["unk_id"]
        self.merges = [((left, right), left + right) for left, right in data["merges"]]
        self._refresh_indexes()


def iter_lines(path, max_lines=None):
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source):
            if max_lines is not None and line_number >= max_lines:
                break
            line = line.strip()
            if line:
                yield line


def evaluate(tokenizer, test_file):
    total_tokens = total_words = total_characters = unknown_tokens = 0
    token_frequencies = collections.Counter()
    examples = []
    for line in iter_lines(test_file):
        token_ids = tokenizer.encode(line)
        total_tokens += len(token_ids)
        total_words += len(line.split())
        total_characters += len(line)
        unknown_tokens += token_ids.count(tokenizer.unk_id)
        token_frequencies.update(token_ids)
        if len(examples) < 3:
            examples.append({
                "text": line,
                "token_ids": token_ids,
                "tokens": [tokenizer._id_to_token[token_id] for token_id in token_ids],
            })

    metrics = {
        "evaluation_split": str(test_file),
        "total_words": total_words,
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "vocab_size": len(tokenizer.vocab),
        "average_characters_per_token": total_characters / total_tokens if total_tokens else 0,
        "average_tokens_per_word": total_tokens / total_words if total_words else 0,
        "unknown_tokens": unknown_tokens,
        "unknown_token_rate": unknown_tokens / total_tokens if total_tokens else 0,
        "top_token_frequencies": [
            {"id": token_id, "token": tokenizer._id_to_token[token_id], "count": count}
            for token_id, count in token_frequencies.most_common(50)
        ],
        "examples": examples,
    }
    print("\n--- Tokenizer evaluation ---")
    print(f"Vocab size: {metrics['vocab_size']:,}")
    print(f"Average characters per token: {metrics['average_characters_per_token']:.2f}")
    print(f"Average tokens per word (fertility): {metrics['average_tokens_per_word']:.2f}")
    print(f"Unknown-token rate: {metrics['unknown_token_rate']:.4%}")
    return metrics


def main():
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=37000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-word-types", type=int, default=10000)
    parser.add_argument("--max-lines", type=int, default=0, help="Training-line cap; 0 uses the full corpus")
    parser.add_argument("--train-file", type=Path, default=base_dir / "datasets" / "train.txt")
    parser.add_argument("--test-file", type=Path, default=base_dir / "datasets" / "test.txt")
    parser.add_argument("--output", type=Path, default=base_dir / "models" / "nepali_tokenizer.json")
    parser.add_argument("--vocab-output", type=Path, default=base_dir / "models" / "nepali_vocabulary.json")
    parser.add_argument("--report-output", type=Path, default=base_dir / "reports" / "nepali_tokenizer_statistics.json")
    args = parser.parse_args()

    if args.vocab_size > MAX_VOCAB_SIZE:
        parser.error(f"--vocab-size cannot exceed {MAX_VOCAB_SIZE:,}")

    tokenizer = CustomBPETokenizer(args.vocab_size)
    started_at = time.time()
    max_lines = None if args.max_lines == 0 else args.max_lines
    tokenizer.train(iter_lines(args.train_file, max_lines), args.min_frequency, args.max_word_types)
    print(f"Training completed in {time.time() - started_at:.1f}s")
    tokenizer.save(args.output)
    tokenizer.save_vocabulary(args.vocab_output)
    print(f"Saved tokenizer to {args.output}")
    print(f"Saved vocabulary to {args.vocab_output}")
    metrics = evaluate(tokenizer, args.test_file)
    metrics["training"] = {
        "train_file": str(args.train_file),
        "max_lines": max_lines,
        "max_word_types": args.max_word_types,
        "min_frequency": args.min_frequency,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved tokenizer report to {args.report_output}")


if __name__ == "__main__":
    main()
