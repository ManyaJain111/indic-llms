#!/usr/bin/env python3
"""
Corpus collection: streams monolingual text from a large hosted corpus
until a target word count is hit, writing raw text to disk (one
document/line per line, untouched — cleaning is a separate deliverable).

Requires: pip install datasets

Usage:
  # Hindi from IndicCorpV2 (recommended default)
  python collect_corpus.py --source indiccorp --lang-split hin_Deva \
      --target-words 200000000 --out data/hindi/raw.txt

  # Fallback / supplement: Sangraha verified subset
  python collect_corpus.py --source sangraha --lang-code hin \
      --target-words 200000000 --out data/hindi/raw_sangraha.txt

  # Later, for Model L (swap --lang-split / --lang-code to the target language)
  python collect_corpus.py --source indiccorp --lang-split <lang>_<Script> \
      --target-words 200000000 --out data/model_l/raw.txt
"""
import argparse
import os
import sys

from datasets import load_dataset


def _text_field(example):
    """Most of these corpora use a 'text' field; fall back to the first str field."""
    if "text" in example:
        return example["text"]
    for k, v in example.items():
        if isinstance(v, str):
            return v
    return None


def stream_indiccorp(lang_split):
    # configs map split name -> underlying file glob, e.g. hin_Deva -> data/hi-*.txt
    return load_dataset(
        "ai4bharat/IndicCorpV2", "indiccorp_v2",
        split=lang_split, streaming=True,
    )


def stream_sangraha(lang_code, subset="verified"):
    # data_dir is "<subset>/<3-letter lang code>", e.g. "verified/hin"
    return load_dataset(
        "ai4bharat/sangraha", data_dir=f"{subset}/{lang_code}",
        split="train", streaming=True,
    )


def stream_cc100(lang_code):
    return load_dataset("statmt/cc100", lang=lang_code, streaming=True, split="train")


def collect(stream, out_path, target_words, flush_every=10_000, skip_docs=0):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_words = 0
    n_docs = 0
    mode = "a" if skip_docs > 0 else "w"
    
    if skip_docs > 0:
        print(f"Skipping first {skip_docs:,} documents from the stream...")
        stream = stream.skip(skip_docs)
        
    with open(out_path, mode, encoding="utf-8") as f:
        for example in stream:
            text = _text_field(example)
            if not text:
                continue
            text = text.strip()
            if not text:
                continue
                
            words_in_text = len(text.split())
            f.write(text.replace("\n", " ") + "\n")
            n_words += words_in_text
            n_docs += 1
            if n_docs % flush_every == 0:
                print(f"\rCollected {n_words:,} / {target_words:,} target "
                      f"({n_docs:,} docs)", end="", flush=True)
                f.flush()
            if n_words >= target_words:
                break
    print(f"\nDone: {n_words:,} words, {n_docs:,} docs -> {out_path}")
    return n_words, n_docs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["indiccorp", "sangraha", "cc100"], required=True)
    ap.add_argument("--lang-split", help="IndicCorpV2 split name, e.g. hin_Deva")
    ap.add_argument("--lang-code", help="Sangraha (3-letter, e.g. hin) or CC-100 (2-letter, e.g. hi) code")
    ap.add_argument("--sangraha-subset", default="verified", choices=["verified", "unverified"])
    ap.add_argument("--skip-docs", type=int, default=0, help="Skip this many documents before collecting")
    ap.add_argument("--target-words", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.source == "indiccorp":
        if not args.lang_split:
            sys.exit("--lang-split required for --source indiccorp (e.g. hin_Deva)")
        stream = stream_indiccorp(args.lang_split)
    elif args.source == "sangraha":
        if not args.lang_code:
            sys.exit("--lang-code required for --source sangraha (e.g. hin)")
        stream = stream_sangraha(args.lang_code, args.sangraha_subset)
    else:
        if not args.lang_code:
            sys.exit("--lang-code required for --source cc100 (e.g. hi)")
        stream = stream_cc100(args.lang_code)

    collect(stream, args.out, args.target_words, skip_docs=args.skip_docs)


if __name__ == "__main__":
    main()