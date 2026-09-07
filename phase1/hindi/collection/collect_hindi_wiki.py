"""
Hindi Wikipedia corpus collection pipeline for Model H (LMA Phase 1).
Downloads latest Hindi Wikipedia dump, extracts clean text, normalizes,
dedupes, and produces train/val/test splits with corpus statistics.

Usage:
    pip install wikiextractor requests --break-system-packages
    python collect_hindi_wiki.py --outdir ./data/hindi/wikipedia
"""

import argparse
import bz2
import json
import os
import re
import subprocess
import unicodedata
import hashlib
import random
from pathlib import Path

import requests

WIKI_DUMP_URL = "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2"

REQUEST_HEADERS = {
    "User-Agent": "LMA-Project-Hindi-Corpus/1.0 (IIIT-H course project; contact: your-email@students.iiit.ac.in)"
}


def download_dump(outdir: Path, max_retries: int = 10) -> Path:
    """Download the Hindi Wikipedia dump with resume support and retries."""
    mirrors = [
        "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2",
        "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2",
        "https://wikipedia.c3sl.ufpr.br/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2",
    ]
    dump_path = outdir / "hiwiki-latest-pages-articles.xml.bz2"
    outdir.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        url = mirrors[attempt % len(mirrors)]
        headers = dict(REQUEST_HEADERS)
        mode = "wb"
        if dump_path.exists():
            existing_size = dump_path.stat().st_size
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
        else:
            existing_size = 0

        try:
            with requests.get(url, headers=headers, stream=True, timeout=(10, 60)) as r:
                if r.status_code == 416:
                    print("Already fully downloaded.")
                    return dump_path
                if r.status_code == 403 and "Range" in headers:
                    # Some mirrors reject Range requests, restart fresh from this mirror
                    headers.pop("Range")
                    mode = "wb"
                    existing_size = 0
                    r.close()
                    continue
                r.raise_for_status()

                total = int(r.headers.get("content-length", 0)) + existing_size
                downloaded = existing_size
                with open(dump_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\rDownloading [{url.split('/')[2]}]: {pct:.1f}%", end="", flush=True)
            print(f"\nSaved dump to {dump_path}")
            return dump_path

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"\nAttempt {attempt+1} failed ({e}). Retrying with resume...")
            continue

    raise RuntimeError("Failed to download dump after max retries.")


def run_wikiextractor(dump_path: Path, extract_dir: Path):
    """Run WikiExtractor to pull clean article text (JSON lines) from the dump."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "-m", "wikiextractor.WikiExtractor",
        str(dump_path),
        "--json",
        "-o", str(extract_dir),
        "--no-templates",
        "--processes", "4",
    ]
    print("Running wikiextractor:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def normalize_text(text: str) -> str:
    """Unicode normalization (NFC) and whitespace cleanup."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_hindi_line(line: str, min_devanagari_ratio: float = 0.4) -> bool:
    """Keep lines that are meaningfully Hindi (Devanagari script), drop junk."""
    if len(line) < 20:
        return False
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", line))
    if len(line) == 0:
        return False
    ratio = devanagari_chars / len(line)
    return ratio >= min_devanagari_ratio


def collect_articles(extract_dir: Path):
    """Walk wikiextractor's output dirs and yield cleaned article texts."""
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            fpath = Path(root) / fname
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        yield text


def dedupe_and_clean(articles):
    """Deduplicate articles by content hash, normalize, filter invalid lines."""
    seen_hashes = set()
    cleaned = []
    for article in articles:
        norm = normalize_text(article)
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        valid_lines = [
            ln for ln in norm.split(". ")
            if is_valid_hindi_line(ln)
        ]
        if not valid_lines:
            continue
        cleaned.append(". ".join(valid_lines))
    return cleaned


def split_and_save(articles, outdir: Path, seed: int = 42):
    """Create train/val/test splits (90/5/5) and save as plain text files."""
    random.seed(seed)
    random.shuffle(articles)

    n = len(articles)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)

    train = articles[:n_train]
    val = articles[n_train:n_train + n_val]
    test = articles[n_train + n_val:]

    splits = {"train": train, "val": val, "test": test}
    outdir.mkdir(parents=True, exist_ok=True)

    stats = {"source": "hi.wikipedia.org dump", "collection_method": "manual (self-scraped dump + extraction)"}
    for name, split in splits.items():
        fpath = outdir / f"{name}.txt"
        with open(fpath, "w", encoding="utf-8") as f:
            for article in split:
                f.write(article + "\n")
        token_count = sum(len(a.split()) for a in split)
        stats[name] = {"num_articles": len(split), "approx_word_tokens": token_count}
        print(f"{name}: {len(split)} articles, ~{token_count} whitespace tokens")

    with open(outdir / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./data/hindi/wikipedia")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    dump_path = outdir / "hiwiki-latest-pages-articles.xml.bz2"
    extract_dir = outdir / "extracted"
    splits_dir = outdir / "splits"

    if not args.skip_download:
        dump_path = download_dump(outdir)

    if not args.skip_extract:
        run_wikiextractor(dump_path, extract_dir)

    print("Collecting and cleaning articles...")
    articles = list(collect_articles(extract_dir))
    print(f"Raw articles: {len(articles)}")

    cleaned = dedupe_and_clean(articles)
    print(f"After dedupe/clean: {len(cleaned)}")

    split_and_save(cleaned, splits_dir)


if __name__ == "__main__":
    main()