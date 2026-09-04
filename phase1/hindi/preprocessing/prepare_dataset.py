#!/usr/bin/env python3
import json
import os
import random
import hashlib
import unicodedata
from pathlib import Path
from tqdm import tqdm

def load_plain_text(path):
    lines = []
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return lines
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f):
            text = line.strip()
            if text:
                lines.append(text)
    return lines

def load_jsonl(path):
    lines = []
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return lines
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            headline = obj.get("headline", "").strip()
            body = obj.get("body", "").strip()
            if headline: lines.append(headline)
            if body: lines.append(body)
    return lines

def load_json(path):
    lines = []
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return lines
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
        for obj in tqdm(data):
            title = obj.get("title", "").strip()
            body = obj.get("body", "").strip()
            if title: lines.append(title)
            if body: lines.append(body)
    return lines

def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())

def main():
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "datasets" / "raw"
    
    sources = [
        (dataset_dir / "hindi-wikipedia" / "cleaned_wiki.txt", "text", "hindi_wikipedia", "downloaded"),
        (dataset_dir / "clean_hindi_subtitles.txt", "text", "hindi_subtitles", "downloaded"),
        (dataset_dir / "data" / "hindi" / "raw.txt", "text", "indic_corpus", "downloaded"),
        (dataset_dir / "data" / "hindi" / "raw_test.txt", "text", "indic_corpus_test", "downloaded"),
        (dataset_dir / "hindi_scraper_output" / "articles.jsonl", "jsonl", "hindi_news_scrape", "manual"),
        (dataset_dir / "pib_hindi_releases.json", "json", "pib_releases", "downloaded"),
        (base_dir / "collection" / "hindi_scraper_data" / "hindi_news_data.txt", "text", "new_hindi_news_scrape", "manual"),
        (dataset_dir / "data" / "hindi" / "raw_sangraha.txt", "text", "sangraha_corpus", "downloaded")
    ]
    
    all_lines = []
    seen_hashes = set()
    source_stats = {}
    
    for path, ftype, source_id, collection_type in sources:
        if ftype == "text":
            lines = load_plain_text(path)
        elif ftype == "jsonl":
            lines = load_jsonl(path)
        elif ftype == "json":
            lines = load_json(path)
        source_stats[source_id] = {"path": str(path), "collection_type": collection_type, "lines": 0, "whitespace_tokens": 0}
        for line in lines:
            line = normalize_text(line)
            if not line:
                continue
            fingerprint = hashlib.sha256(line.encode("utf-8")).digest()
            if fingerprint in seen_hashes:
                continue
            seen_hashes.add(fingerprint)
            all_lines.append((line, source_id))
            source_stats[source_id]["lines"] += 1
            source_stats[source_id]["whitespace_tokens"] += len(line.split())
            
    print(f"\nTotal collected lines/documents: {len(all_lines):,}")
    
    print("Shuffling data...")
    random.seed(42)
    random.shuffle(all_lines)
    
    # 90% train, 5% val, 5% test
    total = len(all_lines)
    train_end = int(0.9 * total)
    val_end = train_end + int(0.05 * total)
    
    train_lines = all_lines[:train_end]
    val_lines = all_lines[train_end:val_end]
    test_lines = all_lines[val_end:]
    
    print(f"Train split: {len(train_lines):,} lines")
    print(f"Val split: {len(val_lines):,} lines")
    print(f"Test split: {len(test_lines):,} lines")
    
    out_dir = base_dir / "datasets"
    out_dir.mkdir(exist_ok=True)
    
    for split_name, data in [("train", train_lines), ("val", val_lines), ("test", test_lines)]:
        out_path = out_dir / f"{split_name}.txt"
        print(f"Writing {out_path}...")
        with open(out_path, "w", encoding="utf-8") as f:
            for line in tqdm(data):
                f.write(line[0] + "\n")
                
    # Calculate word tokens roughly (whitespace separated)
    print("Calculating statistics...")
    total_words = 0
    for line, _ in tqdm(all_lines):
        total_words += len(line.split())
        
    stats = {
        "total_lines": total,
        "total_words": total_words,
        "train_lines": len(train_lines),
        "val_lines": len(val_lines),
        "test_lines": len(test_lines)
    }
    
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    with open(out_dir / "source_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"split_seed": 42, "sources": source_stats}, f, ensure_ascii=False, indent=2)
        
    print(f"Done! Final corpus prepared at {out_dir}")
    print(f"Total roughly {total_words:,} words (whitespace tokens).")

if __name__ == "__main__":
    main()
