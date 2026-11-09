#!/usr/bin/env python3
import re
import os
import hashlib
from pathlib import Path
from tqdm import tqdm

LATIN_LETTER = re.compile(r"[A-Za-z]")
ALLOWED_CHARS = re.compile(r"[^\u0900-\u097F0-9।॥,.\-!?\s]")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

def is_clean_nepali_word(word: str) -> bool:
    return not LATIN_LETTER.search(word)

def process_file(in_path: Path, seen_hashes: set):
    if not in_path.exists():
        return
        
    out_path = in_path.with_name(in_path.name + ".tmp")
    lines_dropped = 0
    lines_kept = 0
    english_words_removed = 0
    
    print(f"\nProcessing {in_path.name}...")
    
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in):
            line_str = line.strip()
            if not line_str:
                continue
                
            original_words = line_str.split()
            words = [w for w in original_words if not LATIN_LETTER.search(w)]
            english_words_removed += len(original_words) - len(words)
            
            line_str = " ".join(words)
            line_str = ALLOWED_CHARS.sub("", line_str)
            line_str = RE_MULTI_SPACE.sub(" ", line_str).strip()
            
            if not line_str:
                lines_dropped += 1
                continue
                
            f_out.write(line_str + "\n")
            lines_kept += 1
            
    print(f"Replacing original file {in_path.name} with cleaned version...")
    os.replace(out_path, in_path)
    
    print(f"Results for {in_path.name}:")
    print(f"  Lines kept: {lines_kept:,}")
    print(f"  Lines dropped (duplicates/empty): {lines_dropped:,}")
    print(f"  English words removed: {english_words_removed:,}")
    
def main():
    base_dir = Path(__file__).parent.parent / "datasets"
    train_file = base_dir / "train.txt"
    test_file = base_dir / "test.txt"
    val_file = base_dir / "val.txt"
    
    if not train_file.exists():
        print(f"Error: {train_file} does not exist.")
        return
        
    seen_hashes = set()
    process_file(train_file, seen_hashes)
    process_file(test_file, seen_hashes)
    process_file(val_file, seen_hashes)
    
    print("\nCorpus cleaning completely finished!")

if __name__ == "__main__":
    main()
