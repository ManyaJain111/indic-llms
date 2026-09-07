#!/usr/bin/env python3
from pathlib import Path
from tqdm import tqdm

def get_stats(file_path: Path):
    lines = 0
    words = 0
    chars = 0
    if not file_path.exists():
        return lines, words, chars
    
    print(f"Calculating stats for {file_path.name}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            lines += 1
            words += len(line.split())
            chars += len(line)
    return lines, words, chars

def main():
    base_dir = Path(__file__).parent.parent / "datasets"
    splits = ["train.txt", "test.txt", "val.txt"]
    
    report = "# Nepali Dataset Statistics Report\n\n"
    report += "Generated from the current Nepali split files. Word counts are whitespace-delimited and are not tokenizer-token counts.\n\n"
    report += "| Split | Lines | Words | Characters |\n"
    report += "|-------|-------|-------|------------|\n"
    
    total_lines, total_words, total_chars = 0, 0, 0
    
    for split in splits:
        file_path = base_dir / split
        l, w, c = get_stats(file_path)
        report += f"| {split} | {l:,} | {w:,} | {c:,} |\n"
        total_lines += l
        total_words += w
        total_chars += c
        
    report += f"| **Total** | **{total_lines:,}** | **{total_words:,}** | **{total_chars:,}** |\n\n"
    
    report_file = Path(__file__).parent / "nepali_dataset_statistics.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\nGenerated statistics report: {report_file.name}")
    print(report)

if __name__ == "__main__":
    main()
