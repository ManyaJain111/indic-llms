import json
import random
import unicodedata
from pathlib import Path
from tqdm import tqdm

def main():
    base_dir = Path(__file__).parent.parent / "datasets" / "raw"
    out_dir = base_dir.parent
    out_dir.mkdir(exist_ok=True)
    
    train_file = out_dir / "train.txt"
    test_file = out_dir / "test.txt"
    val_file = out_dir / "val.txt"
    
    datasets = [
        ("iriis_nepali", base_dir / "iriis_nepali_data.txt", "downloaded"),
        ("nepali_wikipedia", base_dir / "wikipedia_nepalese_data.txt", "downloaded"),
        ("nepali_news_scrape", base_dir / "scraper_output" / "nepali_news_data.txt", "manual"),
    ]
    rng = random.Random(42)
    source_stats = {}
    
    train_lines = 0
    test_lines = 0
    val_lines = 0
    
    print("Preparing aggregated Nepali dataset...")
    print(f"Outputting to {out_dir}")
    
    with open(train_file, "w", encoding="utf-8") as f_train, \
         open(test_file, "w", encoding="utf-8") as f_test, \
         open(val_file, "w", encoding="utf-8") as f_val:
        
        for source_id, ds, collection_type in datasets:
            if not ds.exists():
                print(f"Warning: Dataset not found {ds}")
                continue
            source_stats[source_id] = {
                "path": str(ds), "collection_type": collection_type,
                "lines": 0, "whitespace_tokens": 0,
                "splits": {"train": 0, "val": 0, "test": 0},
            }
            print(f"\nProcessing {ds.name}...")
            
            with open(ds, "r", encoding="utf-8") as f_in:
                for line in tqdm(f_in, desc=ds.name):
                    line = " ".join(unicodedata.normalize("NFC", line).split())
                    if not line:
                        continue
                    source_stats[source_id]["lines"] += 1
                    source_stats[source_id]["whitespace_tokens"] += len(line.split())
                    r = rng.random()
                    if r < 0.05:
                        f_test.write(line + "\n")
                        test_lines += 1
                        source_stats[source_id]["splits"]["test"] += 1
                    elif r < 0.10:
                        f_val.write(line + "\n")
                        val_lines += 1
                        source_stats[source_id]["splits"]["val"] += 1
                    else:
                        f_train.write(line + "\n")
                        train_lines += 1
                        source_stats[source_id]["splits"]["train"] += 1
                        
    print(f"\nDone! Dataset split complete.")
    print(f"Train lines: {train_lines:,}")
    print(f"Test lines: {test_lines:,}")
    print(f"Val lines: {val_lines:,}")
    (out_dir / "source_manifest.json").write_text(
        json.dumps({"split_seed": 42, "sources": source_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
