import time
import sys
import subprocess
from pathlib import Path
import multiprocessing

from nepali_news_scraper import clean_nepali_purity, word_count

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)

TARGET_WORDS = 400_000_000
OUT_FILE = Path("iriis_nepali_data.txt")

def process_article(raw_text):
    if not raw_text or not isinstance(raw_text, str):
        return "", 0
    clean_text = clean_nepali_purity(raw_text)
    words = word_count(clean_text)
    return clean_text, words

def text_generator(dataset):
    for row in dataset:
        yield row.get("Article", row.get("text", row.get("content", "")))

def main():
    print(f"Loading IRIIS-RESEARCH/Nepali-Text-Corpus... (Downloading 8GB cache first)")
    
    # Load dataset without streaming (streaming=True hangs on this specific HF dataset)
    dataset = load_dataset("IRIIS-RESEARCH/Nepali-Text-Corpus", split="train")
    
    total_words = 0
    
    # If the file exists, we should figure out how many words we already have
    initial_words = 0
    if OUT_FILE.exists():
        print("Calculating existing words in file... please wait.")
        try:
            # wc -w is orders of magnitude faster than iterating lines in python
            out = subprocess.check_output(['wc', '-w', str(OUT_FILE)])
            initial_words = int(out.split()[0])
        except Exception as e:
            print(f"wc failed: {e}. Falling back to slow loop.")
            with open(OUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    initial_words += len(line.split())
        print(f"Resuming with {initial_words:,} words already saved.")
    
    total_words = initial_words
    start_time = time.time()
    last_print_time = start_time
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting extraction to {OUT_FILE} using {num_cores} cores...")
    
    with open(OUT_FILE, "a", encoding="utf-8") as out_f:
        try:
            with multiprocessing.Pool(processes=num_cores) as pool:
                for clean_text, words in pool.imap_unordered(process_article, text_generator(dataset), chunksize=1000):
                    if words > 0:
                        out_f.write(clean_text + "\n")
                        total_words += words
                        
                    if total_words >= TARGET_WORDS:
                        print(f"\nGoal Reached! Collected {total_words:,} words.")
                        break
                        
                    current_time = time.time()
                    if current_time - last_print_time >= 5.0:
                        words_this_session = total_words - initial_words
                        speed = words_this_session / max((current_time - start_time), 1) * 60
                        print(f"Progress: {total_words:,} / {TARGET_WORDS:,} words | Speed: {speed:,.0f} words/min")
                        last_print_time = current_time
                        out_f.flush()
        
        except KeyboardInterrupt:
            print(f"\nInterrupted! Saved {total_words:,} words to {OUT_FILE}")

if __name__ == "__main__":
    main()
