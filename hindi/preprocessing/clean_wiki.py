import os
import json
import re
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.resolve()
EXTRACTED_DIR = BASE_DIR / "hindi-wikipedia" / "extracted"
OUTPUT_FILE = BASE_DIR / "hindi-wikipedia" / "cleaned_wiki.txt"

def clean_text(text: str) -> str:
    """Removes English letters and cleans up extra whitespaces."""
    if not text:
        return ""
    
    # Remove all English alphabets (uppercase and lowercase)
    cleaned = re.sub(r'[a-zA-Z]+', '', text)
    
    # Remove multiple spaces that might have been left behind
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    # Remove multiple blank lines
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    
    return cleaned.strip()

def main():
    if not EXTRACTED_DIR.exists():
        print(f"Error: Directory {EXTRACTED_DIR} does not exist.")
        return

    print(f"Starting to process Wikipedia files from: {EXTRACTED_DIR}")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    
    total_files = 0
    total_articles = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        # Walk through all directories like AA, AB, etc.
        for root, _, files in os.walk(EXTRACTED_DIR):
            # Sort files to process them in order (e.g. wiki_00, wiki_01)
            for file_name in sorted(files):
                if not file_name.startswith("wiki_"):
                    continue
                    
                file_path = Path(root) / file_name
                total_files += 1
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            article_data = json.loads(line)
                            raw_text = article_data.get('text', '')
                            
                            # Skip if empty
                            if not raw_text:
                                continue
                                
                            # Clean the text
                            cleaned = clean_text(raw_text)
                            
                            if cleaned:
                                # Write the article title as a header (optional, but good for separation)
                                title = clean_text(article_data.get('title', ''))
                                if title:
                                    out_f.write(f"{title}\n")
                                out_f.write(f"{cleaned}\n\n")
                                total_articles += 1
                                
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse JSON line in {file_path}")
                            continue

                if total_files % 10 == 0:
                    print(f"Processed {total_files} files, {total_articles} articles so far...")

    print("=" * 50)
    print("✅ Done!")
    print(f"Total files processed: {total_files}")
    print(f"Total articles cleaned and saved: {total_articles}")
    print(f"Cleaned corpus is available at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
