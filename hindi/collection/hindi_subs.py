import os
import re
import requests
import zipfile
from tqdm import tqdm

def is_clean_hindi_line(line: str) -> bool:
    """Evaluates if a line is pristine, well-encoded Hindi."""
    line = line.strip()
    if not line:
        return False
        
    # 1. Reject if line contains ANY Latin characters (A-Z, a-z)
    if re.search(r'[a-zA-Z]', line):
        return False

    # 2. Extract Devanagari characters
    devanagari_chars = re.findall(r'[\u0900-\u097F]', line)
    if len(devanagari_chars) < 5:
        return False  # Skip tiny fragments or pure punctuation

    # 3. Count Devanagari Matras (Vowels) to detect legacy encoding corruption
    matras = re.findall(r'[\u093E-\u094C\u0902]', line)
    matra_ratio = len(matras) / len(devanagari_chars)
    
    # Clean Hindi text has a matra ratio of ~0.25 to 0.40. 
    # Corrupted text (missing vowels) drops below 0.15.
    if matra_ratio < 0.18:
        return False

    # 4. Check for Word Fusions (Missing spaces)
    words = line.split()
    avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
    
    # If the average word is over 10 letters, spaces were likely deleted
    if avg_word_length > 10:
        return False

    return True

def download_and_purify(output_filename="clean_hindi_subtitles.txt"):
    # OPUS v2024 Direct Object Storage URL
    opus_url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-hi.txt.zip"
    zip_path = "temp_en_hi.zip"
    
    print("[*] STEP 1: Downloading dataset from OPUS...")
    response = requests.get(opus_url, stream=True)
    if response.status_code != 200:
        print(f"[!] Download failed: Status {response.status_code}")
        return

    # Download with a progress bar
    total_size = int(response.headers.get('content-length', 0))
    with open(zip_path, "wb") as f, tqdm(
        desc="Downloading ZIP", total=total_size, unit="iB", unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
            
    print("\n[*] STEP 2: Extracting and Purifying text on the fly...")
    kept_lines = 0
    discarded_lines = 0
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the specific file ending in '.hi'
        hi_filename = next(name for name in z.namelist() if name.endswith('.hi'))
        
        with z.open(hi_filename) as extracted_file, open(output_filename, 'w', encoding='utf-8') as final_file:
            # We don't know the exact line count without reading it all, so we just show an active counter
            with tqdm(desc="Scanning Lines", unit=" lines") as pbar:
                for raw_line in extracted_file:
                    # Decode from bytes to string
                    line = raw_line.decode('utf-8')
                    
                    if is_clean_hindi_line(line):
                        final_file.write(line.strip() + "\n")
                        kept_lines += 1
                    else:
                        discarded_lines += 1
                        
                    pbar.update(1)
                
    # Clean up the 100MB+ zip file from your hard drive
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print("\n" + "="*40)
    print(" DATASET PURIFICATION RESULTS")
    print("="*40)
    print(f"Pristine Lines Saved: {kept_lines:,}")
    print(f"Corrupted Lines Cut:  {discarded_lines:,}")
    print(f"Saved to:             {output_filename}")
    print("="*40)
    print("[✔] You now have a flawless conversational dataset.")

if __name__ == "__main__":
    download_and_purify()
