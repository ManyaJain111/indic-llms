"""
Fast pre-tokenizer using pool initializer.
Each worker loads the tokenizer ONCE and reuses it (with word cache) across all chunks.
"""
import os, sys, json, re, string, tarfile, glob, numpy as np
from multiprocessing import Pool, cpu_count

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(SCRIPT_DIR, "phase2", "pretokenized")
EXTRACT_BASE = os.path.join(SCRIPT_DIR, "phase2", "_extracted")
TOK_HINDI    = os.path.join(SCRIPT_DIR, "phase2", "tokenizers", "hindi_tokenizer_16000.json")
TOK_NEPALI   = os.path.join(SCRIPT_DIR, "phase2", "tokenizers", "nepali_tokenizer_16000.json")
MAX_TRAIN    = 520_000_000
CHUNK_LINES  = 5000     # larger chunks = fewer imap calls = less overhead

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EXTRACT_BASE, exist_ok=True)

PUNCT    = re.escape(string.punctuation + "।॥''\"\"")
WORD_PAT = re.compile(rf" ?[^{PUNCT}\s]+| ?[{PUNCT}]+", re.UNICODE)

# ------------------------------------------------------------------
# Per-worker globals (loaded ONCE per process, shared across chunks)
# ------------------------------------------------------------------
_vocab       = None
_merge_ranks = None
_b2c         = None
_word_cache  = None   # word -> token_ids  (persists across ALL chunks in this worker)

def _init_worker(tok_path):
    """Called ONCE when each worker process starts — loads tokenizer into globals."""
    global _vocab, _merge_ranks, _b2c, _word_cache
    with open(tok_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _vocab       = data["vocab"]
    merges       = [((a, b), a + b) for a, b in data["merges"]]
    _merge_ranks = {pair: i for i, (pair, _) in enumerate(merges)}
    _b2c         = {i: chr(i) for i in range(256)}
    _word_cache  = {}

def _encode_line(text):
    ids = []
    for w in WORD_PAT.findall(text):
        if w not in _word_cache:
            symbols = [_b2c[b] for b in w.encode("utf-8")]
            while len(symbols) > 1:
                pairs  = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
                ranked = [(_merge_ranks[p], i) for i, p in enumerate(pairs) if p in _merge_ranks]
                if not ranked:
                    break
                _, i = min(ranked)
                symbols = symbols[:i] + [symbols[i]+symbols[i+1]] + symbols[i+2:]
            _word_cache[w] = [_vocab.get(s, 0) for s in symbols]
        ids.extend(_word_cache[w])
    return ids

def _worker_chunk(lines):
    """Tokenize a list of lines using the persistent per-worker tokenizer + cache."""
    out = []
    for line in lines:
        line = line.strip()
        if line:
            out.extend(_encode_line(line))
    return out


# ------------------------------------------------------------------
def download_tar(repo_id, filename):
    from huggingface_hub import hf_hub_download
    print(f"  Downloading {filename} from {repo_id} ...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    print(f"  Cached: {path}")
    return path

def extract_tar(tar_path, extract_dir):
    done = os.path.join(extract_dir, ".done")
    if os.path.exists(done):
        print(f"  Already extracted: {extract_dir}")
        return
    print(f"  Extracting to {extract_dir} ...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    open(done, "w").close()
    print("  Extraction done.")

def tokenize_file(fpath, tok_path, max_tokens=None):
    import time
    print(f"  Reading {os.path.basename(fpath)} ...", flush=True)
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print(f"  {len(lines):,} lines | CHUNK={CHUNK_LINES} | CPUs={cpu_count()-1}", flush=True)

    chunks  = [lines[i:i+CHUNK_LINES] for i in range(0, len(lines), CHUNK_LINES)]
    ncpus   = max(1, cpu_count() - 1)
    tokens  = []
    t0      = time.time()

    # Pool with initializer: each worker loads tokenizer ONCE and keeps cache alive
    with Pool(ncpus, initializer=_init_worker, initargs=(tok_path,)) as pool:
        for i, result in enumerate(pool.imap(_worker_chunk, chunks)):
            tokens.extend(result)
            if i % 10 == 0:
                elapsed = time.time() - t0
                tps     = len(tokens) / max(elapsed, 1)
                print(f"  {i+1}/{len(chunks)} chunks | {len(tokens):,} tokens | {tps:,.0f} tok/s", flush=True)
            if max_tokens and len(tokens) >= max_tokens:
                print(f"  Reached {max_tokens:,} token cap.", flush=True)
                break

    elapsed = time.time() - t0
    print(f"  Done: {len(tokens):,} tokens in {elapsed/60:.1f} min ({len(tokens)/elapsed:,.0f} tok/s)", flush=True)
    return tokens


DATASETS = [
    {
        "lang":    "hindi",
        "repo":    "manyajain111/hindi-lma-corpus",
        "archive": "hindi_submission_phase1.tar.gz",
        "tok":     TOK_HINDI,
        "splits":  {"train": ["train.txt", "hindi_news_data.txt"], "val": ["val.txt"], "test": ["test.txt"]},
    },
    {
        "lang":    "nepali",
        "repo":    "manyajain111/nepali-lma-corpus",
        "archive": "nepali_submission_phase1.tar.gz",
        "tok":     TOK_NEPALI,
        "splits":  {"train": ["train.txt"], "val": ["val.txt"], "test": ["test.txt"]},
    },
]

if __name__ == "__main__":
    for ds in DATASETS:
        lang = ds["lang"]
        tok  = ds["tok"]
        print(f"\n{'='*60}\nLanguage: {lang.upper()}\n{'='*60}")

        tar_path    = download_tar(ds["repo"], ds["archive"])
        extract_dir = os.path.join(EXTRACT_BASE, lang)
        extract_tar(tar_path, extract_dir)

        for split, fnames in ds["splits"].items():
            out_path = os.path.join(OUT_DIR, f"{lang}_{split}.npy")
            if os.path.exists(out_path):
                arr = np.load(out_path)
                print(f"  [{split}] already exists ({len(arr):,} tokens) — skipping.")
                continue

            all_tokens = []
            for fname in fnames:
                matches = glob.glob(os.path.join(extract_dir, "**", fname), recursive=True)
                if not matches:
                    print(f"  [{split}] {fname} not found.")
                    continue
                limit = (MAX_TRAIN - len(all_tokens)) if split == "train" else None
                all_tokens.extend(tokenize_file(matches[0], tok, max_tokens=limit))
                if split == "train" and len(all_tokens) >= MAX_TRAIN:
                    break

            arr = np.array(all_tokens, dtype=np.uint16)
            np.save(out_path, arr)
            print(f"  ✅ [{split}] {len(arr):,} tokens → {out_path}")

    print("\n\n✅ All done! Upload phase2/pretokenized/ as a Kaggle Dataset.")
