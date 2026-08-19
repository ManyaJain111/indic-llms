# Language Models and Agents — Individual Project
## Phase 1: Data Collection and Tokenizer Construction

**Student:** Manya Jain

---

## Languages

| Model | Language | Tier |
|---|---|---|
| Model H | **Hindi** | Higher-resource (non-English Indian) |
| Model L | **Nepali** | Lower-resource (permitted list) |

Both models are fully independent: separate datasets, tokenizers, vocabularies, and no shared data.

---

## Repository Layout

```
/
├── README.md
├── hindi/
│   ├── collection/          # Data collection scripts (scraper, wikipedia, PIB, subtitles)
│   ├── preprocessing/       # clean_corpus.py, prepare_dataset.py, clean_wiki.py
│   ├── training/            # train_tokenizer.py (byte-level BPE)
│   ├── datasets/            # source_manifest.json, stats.json, train/val/test splits
│   ├── models/
│   │   ├── vocab_10000/     # tokenizer JSON + PKL + vocabulary
│   │   ├── vocab_16000/     # tokenizer JSON + PKL + vocabulary
│   │   └── vocab_32000/     # tokenizer JSON + vocabulary (primary submission)
│   ├── reports/
│   │   ├── vocab_10000/     # hindi_tokenizer_statistics_10000.json
│   │   ├── vocab_16000/     # hindi_tokenizer_statistics_16000.json
│   │   └── vocab_32000/     # hindi_tokenizer_statistics_32000.json
│   └── requirements.txt
├── nepali/
│   ├── collection/          # nepali_news_scraper.py, wikipedia_scraper.py, hf_dataset_scraper.py
│   ├── preprocessing/       # clean_corpus.py, prepare_dataset.py, check_words.py
│   ├── training/            # train_tokenizer.py (unicode character-level BPE)
│   ├── datasets/            # source_manifest.json, train/val/test splits
│   ├── models/
│   │   ├── vocab_10000/     # tokenizer JSON + PKL + vocabulary
│   │   ├── vocab_16000/     # tokenizer JSON + PKL + vocabulary
│   │   └── vocab_37000/     # tokenizer JSON + PKL + vocabulary (primary submission)
│   └── reports/
│       ├── vocab_10000/     # nepali_tokenizer_statistics_10000.json
│       ├── vocab_16000/     # nepali_tokenizer_statistics_16000.json
│       ├── vocab_37000/     # nepali_tokenizer_statistics_37000.json
│       └── nepali_dataset_statistics.md
└── report/
    ├── phase_1_report.tex   # Full Phase 1 report source
    └── phase1_report.pdf    # Compiled Phase 1 report (tables, statistics, analysis)
```

---

## Dataset Links

Raw corpora are too large for git (~14GB total). Cleaned train/val/test splits are hosted on HuggingFace:

| Language | Link | Split |
|---|---|---|
| Hindi (Model H) | [manyajain111/hindi-lma-corpus](https://huggingface.co/datasets/manyajain111/hindi-lma-corpus) | 90/5/5 |
| Nepali (Model L) | [manyajain111/nepali-lma-corpus](https://huggingface.co/datasets/manyajain111/nepali-lma-corpus) | 90/5/5 |

> **Access:** Datasets are private. Please contact the repository owner to request access, or use the scripts below to reproduce from scratch.

---

## Corpus Statistics Summary

| | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| Post-cleaning words | 455,866,707 | 513,019,586 |
| Training tokens (32k/37k tokenizer) | ~549.4M | ~618.7M |
| Manual collection share | ~19.7% | ~22.3% |
| Train split | 410,362,761 words | 461,505,584 words |
| Val split | 22,804,967 words | 25,802,084 words |
| Test split | 22,698,979 words | 25,711,918 words |

---

## Tokenizer Summary

Three vocabulary sizes were trained per language to allow flexibility in Phase 2 model sizing (embedding tables consume 18–49% of a 25M-param model at d=384 depending on vocab size).

| Language | Vocab | Fertility | Chars/Token | UNK rate |
|---|---|---|---|---|
| Hindi | 10k | 1.3523 | 3.77 | 0.00% |
| Hindi | 16k | 1.2777 | 3.99 | 0.00% |
| Hindi | 32k | 1.2052 | 4.23 | 0.00% |
| Nepali | 10k | 1.4339 | 4.53 | ~0.0001% |
| Nepali | 16k | 1.3301 | 4.88 | ~0.0001% |
| Nepali | 37k | 1.2059 | 5.38 | ~0.0001% |

Hindi uses **byte-level BPE** (GPT-2 style, 0% UNK guaranteed).
Nepali uses **Unicode character-level BPE** (human-readable vocabulary).

---

## Reproduction Steps

### 1. Install dependencies
```bash
pip install -r hindi/requirements.txt
```

### 2. Reproduce dataset splits
Download cleaned data from HuggingFace (above), or run from scratch:
```bash
python hindi/preprocessing/prepare_dataset.py
python nepali/preprocessing/prepare_dataset.py
```

### 3. Train tokenizers
```bash
# Hindi (byte-level BPE) — choose vocab size with --vocab-size
python hindi/training/train_tokenizer.py --vocab-size 32000

# Nepali (unicode character-level BPE)
python nepali/training/train_tokenizer.py --vocab-size 37000
```

### 4. Evaluate tokenizer statistics
```bash
python hindi/reports/dataset_statistics.py
python nepali/reports/dataset_statistics.py
```

Pre-trained tokenizer files and vocabulary files for all three sizes are already committed under `hindi/models/` and `nepali/models/` respectively — no retraining needed to use them.
