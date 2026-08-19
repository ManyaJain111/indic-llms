# Hindi Dataset Statistics — Model H

## Corpus Overview

| Metric | Value |
|---|---|
| **Language** | Hindi (Devanagari script) |
| **Total raw words** | 461,432,077 |
| **Post-cleaning words** | 455,866,707 |
| **Words removed (dedup + cleaning)** | ~5,565,370 (~1.2%) |
| **Estimated training tokens** (32k tokenizer, fertility 1.2052) | ~549,400,000 |

## Sources

| Category | Source | Word Count |
|---|---|---|
| Downloaded | Sangraha corpus | 194,198,603 |
| Downloaded | Indic corpus | 176,448,929 |
| Manual | Hindi Wikipedia | 42,801,209 |
| Manual | News scrape (new) | 42,383,013 |
| Manual | Subtitles | 5,234,233 |
| Manual | News scrape | 360,205 |
| Manual | PIB releases | 5,885 |
| **Total** | | **461,432,077** |

**Manual collection total:** 90,784,545 words ≈ **19.7% of total** *(marginally below 20% threshold; additional manual collection needed)*

## Cleaning Steps

1. Unicode NFC normalization
2. Exact-content deduplication (cross-split)
3. Devanagari purity filter — removal of lines containing non-Devanagari / English characters
4. Whitespace normalization

## Train / Validation / Test Split

Split ratio: **90 / 5 / 5**, seed 42

| Split | Lines | Words |
|---|---|---|
| Train | 6,205,144 | 410,362,761 |
| Validation | 344,730 | 22,804,967 |
| Test | 344,731 | 22,698,979 |
| **Total** | **6,894,605** | **455,866,707** |

## Tokenizer Statistics

Evaluated on 10,000 held-out lines from `test.txt` (656,900 words, 3,347,181 characters).

| Vocab Size | Fertility (tokens/word) | Avg Chars/Token | Total Tokens | UNK Rate |
|---|---|---|---|---|
| 10,000 | 1.3523 | 3.77 | 888,355 | 0.00% |
| 16,000 | 1.2777 | 3.99 | 839,350 | 0.00% |
| 32,000 | 1.2052 | 4.23 | 791,726 | 0.00% |

Tokenizer type: **Byte-level BPE** (GPT-2 style, 256-byte base vocabulary).
UNK rate is guaranteed 0% at all sizes due to byte-level encoding.

## Top 5 Tokens (32k tokenizer)

| Rank | Token | Meaning | Frequency |
|---|---|---|---|
| 1 | । | Danda (full stop) | 26,578 |
| 2 | के | ke (of/for) | 26,357 |
| 3 | में | mein (in) | 20,000 |
| 4 | है | hai (is) | 18,975 |
| 5 | , | comma | 18,000 |
