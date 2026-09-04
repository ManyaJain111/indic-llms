# Nepali Dataset Card

## Language Choice

Nepali is an allowed lower-resource language for Model L. The corpus uses Nepali-only sources and is kept separate from the Hindi corpus.

## Current Corpus

The current cleaned splits contain 184,627 lines and 36,410,349 whitespace-delimited words. The tokenizer's held-out fertility is 1.51 tokens per word, which projects to roughly 55M tokenizer tokens. This is below the 500M target.

## Sources and Collection Method

The rebuild pipeline records three sources: IRIIS Nepali Text Corpus and Nepali Wikipedia as downloaded sources, plus the project Nepali news scraper as manual collection. The scraper follows robots.txt, extracts article text, and deduplicates content.

## Compliance Status

The existing split files predate source manifests, so their manual/download token split cannot be reconstructed reliably. Therefore the current artifact does not establish the required 20% manual-token fraction. Rebuild the splits with `preprocessing/prepare_dataset.py` to generate `datasets/source_manifest.json`, then report the measured fraction. The current corpus also requires additional collection before it can claim the 500M-token target.

## Cleaning and Splitting

The corrected rebuild pipeline applies Unicode NFC normalization, whitespace normalization, deterministic splitting (seed 42), and cross-split exact deduplication. The legacy splits were deduplicated separately and their current counts are reported in `nepali_dataset_statistics.md`.
