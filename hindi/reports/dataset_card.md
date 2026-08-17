# Hindi Dataset Card

## Language Choice

Hindi is an appropriate higher-resource Indian language for Model H because it has large public monolingual sources, including Hindi Wikipedia, Indic corpora, subtitles, government releases, and news sites.

## Current Corpus

`datasets/stats.json` records 5,927,635 lines and 250,491,092 whitespace-delimited words. The held-out tokenizer fertility is 1.24 tokens per word, so the current corpus is estimated at roughly 311M tokenizer tokens. This is below the 500M target; an exact post-tokenization total must be generated after the corpus rebuild and final tokenizer training.

## Sources and Collection Method

The corrected rebuild pipeline records Hindi Wikipedia, subtitles, Indic corpus material, and PIB releases as downloaded sources. The project Hindi news scraper is classified as manual collection.

## Compliance Status

The existing splits predate source manifests. Their manual/download token share therefore cannot be demonstrated, and the required 20% manual-token fraction is not yet established. Rebuild with `preprocessing/prepare_dataset.py` to emit `datasets/source_manifest.json`, then collect enough manual text and additional public Hindi text to meet both requirements.

## Cleaning and Splitting

The corrected pipeline applies Unicode NFC normalization, exact-content deduplication, and deterministic 90/5/5 splits using seed 42. The legacy Hindi splits must be regenerated for those corrections to apply to the final corpus.
