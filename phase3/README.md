# Synthetic Comparative-Reasoning Dataset (Model H / Model L, Phase 3)

Programmatically generated & NCERT Exemplar-seeded finetuning corpus for transitive/comparative
reasoning, in Hindi (Model H) and Nepali (Model L). Total size: **9,000 items per language**
(6,300 train, 1,350 val, 1,350 test). No external reasoning benchmark was used — ground truth is
constructed and programmatically verified for 100% of examples.

## Files & Structure

```
phase3/
├── generate_6k_dataset.py       # Full dataset generator (NCERT seeds + LLM augmentation + templates)
├── ingest_ncert.py              # NCERT problem extraction script (Physics, Economics, Bio)
├── llm_augmentation.py          # Gemini-based few-shot translation and augmentation
├── evaluate_diversity.py        # Evaluates distinct trigram ratio & diversity improvement
├── merge_datasets.py            # Dataset merging & stats calculator
├── vocab_hi.py                  # Hindi entity pools, attributes, templates
├── vocab_ne.py                  # Nepali entity pools, attributes, templates
└── data/
    ├── seeds/
    │   ├── ncert_seeds_hi.jsonl # 15 authentic NCERT exemplar seeds (Hindi)
    │   └── ncert_seeds_ne.jsonl # 15 authentic NCERT exemplar seeds (Nepali)
    ├── hi/
    │   ├── train.jsonl          # 6,300 records (Model H train)
    │   ├── val.jsonl            # 1,350 records (Model H val)
    │   ├── test.jsonl           # 1,350 records (Model H test)
    │   └── stats.json           # Breakdown by question type & source
    └── ne/
        ├── train.jsonl          # 6,300 records (Model L train)
        ├── val.jsonl            # 1,350 records (Model L val)
        ├── test.jsonl           # 1,350 records (Model L test)
        └── stats.json           # Breakdown by question type & source
```

## Leakage Avoidance & Governed Entity Namespaces

All entities are strictly partitioned **70/15/15** once with a fixed internal seed (`_SPLIT_SEED = 20260913`):
1. **Person Names:** `PERSON_NAMES_SPLIT` (50 names partitioned into Train: 34, Val: 8, Test: 8).
2. **Item Nouns:** `ITEM_NAMES_SPLIT` (30 objects partitioned into Train: 20, Val: 5, Test: 5).
3. **Geography Entities:** `GEOGRAPHY_NAMES_SPLIT` (32 cities/states/regions in Devanagari partitioned into Train: 22, Val: 5, Test: 5).
4. **Science / Substance Entities:** `SCIENCE_NAMES_SPLIT` (32 metals/liquids/materials/vehicles in Devanagari partitioned into Train: 22, Val: 5, Test: 5).

No Latin/English letter identifiers (`A, B, C...`) are used. All entities are 100% native Devanagari tokens, preventing byte-fallback tokenization and maintaining perfect vocabulary alignment with Model H and Model L's tokenizers. **There is zero entity or prompt overlap across train, val, and test splits.**


## Question Types (13 Comparative & Structural Reasoning Families)

All 13 types produce clean, canonical string labels enabling uniform programmatic verification:

| type | entities | given | asks | label shape |
|---|---|---|---|---|
| `direct_value_more` | 2 | numeric values | which has the greater value | Single entity |
| `direct_value_less` | 2 | numeric values | which has the smaller value | Single entity |
| `transitive_relation` | 3 | pure relations ($A>B, B>C$) | max / min / A-vs-C chaining | Single entity |
| `order_3way_max` | 3 | numeric values | superlative maximum | Single entity |
| `order_3way_min` | 3 | numeric values | superlative minimum | Single entity |
| `transitive_chain_4way` | 4 | pure relations ($A>B>C>D$) | 4-hop multi-hop reasoning | Single entity |
| `order_4way_rank` | 4 | numeric values | 2nd largest or 2nd smallest | Single entity |
| `threshold_comparison` | 2 | numeric values + threshold $T$ | which entity exceeds/falls below limit | Single entity |
| `difference_comparison` | 3 | reference $e_1$ + targets $e_2, e_3$ | which target has larger/smaller gap from $e_1$ | Single entity |
| `temporal_ordering` | 3 | sequential events ($A$ before $B$, $B$ before $C$) | who arrived / finished first or last | Single entity |
| `contradiction_detection` | 2 | two relational statements | whether statements contradict or agree | Canonical boolean (`"हाँ"` / `"नहीं"`, `"हो"` / `"होइन"`) |
| `rule_classification` | 1 | property value + rule boundary | which category the entity belongs to | Category string |
| `multistep_if_then` | 3 | conditional inequality chain ($A>B \land B>C$) | deductive extreme between $A$ and $C$ | Single entity |

Fully deterministic given `--seed` (data sampling) — the train/val/test
pool split of entities and templates uses its own fixed internal seed
(`_SPLIT_SEED` in the vocab files) so it does not vary with `--seed`.

## Record format (JSONL, one object per line)

```json
{
  "id": "hi_test_000004",
  "split": "test",
  "type": "direct_value_more",
  "prompt": "मान लीजिए तकिया की कीमत 785 रुपये है और अनार की कीमत 2221 रुपये है। ज़्यादा कीमत किसकी है?",
  "answer": "अनार की कीमत ज़्यादा है।",
  "label": "अनार",
  "meta": {"attribute": "कीमत", "values": {"तकिया": 785, "अनार": 2221}}
}
```
- `prompt` — the model input.
- `answer` — a natural-language completion sentence (use for
  generation-style finetuning / qualitative inspection).
- `label` — a short canonical answer (the winning entity's name, or the
  literal string for "equal" / "yes" / "no" / category) for exact-match scoring.
- `meta` — the ground-truth numbers/relations used to build the item;
  not meant to be shown to the model, only for analysis/debugging.

## Dataset size (per language: 9,000 total)

* **Train:** 6,300 records (70%)
* **Val:** 1,350 records (15%)
* **Test:** 1,350 records (15%)

Per-type breakdown is in `phase3/data/{lang}/stats.json`.

## Avoiding train/test leakage

Three independent axes are each split 70% / 15% / 15% up front, with a
fixed split seed, before any example is generated:

1. **Entity names** — person-name pool and item-noun pool are each
   partitioned into train/val/test subsets with **zero overlap**. An
   entity that appears in test never appears in train or val, and
   vice versa. This is verified programmatically (see below) — every
   generated record's entities are drawn only from that split's pool.
2. **Template phrasings** — for every question type, the list of
   surface-form templates is likewise partitioned 70/15/15 with no
   overlap, so test also exercises phrasings the model never saw
   during finetuning (not just new entity/value combinations of a
   memorized wording).
3. **Numeric values** — resampled fresh per example from the same
   range for all splits (values are cheap to interpolate/extrapolate,
   so the more meaningful generalization tests are (1) and (2) above).

Verification performed on the generated files (see repo commit): for
every record in every split, all entity names referenced in `meta`
are a subset of that split's allowed name pool — checked to be true
for 100% of records in both languages. Train/val/test entity pools
and template pools were also checked pairwise for zero set overlap.

Small template lists (as few as 3 raw templates for `order_3way_*`)
are handled by a split rule that guarantees at least one template per
split whenever the raw list has ≥3 entries, rather than a naive
`int(n*0.15)` that can round down to zero.

Within a split, generation also rejects an exact-duplicate `prompt`
string, so the same example is never repeated verbatim.

## Language notes

- Hindi (Model H): Devanagari script, byte-level BPE tokenizer
  (vocab 32k) per the Phase 1 setup. Comparative wording avoids
  gender-agreeing adjectives (e.g. "ज़्यादा/कम" rather than "बड़ा/छोटा")
  since entities span both persons and inanimate items.
- Nepali (Model L): Devanagari script, Unicode character-level BPE
  tokenizer (vocab 37k) per the Phase 1 setup. Uses the "X भन्दा बढी/कम"
  comparative construction.

## Protocol notes (for the finetuning run itself)

- Start from each language's own Phase-2 pretrained checkpoint; keep
  that language's tokenizer/vocab fixed (no retraining the tokenizer).
- Finetune on `train.jsonl`, select checkpoint on `val.jsonl`, report
  final numbers on `test.jsonl` only once.
- Save finetuned checkpoints in the same resume-capable format used
  for pretraining; put the Google Drive links in the top-level
  project README (not this one).
- Report pretrained-vs-finetuned accuracy/exact-match on `test.jsonl`
  (compare `label` to the model's generated answer), broken down by
  `type`, plus qualitative successes/failures and an H-vs-L comparison.
