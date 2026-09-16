# Language Models and Agents — Individual Project

**Manya Jain**

- Code: [github.com/manyajain111/individual-project-ManyaJain111](https://github.com/manyajain111/individual-project-ManyaJain111)
- Checkpoints: [huggingface.co/manyajain111/LMA-Individual-Project](https://huggingface.co/manyajain111/LMA-Individual-Project)

---


| | Model H | Model L |
| :--- | :--- | :--- |
| Language | Hindi (higher-resource) | Nepali (lower-resource) |
| Vocab | ~16k BPE | ~16k BPE |
| Layers / heads | 6 / 8 | 6 / 8 |
| d_model / d_ff | 256 / 1024 | 256 / 1024 |
| Context length | 256 | 256 |
| Parameters | ~24.6M | ~24.6M |

The work is split into three phases plus a bonus experiment:

1. **Phase 1** — crawling and cleaning the corpora, then training the BPE tokenizers.
2. **Phase 2** — pretraining both models and evaluating them (perplexity, bits-per-byte, BLEU, chrF, ROUGE-L, repetition stats) plus a look at what the attention heads are doing.
3. **Phase 3** — generating a synthetic reasoning dataset with a grammar, finetuning both models on it, and testing whether they can do transitive comparisons. I also compared attention maps before and after finetuning.
4. **Bonus** — retrained the Hindi model with the positional embeddings removed, to see how badly it breaks.

---

## Repository layout

```
├── data/                                 # reasoning datasets
│   ├── hi/                               # Hindi train/val/test + stats
│   ├── ne/                               # Nepali train/val/test + stats
│   └── simple/                           # the easier eval prompts
├── notebooks/
│   ├── phase3_reasoning_finetuning.ipynb # finetuning + eval
│   ├── bonus_training_final.ipynb        # ablation training
│   └── bonus_eval_final.ipynb            # ablation eval
├── scripts/
│   ├── generate_reasoning_dataset.py     # grammar-based dataset generator
│   ├── generate_easy_test.py             # 60-item simplified benchmark
│   ├── vocab_hi.py, vocab_ne.py          # Hindi and Nepali lexicons
│   └── generate_all_plots.py             # attention + entropy plots
└── report/
    ├── individual_project_report.pdf     # final submission report
    ├── individual_project_report.tex     # LaTeX source
    ├── phase3_report.md
    ├── bonus_report.md
    ├── eval_results.json                 # all reasoning eval numbers
    ├── finetune_log_combined.csv
    ├── bonus_train_log.csv
    └── plots/                            # named by phase, e.g.
        ├── phase2_hindi_loss_curve.png
        ├── phase2_*_attn_layer*.png
        ├── phase3_finetuning_loss_curves.png
        ├── phase3_*_attn_compare_layer*.png
        └── bonus_ablation_*.png
```

**A note on branches**: this branch (`Phase-3`) holds the final Phase 3 and bonus code, report and metrics, as the submission asked for. The Phase 1 scrapers and tokenizer trainers live on `Phase-1`, and the Phase 2 pretraining and eval scripts live on `Phase-2`.

---

## Checkpoints and logs

The `.pt` files are too big for git, so they are on the Hugging Face Hub.

**Phase 2 pretrained** ([manyajain111/lma-phase2](https://huggingface.co/manyajain111/lma-phase2)):

- Hindi: [`hindi/model_final.pt`](https://huggingface.co/manyajain111/lma-phase2/blob/main/hindi/model_final.pt)
- Nepali: [`nepali/model_final.pt`](https://huggingface.co/manyajain111/lma-phase2/blob/main/nepali/model_final.pt)
- Logs: `hindi/train/train_log.csv` and `nepali/train/train_log.csv` in the repo, mirrored on HF as [`logs/hindi_pretrain_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/hindi_pretrain_log.csv) and [`logs/nepali_pretrain_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/nepali_pretrain_log.csv)

**Phase 3 finetuned** ([manyajain111/LMA-Individual-Project](https://huggingface.co/manyajain111/LMA-Individual-Project)):

- Hindi: [`finetuned/hindi_model_finetuned.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/finetuned/hindi_model_finetuned.pt)
- Nepali: [`finetuned/nepali_model_finetuned.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/finetuned/nepali_model_finetuned.pt)
- Per-epoch Hindi checkpoints: [epoch 1](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/checkpoints/phase3/finetuned/hindi/model_finetuned_epoch_1.pt), [epoch 2](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/checkpoints/phase3/finetuned/hindi/model_finetuned_epoch_2.pt), [epoch 3](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/checkpoints/phase3/finetuned/hindi/model_finetuned_epoch_3.pt)
- Both models are also mirrored under [`checkpoints/phase3/finetuned/`](https://huggingface.co/manyajain111/LMA-Individual-Project/tree/main/checkpoints/phase3/finetuned/hindi)
- Log: [`logs/phase3_reasoning_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/phase3_reasoning_log.csv)

**Bonus ablation** (everything under [`ablation/`](https://huggingface.co/manyajain111/LMA-Individual-Project/tree/main/ablation)):

- Final weights: [`ablation/model_ablated.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated.pt)
- Intermediate checkpoints every 10k steps: [10k](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_10000.pt) · [20k](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_20000.pt) · [30k](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_30000.pt) · [40k](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_40000.pt) · [50k](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_50000.pt) · [60k](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_60000.pt)
- Log: [`ablation/bonus_ablation_train_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/bonus_ablation_train_log.csv)
- Metrics: [`ablation/results.json`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/results.json)
- Plots: [loss curve](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/plots/bonus_loss_curve.png) · [attention entropy and distance](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/plots/attn_entropy_distance.png) · [generation diversity](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/plots/generation_diversity_metrics.png)

---

## How to reproduce

### Setup

```bash
git clone https://github.com/manyajain111/individual-project-ManyaJain111.git
cd individual-project-ManyaJain111
pip install -r requirements.txt
```

### Phases 1 and 2

These are on their own branches:

```bash
git checkout Phase-1   # crawlers and tokenizer training
git checkout Phase-2   # pretraining loop and eval scripts
```

### Phase 3: reasoning data and finetuning

Generate the datasets:

```bash
python scripts/generate_reasoning_dataset.py
python scripts/generate_easy_test.py
```

This writes the splits into `data/hi/` and `data/ne/`. The entity sets across train, val and test are disjoint, so nothing leaks between splits.

Then run `notebooks/phase3_reasoning_finetuning.ipynb`. I ran it on Kaggle; any CUDA GPU should work.

### Bonus: positional embedding ablation

1. `notebooks/bonus_training_final.ipynb` trains the model with positional embeddings stripped out, for 60,000 steps. It checkpoints every 10,000 steps and saves the optimizer state, so you can resume if the session dies (which it did, twice).
2. `notebooks/bonus_eval_final.ipynb` computes the PPL, BPB, 4-gram repetition rate and the attention heatmaps.

---

## Results

### Pretraining (Phase 2)

| Model | Language | Vocab | Val loss | PPL | BPB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| H | Hindi | 16,000 | 3.6487 | 39.11 | 0.5131 |
| L | Nepali | 16,000 | 3.5210 | 33.82 | 0.3939 |

Nepali came out with the lower perplexity, which surprised me given it's the lower-resource language. I think this is mostly a property of the corpus rather than the model: the Nepali data I could crawl is narrower in domain and more repetitive, so it's easier to predict. The BPB gap points the same way. This is worth keeping in mind before reading the numbers as "Nepali is easier".

### Reasoning finetuning (Phase 3)

I report two numbers. **Strict exact match** is literal string equality between prediction and target. **Substring match** checks whether the target answer appears anywhere in the generated text.

| Metric | Model H (Hindi) | Model L (Nepali) |
| :--- | :--- | :--- |
| Strict exact match, pretrained | 0.00% | 0.00% |
| Strict exact match, finetuned | 0.00% | 0.00% |
| Substring match, pretrained | 1.04% | 1.93% |
| Substring match, finetuned | 27.26% | 26.67% |
| Simplified 60-item test | 50.00% | — |

Strict exact match is 0% everywhere, and that's a measurement artefact rather than a total failure. The model is an unconstrained autoregressive decoder, so it keeps generating a full sentence instead of stopping after the entity name. It produces `उत्तर: ओडिशा की आय अधिक है।` when the target is just `ओडिशा`. The string never matches even when the answer is right. The raw generations are in `report/eval_results.json` if you want to check this.

Under substring match, finetuning takes both models from roughly chance (~1-2%) to around 27%. That's a real gain but still well below what I'd want, and the 50% on the simplified two-choice test is exactly chance, so I can't claim the Hindi model is genuinely doing the comparison there. At 24.6M parameters on synthetic data this is about what I'd expect; a proper constrained decoder or a scoring-based eval would give a much cleaner read on what the model actually knows.

### Bonus: what happens without positional embeddings

I retrained the Hindi model with the positional embedding table removed entirely, keeping everything else identical.

| Metric | With positions | Without positions | Change |
| :--- | :--- | :--- | :--- |
| PPL | 39.11 | 58.54 | +49.7% |
| Val loss | 3.6487 | 4.0697 | worse |
| 4-gram repetition | 26.59% | 68.52% | much worse |
| Distinct-1 | 0.4270 | 0.1703 | much worse |
| Avg attention distance | 7.15 tokens | 2.18 tokens | far more local |
| Layer 0 attention | spread out | sharp diagonal | collapsed |

The perplexity hit is the least interesting part. The generation quality is where it really shows: the ablated model loops, repeating the same 4-grams two thirds of the time, and its distinct-1 drops by more than half. The attention plots explain why. Without positional information, layer 0 collapses onto the diagonal and the model attends mostly to the current token, so average attention distance falls from about 7 tokens to about 2. It can still learn which tokens co-occur, but it has no way to represent the order they came in, so it can't tell a repeat from a continuation.

Full write-ups for Phase 3 and the ablation are in `report/phase3_report.md` and `report/bonus_report.md`, and everything is consolidated in `report/individual_project_report.pdf`.
