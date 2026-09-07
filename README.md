# Phase 2: Monolingual Transformer Language Models (Hindi & Nepali)

This repository contains the implementation, pretraining pipelines, and comprehensive evaluation suite for **Phase 2**: building and analyzing two completely independent decoder-only Transformer language models (~25M parameters each) from scratch in PyTorch.

- **Model H (Higher-resource)**: Hindi
- **Model L (Lower-resource)**: Nepali

---

## 1. Repository Layout (as per Project Guidelines)

```
.
├── README.md               # Reproduction steps, key results, and model download links
├── requirements.txt        # Python package dependencies
├── hindi/                  # Self-contained Hindi pipeline (Model H)
│   ├── config/             # Model H hyperparameters (vocab 16,000, 6 layers, 8 heads, d_model 512)
│   ├── tokenizer/          # Custom trained BPE tokenizer
│   ├── model/              # Custom Transformer architecture (from scratch in PyTorch)
│   ├── train/              # Training loop with AdamW, lr schedule, train_log.csv
│   └── eval/               # Evaluation script for PPL, BPB, BLEU, chrF, ROUGE-L, Attention
├── nepali/                 # Self-contained Nepali pipeline (Model L)
│   ├── config/             # Model L hyperparameters (vocab 16,240, 6 layers, 8 heads, d_model 512)
│   ├── tokenizer/          # Custom trained BPE tokenizer
│   ├── model/              # Custom Transformer architecture (from scratch in PyTorch)
│   ├── train/              # Training loop with AdamW, lr schedule, train_log.csv
│   └── eval/               # Evaluation scripts & Devanagari ROUGE-L calculation
├── report/                 # Graded Phase 2 deliverables
│   ├── phase2_report.pdf   # Complete student-written scientific report
│   ├── phase2_report.tex   # LaTeX source for report
│   ├── report.md           # Markdown technical report
│   ├── comparison_loss_curve.png
│   ├── hindi/              # Loss curves, attention heatmaps, metric JSONs, generated samples
│   └── nepali/             # Loss curves, attention heatmaps, metric JSONs, generated samples
├── notebooks/              # Complete end-to-end training and evaluation notebooks
│   ├── hindi_training_part1.ipynb
│   ├── hindi_training_part2.ipynb
│   ├── hindi_evaluation.ipynb
│   ├── nepali_training.ipynb
│   └── nepali_evaluation.ipynb
└── phase2/                 # Backwards-compatible workspace copy
```

---

## 2. Model Checkpoints (Downloads)

The pretrained model weights and configurations for both **Model H** and **Model L** have been uploaded and released on Hugging Face:

- **Hugging Face Model Hub**: [https://huggingface.co/manyajain111/lma-phase2](https://huggingface.co/manyajain111/lma-phase2)

**Files available:**
- `hindi/model_final.pt` (~98.4 MB) — Hindi model weights at step 61,036
- `hindi/config.json` — Model H architecture config
- `hindi/tokenizer.json` — Hindi BPE tokenizer
- `nepali/model_final.pt` (~98.9 MB) — Nepali model weights at step 61,036
- `nepali/config.json` — Model L architecture config
- `nepali/tokenizer.json` — Nepali BPE tokenizer
- `model.py` — Standalone PyTorch model class for inference

---

## 3. Architecture Specifications (~25M Parameters)

- **Input Representation**: Learned positional embeddings + token embeddings with weight tying.
- **Attention**: Custom multi-head causal self-attention with scaled dot-product attention and upper-triangular masking.
- **Blocks**: 6 Transformer layers, pre-LayerNorm, 8 heads ($d_k = 64$), hidden dimension 512, feed-forward dimension 1536 ($3 \times d_{\text{model}}$), GELU activation.
- **Model H Parameters**: ~24.22M (Vocab: 16,000)
- **Model L Parameters**: ~24.34M (Vocab: 16,240)

---

## 4. Key Results Summary

| Language / Model | Validation Loss | Perplexity (PPL) | Bits-per-Byte (BPB) | Repetition Rate (Greedy) | Distinct-2 (Temp 1.0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hindi (Model H)** | 3.6487 | **39.11** | 0.5131 | 0.2659 | 0.9313 |
| **Nepali (Model L)** | 3.5910 | **36.32** | 0.3939 | 0.6712 | 0.9586 |

All plots, loss curves, and attention heatmaps are available in [`report/`](report/) and compiled into [`report/phase2_report.pdf`](report/phase2_report.pdf).

---

## 5. Reproduction Instructions

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training from Scratch
To train Hindi (Model H):
```bash
python hindi/train/train.py \
    --train_bin <path_to_hindi_train.bin> \
    --val_bin <path_to_hindi_val.bin> \
    --config hindi/config/config.json \
    --output_dir checkpoints/hindi
```

To train Nepali (Model L):
```bash
python nepali/train/train.py \
    --train_bin <path_to_nepali_train.bin> \
    --val_bin <path_to_nepali_val.bin> \
    --config nepali/config/config.json \
    --output_dir checkpoints/nepali
```

### Evaluation
To evaluate Model H (Hindi):
```bash
python hindi/eval/evaluate.py \
    --checkpoint checkpoints/hindi/model_final.pt \
    --tokenizer hindi/tokenizer/tokenizer.json \
    --language hindi \
    --output_dir report/hindi
```
