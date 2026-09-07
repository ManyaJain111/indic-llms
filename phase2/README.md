# Phase 2: Monolingual Transformer Language Models (Hindi & Nepali)

This repository contains the implementation, pretraining, and evaluation suite for Phase 2: building two completely independent, monolingual decoder-only Transformer language models (~25M parameters each) from scratch in PyTorch.

- **Model H (Higher-resource)**: Hindi
- **Model L (Lower-resource)**: Nepali

---

## 1. Directory Layout

```
phase2/
├── config/
│   ├── model_h_config.json      # Model H (Hindi) architecture hyperparameters
│   └── model_l_config.json      # Model L (Nepali) architecture hyperparameters
├── tokenizers/
│   ├── hindi_tokenizer_16000.json
│   └── nepali_tokenizer_16000.json
├── checkpoints/
│   ├── hindi/
│   │   ├── model_final.pt       # Pretrained weights at step 61,036
│   │   └── train_log.csv        # Complete step-by-step training and validation loss log
│   └── nepali/
│       ├── model_final.pt       # Pretrained weights at step 61,036
│       ├── nepali_checkpoints/  # Intermediate checkpoints (step 61,000, latest.pt)
│       └── train_log.csv        # Complete training log
├── src/
│   ├── model.py                 # Transformer architecture (MHA, Causal Masking, Pre-LN, Weight Tying)
│   ├── train.py                 # Training pipeline with AdamW, lr schedule, checkpoint resume
│   ├── evaluate.py              # Evaluation script (PPL, BPB, Generation, Attention analysis)
│   ├── dataset.py               # Memory-mapped binary token dataloader
│   └── tokenizer_wrapper.py     # Custom BPE tokenizer wrapper
├── reports/
│   ├── report.md                # Consolidated Phase 2 technical report
│   ├── comparison_loss_curve.png
│   ├── hindi/                   # Loss curves, attention heatmaps, metric JSONs, generated samples
│   └── nepali/                  # Loss curves, attention heatmaps, metric JSONs, generated samples
└── notebooks/
    ├── hindi_training_part1.ipynb
    ├── hindi_training_part2.ipynb
    ├── hindi_evaluation.ipynb
    ├── nepali_training.ipynb
    └── nepali_evaluation.ipynb
```

---

## 2. Architecture Specifications (~25M Parameters)

- **Input Representation**: Learned positional embeddings + token embeddings with weight tying.
- **Attention**: Custom multi-head causal self-attention with scaled dot-product attention and upper-triangular masking (hBc\infty$).
- **Blocks**: 6 Transformer layers, pre-LayerNorm, 8 heads ( = 64$), hidden dimension 512, feed-forward dimension 1536 ( 	imes d_{	ext{model}}$), GELU activation.
- **Model H Parameters**: ~24.22M
- **Model L Parameters**: ~24.34M

---

## 3. Reproduction Instructions

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training from Scratch
To train Model H (Hindi):
```bash
python src/train.py     --train_bin <path_to_hindi_train.bin>     --val_bin <path_to_hindi_val.bin>     --config config/model_h_config.json     --output_dir checkpoints/hindi
```

### Evaluation & Metrics
To run intrinsic evaluation (PPL/BPB), generation quality (BLEU-4, chrF, ROUGE-L), and attention analysis:
```bash
python src/evaluate.py     --checkpoint checkpoints/hindi/model_final.pt     --tokenizer tokenizers/hindi_tokenizer_16000.json     --language hindi     --output_dir reports/hindi
```

---

## 4. Key Results Summary

| Language / Model | Validation Loss | Perplexity (PPL) | Bits-per-Byte (BPB) | Repetition Rate (Greedy) | Distinct-2 (Temp 1.0) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hindi (Model H)** | 3.6487 | 39.11 | 0.5131 | 0.2659 | 0.9313 |
| **Nepali (Model L)** | 3.5910 | 36.32 | 0.3939 | 0.6712 | 0.9586 |

*Detailed comparative analysis, attention heatmaps, and metric breakdowns are available in [reports/report.md](reports/report.md).*
