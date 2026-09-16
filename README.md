# Phase 2: Monolingual Transformer Language Models (Hindi & Nepali)

This repository contains the implementation, pretraining pipelines, and evaluation suite for building two independent ~25M parameter decoder-only Transformer language models from scratch in PyTorch.

- **Model H (Hindi)**: Higher-resource
- **Model L (Nepali)**: Lower-resource
-  **Model Weights & Configs**: [Hugging Face (manyajain111/lma-phase2)](https://huggingface.co/manyajain111/lma-phase2)

---

## Architecture Specifications

Both models feature a custom PyTorch Transformer built entirely from scratch:
- **Dimensions**: 6 Layers | 8 Heads ($d_k = 64$) | Hidden $d_{model} = 512$ | Feed-Forward = 1536
- **Features**: Learned positional & token embeddings (weight-tied), pre-LayerNorm, GELU activation, and causal self-attention with upper-triangular masking.
- **Model H (Hindi)**: ~24.22M parameters | 16,000 Vocab Size
- **Model L (Nepali)**: ~24.34M parameters | 16,240 Vocab Size

---

## Key Results

*(All comprehensive plots, loss curves, and attention heatmaps are available in the [`report/`](report/) directory).*

| Language / Model | Val Loss | Perplexity (PPL) | Bits-per-Byte | Repetition (Greedy) | Distinct-2 (Temp 1.0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hindi (Model H)** | 3.65 | **39.11** | 0.513 | 0.266 | 0.931 |
| **Nepali (Model L)** | 3.59 | **36.32** | 0.394 | 0.671 | 0.959 |

---

## Reproduction & Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Training from Scratch
```bash
# Train Hindi (Model H)
python hindi/train/train.py --config hindi/config/config.json --train_bin <hindi_train.bin> --val_bin <hindi_val.bin> --output_dir checkpoints/hindi

# Train Nepali (Model L)
python nepali/train/train.py --config nepali/config/config.json --train_bin <nepali_train.bin> --val_bin <nepali_val.bin> --output_dir checkpoints/nepali
```

### 3. Evaluation
```bash
# Evaluate Hindi (Swap `hindi` for `nepali` in paths to evaluate Model L)
python hindi/eval/evaluate.py \
    --checkpoint checkpoints/hindi/model_final.pt \
    --tokenizer hindi/tokenizer/tokenizer.json \
    --language hindi \
    --output_dir report/hindi
```

---

## Repository Layout


```text
.
├── hindi/ & nepali/          # Self-contained pipelines for Model H & Model L
│   ├── config/               # Hyperparameters (vocab, layers, dimensions)
│   ├── tokenizer/            # Custom trained BPE tokenizers
│   ├── model/                # Transformer architecture (from scratch in PyTorch)
│   ├── train/                # Training loop with AdamW, lr schedule, logs
│   └── eval/                 # Evaluators (PPL, BPB, BLEU, chrF, ROUGE-L, Attention)
├── report/                   # Graded deliverables
│   ├── phase2_report.pdf     # Complete scientific report
│   └── hindi/ & nepali/      # Loss curves, heatmaps, metric JSONs, generated samples
├── notebooks/                # End-to-end training and evaluation Jupyter notebooks
├── phase2/                   # Backwards-compatible workspace copy
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```