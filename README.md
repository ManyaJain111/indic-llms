# Phase 2: Decoder-Only Language Model Pretraining (Hindi & Nepali)

This repository contains the code, configurations, notebooks, and evaluation reports for pretraining and evaluating ~25M parameter decoder-only Transformer models (GPT architecture) from scratch on Hindi and Nepali corpora.

---

## 📁 Repository Structure

```
.
├── config/                  # Model architectures and hyperparameter configurations
│   ├── model_h_config.json  # Hindi GPT model config (~24.3M parameters)
│   └── model_l_config.json  # Nepali GPT model config (~24.3M parameters)
├── src/                     # Core PyTorch source implementation
│   ├── model.py             # From-scratch Decoder-only Transformer implementation
│   ├── dataset.py           # Memory-mapped binary dataset loader
│   ├── train.py             # Pretraining script (DataParallel, LR schedule, checkpointing)
│   ├── evaluate.py          # Comprehensive evaluation pipeline (PPL, BPB, BLEU, ROUGE, Attention)
│   └── tokenizer_wrapper.py # BPE Tokenizer wrapper with memoization caching
├── tokenizers/              # Custom 16k vocabulary BPE tokenizers
│   ├── hindi_tokenizer_16000.json
│   └── nepali_tokenizer_16000.json
├── notebooks/               # Standalone Kaggle pretraining & evaluation notebooks
│   ├── hindi_training.ipynb     # Standalone Hindi pretraining notebook (up to 60k steps)
│   ├── hindi_evaluation.ipynb   # Standalone Hindi evaluation notebook
│   └── nepali.ipynb         # End-to-end Nepali pretraining & evaluation notebook
├── reports/                 # Output metrics, generation samples, and attention plots
│   ├── hindi/               # Hindi evaluation metrics, generated samples, & attention maps
│   └── nepali/              # Nepali evaluation metrics, generated samples, & attention maps
├── phase1/                  # Phase 1 data collection, scrapers, and tokenizer training
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Requirements & Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Pretraining & Evaluation Workflow

### 1. Pretraining

To run pretraining locally or on a server for **Hindi**:

```bash
python src/train.py \
    --config config/model_h_config.json \
    --tokenizer tokenizers/hindi_tokenizer_16000.json \
    --dataset manyajain111/hindi-lma-corpus \
    --output_dir checkpoints/hindi \
    --batch_size 16 --grad_accum 4 \
    --max_steps 30000 --save_steps 1000 --val_steps 1000
```

To run pretraining for **Nepali**:

```bash
python src/train.py \
    --config config/model_l_config.json \
    --tokenizer tokenizers/nepali_tokenizer_16000.json \
    --dataset manyajain111/nepali-lma-corpus \
    --output_dir checkpoints/nepali \
    --batch_size 16 --grad_accum 4 \
    --max_steps 30000 --save_steps 1000 --val_steps 1000
```

### 2. Evaluation

To evaluate the trained model on test data, generate text samples, and visualize attention heatmaps:

```bash
# Hindi Evaluation
python src/evaluate.py \
    --checkpoint checkpoints/hindi/model_final.pt \
    --tokenizer tokenizers/hindi_tokenizer_16000.json \
    --dataset manyajain111/hindi-lma-corpus \
    --language hindi \
    --output_dir reports/hindi

# Nepali Evaluation
python src/evaluate.py \
    --checkpoint checkpoints/nepali/model_final.pt \
    --tokenizer tokenizers/nepali_tokenizer_16000.json \
    --dataset manyajain111/nepali-lma-corpus \
    --language nepali \
    --output_dir reports/nepali
```

---

## 📊 Summary of Evaluation Metrics

The final evaluation metrics, generated samples, and attention visualization heatmaps are stored under `reports/hindi/` and `reports/nepali/`.
