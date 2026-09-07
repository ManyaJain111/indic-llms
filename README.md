# Phase 2: Decoder-Only Language Model Pretraining (Hindi & Nepali)

## Model Checkpoints

Due to GitHub file size limits, the trained model `.pt` checkpoint files are hosted on Hugging Face.

[Hugging Face — LMA Phase 2](https://huggingface.co/manyajain111/lma-phase2)
---

## Model Summary

| Property               | Hindi                          | Nepali                         |
| ---------------------- | ------------------------------ | ------------------------------ |
| Architecture           | Decoder-only Transformer (GPT) | Decoder-only Transformer (GPT) |
| Parameters             | ~24.3M                         | ~24.3M                         |
| Vocabulary             | 16,000 BPE                     | 16,000 BPE                     |
| Training               | From scratch                   | From scratch                   |
| Framework              | PyTorch                        | PyTorch                        |
| Dataset                | Hindi LMA Corpus               | Nepali LMA Corpus              |
| Context / Architecture | Defined in model config        | Defined in model config        |

---

## Phase 2 Overview

The overall pipeline is:

```text
Phase 1
  │
  ├── Data Collection
  ├── Data Cleaning
  └── Tokenizer Training
          │
          ▼
Phase 2
  │
  ├── BPE Tokenization
  ├── Memory-Mapped Dataset
  ├── Decoder-Only Transformer
  ├── Pretraining from Scratch
  ├── Checkpointing
  │
  ▼
Evaluation
  │
  ├── Intrinsic Metrics
  ├── Text Generation
  ├── BLEU / ROUGE
  ├── Diversity Metrics
  └── Attention Analysis
```

---

## Directory Structure

```text
.
├── phase1/                              # Scrapers, data collection, and tokenizers from Phase 1
├── phase2/
│   ├── config/                          # Architecture and hyperparameter JSON files
│   │   ├── model_h_config.json          # Hindi GPT config (~24.3M parameters)
│   │   └── model_l_config.json          # Nepali GPT config (~24.3M parameters)
│   │
│   ├── src/                             # Main PyTorch implementation
│   │   ├── model.py                     # Decoder-only Transformer written from scratch
│   │   ├── dataset.py                   # Dataloader for memory-mapped binary files
│   │   ├── train.py                     # Pretraining script with checkpointing and LR schedule
│   │   ├── evaluate.py                  # Evaluation: PPL, BPB, BLEU, ROUGE, attention maps
│   │   └── tokenizer_wrapper.py         # BPE tokenizer wrapper
│   │
│   ├── tokenizers/                      # 16k-vocabulary BPE tokenizers
│   │   ├── hindi_tokenizer_16000.json
│   │   └── nepali_tokenizer_16000.json
│   │
│   ├── notebooks/                       # Kaggle notebooks for training and evaluation
│   │   ├── hindi_training_part1.ipynb
│   │   ├── hindi_training_part2.ipynb
│   │   ├── hindi_evaluation.ipynb
│   │   ├── nepali_training.ipynb
│   │   └── nepali_evaluation.ipynb
│   │
│   ├── reports/                         # Evaluation outputs
│   │   ├── hindi/                       # Hindi metrics, samples, and attention plots
│   │   └── nepali/                      # Nepali metrics, samples, and attention plots
│   │
│   └── requirements.txt
│
├── requirements.txt
└── README.md
```

---

## Setup and Installation

Install the required dependencies before running the scripts:

```bash
cd phase2
pip install -r requirements.txt
```

---

## Running the Code

### 1. Pretraining

To train the **Hindi** model locally or on a GPU cluster:

```bash
cd phase2

python src/train.py \
    --config config/model_h_config.json \
    --tokenizer tokenizers/hindi_tokenizer_16000.json \
    --dataset manyajain111/hindi-lma-corpus \
    --output_dir checkpoints/hindi \
    --batch_size 16 \
    --grad_accum 4 \
    --max_steps 30000 \
    --save_steps 1000 \
    --val_steps 1000
```

For the **Nepali** model, swap the configuration, tokenizer, and dataset:

```bash
cd phase2

python src/train.py \
    --config config/model_l_config.json \
    --tokenizer tokenizers/nepali_tokenizer_16000.json \
    --dataset manyajain111/nepali-lma-corpus \
    --output_dir checkpoints/nepali \
    --batch_size 16 \
    --grad_accum 4 \
    --max_steps 30000 \
    --save_steps 1000 \
    --val_steps 1000
```

---

### 2. Evaluation

Before running evaluation, download the trained checkpoints from Hugging Face and place them in:

```text
phase2/checkpoints/hindi/
phase2/checkpoints/nepali/
```

The evaluation script computes the relevant metrics, generates text samples, and produces attention heatmaps.

#### Hindi

```bash
python src/evaluate.py \
    --checkpoint checkpoints/hindi/model_final.pt \
    --tokenizer tokenizers/hindi_tokenizer_16000.json \
    --dataset manyajain111/hindi-lma-corpus \
    --language hindi \
    --output_dir reports/hindi
```

#### Nepali

```bash
python src/evaluate.py \
    --checkpoint checkpoints/nepali/model_final.pt \
    --tokenizer tokenizers/nepali_tokenizer_16000.json \
    --dataset manyajain111/nepali-lma-corpus \
    --language nepali \
    --output_dir reports/nepali
```

All evaluation outputs are saved under:

```text
phase2/reports/
```

---

## Deliverables


| Requirement                    | Location                                                             |
| ------------------------------ | -------------------------------------------------------------------- |
| **Transformer Implementation** | `phase2/src/model.py`                                                |
| **Multi-Head Attention**       | `phase2/src/model.py`                                                |
| **Causal Masking**             | `phase2/src/model.py`                                                |
| **Positional Embeddings**      | `phase2/src/model.py`                                                |
| **Model Configurations**       | `phase2/config/`                                                     |
| **Parameter Counts**           | `phase2/config/` and training logs                                   |
| **Training Scripts**           | `phase2/src/train.py`                                                |
| **Training Runs / Notebooks**  | `phase2/notebooks/`                                                  |
| **Model Weights**              | Hugging Face repository                                              |
| **Training Logs**              | `train_log.csv` / `phase2/notebooks/`                                |
| **Hindi Evaluation Metrics**   | `phase2/reports/hindi/results.json`                                  |
| **Nepali Intrinsic Metrics**   | `phase2/reports/nepali/intrinsic_metrics.json`                       |
| **Nepali Generation Quality**  | `phase2/reports/nepali/generation_quality.json`                      |
| **Generated Samples**          | `generated_samples.json` in respective report directories            |
| **Diversity Metrics**          | `diversity_metrics.json` in respective report directories            |
| **Attention Heatmaps**         | PNG files in the respective report directories                       |
| **Attention Statistics**       | Attention statistics JSON files in the respective report directories |
| **Resource Comparison**        | Final submitted project report                                       |

---

## Evaluation Outputs

The evaluation pipeline produces:

* **Perplexity (PPL)**
* **Bits Per Byte (BPB)**
* **BLEU**
* **ROUGE**
* **Generated text samples**
* **Generation diversity metrics**
* **Attention heatmaps**
* **Attention statistics**

These outputs are organized separately for Hindi and Nepali under `phase2/reports/`.

---
