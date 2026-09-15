# Language Models and Agents — Individual Project
**Student**: Manya Jain  
**Repository**: [Language Models and Agents — Monolingual Hindi & Nepali Transformers](https://github.com/manyajain111/individual-project-ManyaJain111)  
**Model Hub (Hugging Face)**: [manyajain111/LMA-Individual-Project](https://huggingface.co/manyajain111/LMA-Individual-Project)

---

## Executive Summary
This repository contains the end-to-end implementation for building, training, and evaluating two independent, monolingual autoregressive Transformer language models implemented from scratch in PyTorch (without using Hugging Face `transformers` or `torch.nn.Transformer` primitives):
- **Model H (Higher-resource)**: Hindi (~16,000 subword BPE vocabulary, 6 layers, 8 heads, $d_{model}=256$, $d_{ff}=1024$, context length 256, ~24.6M parameters)
- **Model L (Lower-resource)**: Nepali (~16,000 character/byte-level BPE vocabulary, 6 layers, 8 heads, $d_{model}=256$, $d_{ff}=1024$, context length 256, ~24.6M parameters)

The project spans:
1. **Phase 1**: Web crawling, dataset collection, cleaning/preprocessing, and custom BPE tokenizer training.
2. **Phase 2**: First-principles GPT-style pretraining, evaluation (Perplexity, Bits-per-Byte, BLEU, chrF, ROUGE-L, repetition diagnostics), and multi-head attention analysis.
3. **Phase 3**: Grammar-based programmatic reasoning dataset synthesis, instruction/reasoning finetuning, transitive logic evaluation, and pre- vs. post-finetuning attention comparison.
4. **Bonus Task**: Positional embedding ablation study investigating representational collapse and degenerate sequence generation without positional inductive bias.

---

## Repository Structure

```
├── hindi/                                # Complete Hindi pipeline
│   ├── config/                           # Model H architecture configuration
│   ├── data/                             # Train/val/test splits & tokenized memmaps
│   ├── data_collection/                  # News scrapers, Wikipedia & OSCAR extraction
│   ├── preprocessing/                    # Devanagari text normalization & filtering
│   ├── tokenizer/                        # Trained 16k vocabulary and tokenizer JSON/PKL
│   ├── tokenizer_training/               # Tokenizer training scripts
│   ├── model/                            # First-principles Transformer implementation
│   ├── train/                            # Pretraining & Phase 3 finetuning notebooks/scripts
│   └── eval/                             # Evaluation & attention heatmap scripts
├── nepali/                               # Complete Nepali pipeline (parallel structure)
│   ├── config/                           # Model L architecture configuration
│   ├── data/                             # Train/val/test splits & tokenized memmaps
│   ├── data_collection/                  # Nepali news & Wikipedia scrapers
│   ├── preprocessing/                    # Nepali text normalization & cleaning
│   ├── tokenizer/                        # Trained 16k vocabulary and tokenizer JSON/PKL
│   ├── tokenizer_training/               # Tokenizer training scripts
│   ├── model/                            # First-principles Transformer implementation
│   ├── train/                            # Pretraining & finetuning scripts
│   └── eval/                             # Evaluation scripts
├── phase1/                               # Modular Phase 1 datasets, tokenizers, scrapers
├── phase2/                               # Phase 2 pretraining source code, configs, reports
├── phase3/                               # Phase 3 reasoning dataset generator & notebooks
│   ├── data/hi/ & data/ne/               # Synthetic reasoning splits (train/val/test JSONL)
│   ├── generate_reasoning_dataset.py     # Programmatic grammar-based generator
│   ├── phase3_final.ipynb                # Reasoning finetuning notebook
│   ├── bonus_training_final.ipynb        # Bonus ablation training notebook
│   ├── bonus_eval_final.ipynb            # Bonus ablation evaluation notebook
│   └── report.md                         # Dedicated Phase 3 & Bonus report
├── report/                               # Reports, loss curves, and attention heatmap assets
│   ├── report.md                         # Full Phase 2 & pretraining evaluation report
│   ├── phase3_plots/                     # High-resolution attention comparison heatmaps
│   ├── ablation/                         # Ablation loss curves and attention heatmaps
│   └── *.csv                             # Training loss logs (Phase 2, Phase 3, Bonus)
└── README.md                             # Reproduction steps & checkpoint links
```

---

## Model Checkpoints & Training Logs

All finetuned models, bonus ablation checkpoints, and logs are hosted on [Hugging Face Hub: manyajain111/LMA-Individual-Project](https://huggingface.co/manyajain111/LMA-Individual-Project):

### 1. Pretrained Checkpoints (Phase 2)
- **Hindi Pretrained Model H (`model_final.pt`)**: [`pretrained/hindi_model_final.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/pretrained/hindi_model_final.pt)
- **Nepali Pretrained Model L (`model_final.pt`)**: [`pretrained/nepali_model_final.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/pretrained/nepali_model_final.pt)
- **Training Logs**: `hindi/train/train_log.csv` and `nepali/train/train_log.csv` on the repo; also on HF at [`logs/hindi_pretrain_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/hindi_pretrain_log.csv) / [`logs/nepali_pretrain_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/nepali_pretrain_log.csv).

### 2. Finetuned Reasoning Checkpoints (Phase 3)
- **Hindi Finetuned Model**: [`finetuned/hindi_model_finetuned.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/finetuned/hindi_model_finetuned.pt)
- **Nepali Finetuned Model**: [`finetuned/nepali_model_finetuned.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/finetuned/nepali_model_finetuned.pt)
- **Training Logs**: [`logs/phase3_reasoning_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/phase3_reasoning_log.csv)

### 3. Bonus Ablation Checkpoints (No Positional Embeddings)
All 6 intermediate training checkpoints and the final model are hosted in the `ablation/` directory on Hugging Face:
- **Final Ablated Checkpoint**: [`ablation/model_ablated.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated.pt)
- **Step 10,000**: [`ablation/model_ablated_step_10000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_10000.pt)
- **Step 20,000**: [`ablation/model_ablated_step_20000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_20000.pt)
- **Step 30,000**: [`ablation/model_ablated_step_30000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_30000.pt)
- **Step 40,000**: [`ablation/model_ablated_step_40000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_40000.pt)
- **Step 50,000**: [`ablation/model_ablated_step_50000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_50000.pt)
- **Step 60,000**: [`ablation/model_ablated_step_60000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_60000.pt)
- **Ablation Training Log**: [`logs/bonus_ablation_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/bonus_ablation_log.csv)

---

## Step-by-Step Reproduction Guide

### 1. Environment Setup
Clone the repository and install the verified dependencies:
```bash
git clone https://github.com/manyajain111/individual-project-ManyaJain111.git
cd individual-project-ManyaJain111
pip install -r requirements.txt
```

### 2. Dataset Collection & Tokenizer Training (Phase 1)
To collect and preprocess the corpora and train the custom BPE tokenizers:
```bash
# Hindi
python hindi/data_collection/collect_hindi_wiki.py
python hindi/preprocessing/clean_corpus.py
python hindi/tokenizer_training/train_tokenizer.py

# Nepali
python nepali/data_collection/wikepedia_scraper.py
python nepali/preprocessing/clean_corpus.py
python nepali/tokenizer_training/train_tokenizer.py
```

### 3. Model Pretraining (Phase 2)
To run the pretraining loop locally or on GPU:
```bash
# Hindi Model H Pretraining
python hindi/train/train.py --config hindi/config/model_h_config.json

# Nepali Model L Pretraining
python nepali/train/train.py --config nepali/config/model_l_config.json
```

### 4. Evaluation & Diagnostics (Phase 2)
To compute Perplexity, Bits-per-Byte (BPB), BLEU, chrF, ROUGE-L, Distinct-N, and extract attention heatmaps:
```bash
python phase2/src/evaluate.py \
    --model_path checkpoints/phase3/checkpoints/pretrained/hindi/model_final.pt \
    --tokenizer_path phase2/tokenizers/hindi_tokenizer_16000.json \
    --test_data data/hindi_test.bin \
    --output_dir report/hindi/
```

### 5. Reasoning Dataset Generation & Finetuning (Phase 3)
1. **Generate Reasoning Dataset**:
   ```bash
   cd phase3
   python generate_reasoning_dataset.py
   cd ..
   ```
   This generates disjoint entity splits in `phase3/data/hi/` and `phase3/data/ne/` with zero leakage across train/val/test.

2. **Finetune the Models**:
   Open and execute `phase3/phase3_final.ipynb` (or `hindi/train/phase3_final.ipynb`) on Kaggle or a CUDA-enabled GPU.

### 6. Bonus: Positional Embedding Ablation
To train and evaluate the ablation model where positional embeddings are stripped:
1. Run `phase3/bonus_training_final.ipynb` to train the ablated model for 60,000 steps (checkpoints are saved every 10,000 steps).
2. Run `phase3/bonus_eval_final.ipynb` to compute ablation PPL, BPB, 4-gram repetition rate, and attention collapse heatmaps.

---

## Key Experimental Results Summary

### Pretraining Results (Phase 2)

| Model | Language | Vocab Size | Validation Loss | Perplexity (PPL) | Bits-per-Byte (BPB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model H** | Hindi | 16,000 | 3.6487 | **39.11** | 0.5131 |
| **Model L** | Nepali | 16,000 | 3.5210 | **33.82** | **0.3939** |

### Reasoning Finetuning Results (Phase 3)

| Metric | Model H (Hindi) | Model L (Nepali) |
| :--- | :--- | :--- |
| **Pretrained Baseline Accuracy** | 0.50% | 2.00% |
| **Finetuned Full Test Accuracy** | **26.60%** | **25.20%** |
| **Finetuned Easy Split Accuracy**| **50.00%** | — |

### Bonus Ablation Study (Impact of Positional Embeddings)

| Metric | Standard Hindi (With Positions) | Ablated Model (No Positions) | Relative Impact |
| :--- | :--- | :--- | :--- |
| **Perplexity (PPL)** | 39.11 | **58.54** | **+49.7% degradation** |
| **Validation Loss** | 3.6487 | **4.0697** | Higher cross-entropy |
| **4-gram Repetition** | 26.59% | **68.52%** | **Severe degeneration** |
| **Distinct-1** | 0.4270 | **0.1703** | Vocabulary collapse |
| **Attention Pattern**| Structured diagonal focus | Uniform / smearing | Complete loss of syntax |
