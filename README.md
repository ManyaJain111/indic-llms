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

### Repository Structure

```
├── data/                                 # Reasoning datasets
│   ├── hi/                               # Hindi train/val/test splits & stats
│   ├── ne/                               # Nepali train/val/test splits & stats
│   └── simple/                           # Simplified evaluation prompts
├── notebooks/                            # All project Jupyter notebooks
│   ├── phase3_reasoning_finetuning.ipynb # Reasoning finetuning loop & evaluation
│   ├── bonus_training_final.ipynb        # Positional embedding ablation training
│   └── bonus_eval_final.ipynb            # Ablation evaluation & metrics extraction
├── scripts/                              # Core Python scripts
│   ├── generate_reasoning_dataset.py     # Programmatic grammar-based reasoning generator
│   ├── generate_easy_test.py             # 60-item simplified benchmark generator
│   ├── vocab_hi.py & vocab_ne.py         # Hindi and Nepali lexicon modules
│   └── generate_all_plots.py             # Multi-head attention & entropy plotting toolkit
├── report/                               # Reports, metrics, logs, and plots
│   ├── individual_project_report.pdf     # Consolidated final PDF submission report
│   ├── individual_project_report.tex     # Master LaTeX report source
│   ├── phase3_report.md                  # Detailed Phase 3 report
│   ├── bonus_report.md                   # Detailed Bonus Ablation report
│   ├── eval_results.json                 # Comprehensive reasoning evaluation metrics
│   ├── finetune_log_combined.csv         # Finetuning loss logs
│   ├── bonus_train_log.csv               # Ablation training loss logs
│   └── plots/                            # All plots clearly named
│       ├── phase2_hindi_loss_curve.png
│       ├── phase2_nepali_loss_curve.png
│       ├── phase2_comparison_loss_curve.png
│       ├── phase2_hindi_attn_layer*.png
│       ├── phase2_nepali_attn_layer*.png
│       ├── phase3_finetuning_loss_curves.png
│       ├── phase3_hindi_attn_compare_layer*.png
│       ├── phase3_nepali_attn_compare_layer*.png
│       ├── bonus_ablation_loss_curve*.png
│       ├── bonus_ablation_attn_layer*.png
│       └── bonus_ablation_generation_diversity.png
```

---

## Model Checkpoints & Training Logs

All finetuned models, bonus ablation checkpoints, and logs are hosted on [Hugging Face Hub: manyajain111/LMA-Individual-Project](https://huggingface.co/manyajain111/LMA-Individual-Project):

### 1. Pretrained Checkpoints (Phase 2)
Hosted on [Hugging Face Hub: manyajain111/lma-phase2](https://huggingface.co/manyajain111/lma-phase2):
- **Hindi Pretrained Model H (`model_final.pt`)**: [`hindi/model_final.pt`](https://huggingface.co/manyajain111/lma-phase2/blob/main/hindi/model_final.pt)
- **Nepali Pretrained Model L (`model_final.pt`)**: [`nepali/model_final.pt`](https://huggingface.co/manyajain111/lma-phase2/blob/main/nepali/model_final.pt)
- **Pretraining Logs**: `hindi/train/train_log.csv` and `nepali/train/train_log.csv` on the repo; also on HF at [`logs/hindi_pretrain_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/hindi_pretrain_log.csv) / [`logs/nepali_pretrain_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/nepali_pretrain_log.csv).

### 2. Finetuned Reasoning Checkpoints (Phase 3)
Hosted on [Hugging Face Hub: manyajain111/LMA-Individual-Project](https://huggingface.co/manyajain111/LMA-Individual-Project):
- **Hindi Finetuned Model**: [`finetuned/hindi_model_finetuned.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/finetuned/hindi_model_finetuned.pt) (also in [`checkpoints/phase3/finetuned/hindi/`](https://huggingface.co/manyajain111/LMA-Individual-Project/tree/main/checkpoints/phase3/finetuned/hindi))
- **Nepali Finetuned Model**: [`finetuned/nepali_model_finetuned.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/finetuned/nepali_model_finetuned.pt) (also in [`checkpoints/phase3/finetuned/nepali/`](https://huggingface.co/manyajain111/LMA-Individual-Project/tree/main/checkpoints/phase3/finetuned/nepali))
- **Epoch Checkpoints (Hindi)**: [Epoch 1](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/checkpoints/phase3/finetuned/hindi/model_finetuned_epoch_1.pt) · [Epoch 2](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/checkpoints/phase3/finetuned/hindi/model_finetuned_epoch_2.pt) · [Epoch 3](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/checkpoints/phase3/finetuned/hindi/model_finetuned_epoch_3.pt)
- **Finetuning Training Logs**: [`logs/phase3_reasoning_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/phase3_reasoning_log.csv)

### 3. Bonus Ablation Checkpoints (No Positional Embeddings)
All intermediate training checkpoints, final ablated weights, metrics, and visualization plots are hosted in the [`ablation/` directory on Hugging Face](https://huggingface.co/manyajain111/LMA-Individual-Project/tree/main/ablation):
- **Final Ablated Checkpoint**: [`ablation/model_ablated.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated.pt)
- **Step 10,000**: [`ablation/model_ablated_step_10000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_10000.pt)
- **Step 20,000**: [`ablation/model_ablated_step_20000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_20000.pt)
- **Step 30,000**: [`ablation/model_ablated_step_30000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_30000.pt)
- **Step 40,000**: [`ablation/model_ablated_step_40000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_40000.pt)
- **Step 50,000**: [`ablation/model_ablated_step_50000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_50000.pt)
- **Step 60,000**: [`ablation/model_ablated_step_60000.pt`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/model_ablated_step_60000.pt)
- **Ablation Training Log**: [`ablation/bonus_ablation_train_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/bonus_ablation_train_log.csv) (also at [`logs/bonus_ablation_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/bonus_ablation_log.csv))
- **Evaluation Results & Metrics**: [`ablation/results.json`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/results.json)
- **Diagnostic Plots**:
  - [Loss Curve (`bonus_loss_curve.png`)](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/plots/bonus_loss_curve.png)
  - [Attention Diagnostics (`attn_entropy_distance.png`)](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/plots/attn_entropy_distance.png)
  - [Generation & Diversity Metrics (`generation_diversity_metrics.png`)](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/ablation/plots/generation_diversity_metrics.png)

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
   Open and execute `phase3/notebooks/phase3_reasoning_finetuning.ipynb` (or `hindi/train/phase3_reasoning_finetuning.ipynb`) on Kaggle or a CUDA-enabled GPU.

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
