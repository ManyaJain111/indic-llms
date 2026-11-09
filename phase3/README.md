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
│   ├── phase3_reasoning_finetuning.ipynb # Resumable reasoning finetuning loop & evaluation
│   ├── checkpoints-saved.ipynb           # Resumable checkpoint finetuning notebook
│   ├── bonus_training_final.ipynb        # Positional embedding ablation training
│   └── bonus_eval_final.ipynb            # Ablation evaluation & metrics extraction
├── logs/                                 # Raw training and execution logs
│   ├── bonus.log                         # Bonus ablation 60k-step training log
│   ├── checkpoints-saved.log             # Phase 3 finetuning & evaluation run log
│   └── phase3_finetuning.log             # Alias for Phase 3 finetuning log
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
- **Epoch Checkpoints (Nepali)**: [Epoch 1](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/checkpoints/phase3/finetuned/nepali/model_finetuned_epoch_1.pt) · [Epoch 2](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/checkpoints/phase3/finetuned/nepali/model_finetuned_epoch_2.pt) · [Epoch 3](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/checkpoints/phase3/finetuned/nepali/model_finetuned_epoch_3.pt)
- **Finetuning Training Logs**: [`logs/phase3_reasoning_log.csv`](https://huggingface.co/manyajain111/LMA-Individual-Project/blob/main/logs/phase3_reasoning_log.csv) (console logs at [`logs/checkpoints-saved.log`](logs/checkpoints-saved.log))

> **Resumable Checkpoint Architecture**: In accordance with course guidelines, each intermediate epoch checkpoint is saved as a complete state dictionary (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `step`, `val_loss`, `train_loss`, `config`, `finetune_hyperparams`), allowing seamless mid-training resumption after disconnects or preemptions.

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

> **Branch Organization Note**: In accordance with submission requirements, this branch (`Phase-3`) contains the final consolidated Phase 3 and Bonus Ablation codebase, report, and metrics. Complete source pipelines for Phase 1 (scraping & tokenizers) and Phase 2 (pretraining & benchmark suites) are archived in their respective branches `Phase-1` and `Phase-2`.

### 2. Phase 1 & 2 Historical Pipelines (Archived on `Phase-1` and `Phase-2`)
The data collection, custom BPE training, and pretraining scripts reside on their dedicated phase branches:
```bash
# Checkout Phase 1 branch for crawlers & tokenizer trainers
git checkout Phase-1

# Checkout Phase 2 branch for pretraining loops & evaluation scripts
git checkout Phase-2
```

### 5. Reasoning Dataset Generation & Finetuning (Phase 3)
1. **Generate Reasoning Dataset**:
   ```bash
   python scripts/generate_reasoning_dataset.py
   python scripts/generate_easy_test.py
   ```
   This generates disjoint entity splits in `data/hi/` and `data/ne/` with zero leakage across train/val/test.

2. **Finetune the Models & Checkpoint Resumption**:
   Open and execute `notebooks/phase3_reasoning_finetuning.ipynb` (or `notebooks/checkpoints-saved.ipynb`) on Kaggle or a CUDA-enabled GPU. Checkpoints are automatically saved after each epoch with complete optimizer and scheduler state.

   To resume training from an intermediate epoch checkpoint:
   ```python
   # Load model, optimizer, scheduler, and epoch state
   ckpt = torch.load("checkpoints/finetuned/hindi/model_finetuned_epoch_1.pt", map_location=device)
   model.load_state_dict(ckpt["model_state_dict"])
   optimizer.load_state_dict(ckpt["optimizer_state_dict"])
   scheduler.load_state_dict(ckpt["scheduler_state_dict"])
   start_epoch = ckpt["epoch"] + 1  # Resumes smoothly from epoch 2
   ```

### 6. Bonus: Positional Embedding Ablation
To train and evaluate the ablation model where positional embeddings are stripped:
1. Run `notebooks/bonus_training_final.ipynb` to train the ablated model for 60,000 steps (checkpoints are saved every 10,000 steps with full optimizer resumability).
2. Run `notebooks/bonus_eval_final.ipynb` to compute ablation PPL, BPB, 4-gram repetition rate, and attention collapse heatmaps.

---

## Key Experimental Results Summary

### Pretraining Results (Phase 2)

| Model | Language | Vocab Size | Validation Loss | Perplexity (PPL) | Bits-per-Byte (BPB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model H** | Hindi | 16,000 | 3.6487 | **39.11** | 0.5131 |
| **Model L** | Nepali | 16,000 | 3.5210 | **33.82** | **0.3939** |

### Reasoning Finetuning Results (Phase 3)

### Reasoning Finetuning Results (Phase 3)

We report both **Strict Exact Match** (literal string equality $pred = target$) and **Answer Substring Match** ($target \in pred$):

| Metric | Model H (Hindi) | Model L (Nepali) | Evaluation Methodology / Notes |
| :--- | :--- | :--- | :--- |
| **Strict Exact Match (Pretrained)** | 0.00% | 0.00% | Generates unconstrained sequence |
| **Strict Exact Match (Finetuned)** | 0.00% | 0.00% | Model generates full sentence continuation |
| **Answer Substring Match (Pretrained)** | 1.04% | 1.93% | Pretrained baseline chance rate |
| **Answer Substring Match (Finetuned)** | **27.26%** | **26.67%** | Structured multi-hop reasoning |
| **Simplified Prompts (60-item test)** | **50.00%** | — | Direct comparative 2-choice questions |

> **Note on Exact Match vs. Substring Match**: As documented in `report/eval_results.json`, literal exact string match is 0.0% because unconstrained autoregressive decoders generate full sentence completions (e.g., `उत्तर: ओडिशा की आय अधिक है।`) rather than terminating immediately after the isolated entity name (`ओडिशा`). Under substring containment, finetuning lifts answer extraction from ~1% to ~27% overall.

### Bonus Ablation Study (Impact of Positional Embeddings)

| Metric | Standard Hindi (With Positions) | Ablated Model (No Positions) | Relative Impact & Analysis |
| :--- | :--- | :--- | :--- |
| **Perplexity (PPL)** | 39.11 | **58.54** | **+49.7% degradation** |
| **Validation Loss** | 3.6487 | **4.0697** | Higher cross-entropy loss |
| **4-gram Repetition** | 26.59% | **68.52%** | **Severe looping degeneration** |
| **Distinct-1** | 0.4270 | **0.1703** | Severe vocabulary collapse |
| **Layer 0 Attention** | Broad contextual diffusion | Sharp diagonal collapse | Model collapses to self-attention identity |
| **Avg Attention Distance**| 7.15 tokens (broad) | 2.18 tokens (ultra-local) | Cannot attend across long distances |
