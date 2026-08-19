# Language Models and Agents - Phase 1 Submission

This repository contains the Phase 1 implementation of two independent decoder-only Transformer language models (Model H for Hindi, Model L for Nepali).

## Project Links (Datasets & Archives)

Due to the massive scale of the corpora (approx. ~500M words / 14GB total), the datasets and checkpoints have been packaged and securely hosted on HuggingFace as required by the submission guidelines.

* **Model H (Hindi) Dataset Archive:** [https://huggingface.co/datasets/manyajain111/hindi-lma-corpus](https://huggingface.co/datasets/manyajain111/hindi-lma-corpus)
* **Model L (Nepali) Dataset Archive:** [https://huggingface.co/datasets/manyajain111/nepali-lma-corpus](https://huggingface.co/datasets/manyajain111/nepali-lma-corpus)


## Repository Structure

As per the rubric, the repository is strictly divided by language:
* `hindi/` - Contains all data collection, tokenization, and pipeline code for Model H.
* `nepali/` - Contains all data collection, tokenization, and pipeline code for Model L.
* `report/` - Contains the master Phase 1 Dataset and Tokenizer statistics report.
## Reproduction Steps

1. **Setup Environment**: Install dependencies from `requirements.txt`.
2. **Dataset**: Run `hindi/preprocessing/prepare_dataset.py` and `nepali/preprocessing/prepare_dataset.py` to recreate splits. Alternatively, download the cleaned data from HuggingFace.
3. **Tokenizer Training**: Run `hindi/training/train_tokenizer.py` and `nepali/training/train_tokenizer.py`.

The final trained tokenizer files and vocabularies are available directly in the repository under `hindi/models/` and `nepali/models/`. The final statistical evaluations are under `hindi/reports/` and `nepali/reports/`.
