# Language Models and Agents - Phase 1 Submission

This repository contains the Phase 1 implementation of two independent decoder-only Transformer language models (Model H for Hindi, Model L for Nepali).

## Project Links (Datasets & Archives)

Due to the massive scale of the corpora (approx. ~500M words / 14GB total), the datasets and checkpoints have been packaged and securely hosted on HuggingFace as required by the submission guidelines.

* **Model H (Hindi) Dataset Archive:** [https://huggingface.co/datasets/manyajain111/hindi-lma-corpus](https://huggingface.co/datasets/manyajain111/hindi-lma-corpus)
* **Model L (Nepali) Dataset Archive:** [https://huggingface.co/datasets/manyajain111/nepali-lma-corpus](https://huggingface.co/datasets/manyajain111/nepali-lma-corpus)

*(Note: These datasets are marked as Private per the project requirements. Please ensure you have requested access or are logged into the appropriate HuggingFace account to view them).*

## Repository Structure

As per the rubric, the repository is strictly divided by language:
* `hindi/` - Contains all data collection, tokenization, and pipeline code for Model H.
* `nepali/` - Contains all data collection, tokenization, and pipeline code for Model L.
* `report/` - Contains the master Phase 1 Dataset and Tokenizer statistics report.
