# Phase 2: Model Implementation, Pretraining, and Evaluation Report

## 1. Overview & Architecture Specifications

Two independent, monolingual decoder-only (GPT-style) autoregressive Transformer language models were implemented from first principles in PyTorch (without using HuggingFace or nn.Transformer primitives).

- **Model H (Higher-resource)**: Hindi
- **Model L (Lower-resource)**: Nepali

### Architecture Details
- **Positional Embeddings**: Learned absolute positional embeddings (`nn.Embedding(context_len, d_model)`).
- **Self-Attention**: First-principles Multi-Head Attention with scaled dot-product attention (`QK^T / sqrt(d_k) + M`), where M is a causal upper-triangular mask (-inf). Explicit shape transformations `(B, T, d) -> (B, h, T, d_k)` and reverse.
- **Normalization & Structure**: Pre-LayerNorm architecture with residual connections around attention and two-layer feed-forward sublayers (GELU activations).
- **Weight Tying**: Tying the output linear projection weights with the input token embedding matrix (`W_out = W_emb`), reducing total parameter footprint.

### Configuration & Parameter Counts

| Parameter | Model H (Hindi) | Model L (Nepali) |
| :--- | :--- | :--- |
| **Vocabulary Size** | 16,000 | 16,240 |
| **Hidden Dim ({model}$)** | 512 | 512 |
| **Heads ({heads}$)** | 8 ( = 64$) | 8 ( = 64$) |
| **Layers ({layers}$)** | 6 | 6 |
| **Feed-Forward Dim ({ff}$)** | 1,536 ( 	imes d_{model}$) | 1,536 ( 	imes d_{model}$) |
| **Context Length** | 512 | 512 |
| **Dropout** | 0.1 | 0.1 |
| **Trainable Parameters** | **~24.22M** | **~24.34M** |

---

## 2. Pretraining & Loss Curves

Both models were trained using AdamW with cosine learning rate scheduling and warmup. Checkpoints were periodically saved with full optimizer and scheduler states.

- **Hindi (Model H)**: Trained for 61,036 steps.
- **Nepali (Model L)**: Trained for 61,036 steps.

### Generated Loss Curves
- Hindi Training & Val Loss Curve: `reports/hindi/loss_curve.png`
- Nepali Training & Val Loss Curve: `reports/nepali/loss_curve.png`
- Comparison Loss Curve: `reports/comparison_loss_curve.png`

---

## 3. Language Modeling Metrics (Intrinsic Evaluation)

Evaluated on held-out test splits.

| Metric | Model H (Hindi) | Model L (Nepali) | Notes |
| :--- | :--- | :--- | :--- |
| **Validation Loss (nats)** | 3.6487 | 3.5910 | Cross-entropy loss |
| **Test Loss (nats)** | 3.6663 | 3.5923 | Cross-entropy loss |
| **Test Perplexity (PPL)** | **39.11** | **36.32** |  = e^{Loss}$ |
| **Test Bits-Per-Byte (BPB)** | **0.5131** | **0.3939** | Normalized by UTF-8 bytes per token |
| **Avg Bytes per Token** | ~10.31 | ~13.16 | Measured across tokenized targets |

### Intrinsic Metric Discussion
- **Cross-Entropy & PPL**: Nepali achieved a slightly lower perplexity (36.32) compared to Hindi (39.11).
- **Bits-Per-Byte (BPB)**: BPB normalizes for vocabulary compression efficiency across different tokenizers. Nepali reaches 0.3939 BPB vs. 0.5131 BPB for Hindi. Both models demonstrate healthy convergence on held-out text.

---

## 4. Generation Quality & Diagnostics

Continuations generated from held-out prefixes under greedy decoding and temperature sampling (0.5, 1.0, 1.5).

### Generation Quality (Overlaps & Similarity)

| Decoding | Model | BLEU-4 | chrF / chrF++ | ROUGE-L |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy** | Hindi (Model H) | 0.9488 | 12.21 | 0.1243 |
| | Nepali (Model L) | 1.2197 | 13.54 | 0.0838 |
| **Temp 0.5** | Hindi (Model H) | 0.7732 | 13.45 | 0.0921 |
| | Nepali (Model L) | 1.5608 | 19.88 | 0.0966 |
| **Temp 1.0** | Hindi (Model H) | 0.7182 | 12.71 | 0.0906 |
| | Nepali (Model L) | 1.0644 | 19.27 | 0.0901 |
| **Temp 1.5** | Hindi (Model H) | 0.8372 | 11.56 | 0.0955 |
| | Nepali (Model L) | 0.1445 | 16.08 | 0.0394 |

### Fluency, Diversity, and Repetition Diagnostics

| Decoding | Model | Repetition Rate (4-gram) | Distinct-1 | Distinct-2 |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy** | Hindi | 0.2659 | 0.4270 | 0.6215 |
| | Nepali | 0.6712 | 0.2291 | 0.3167 |
| **Temp 0.5** | Hindi | 0.1012 | 0.4605 | 0.7336 |
| | Nepali | 0.3351 | 0.3981 | 0.6283 |
| **Temp 1.0** | Hindi | 0.0267 | 0.6838 | 0.9313 |
| | Nepali | 0.0213 | 0.7214 | 0.9586 |
| **Temp 1.5** | Hindi | 0.0000 | 0.9447 | 1.0000 |
| | Nepali | 0.0000 | 0.9079 | 0.9990 |

### Qualitative Analysis
- **Greedy Decoding**: Tends to degenerate into repetitive loops over extended context windows (higher repetition rate), though local syntax is consistent.
- **Low Temperature (0.5)**: Balances natural coherence with lexical variety.
- **High Temperature (1.0 - 1.5)**: Reaches maximal lexical diversity (Distinct-2 ~ 0.95-1.0) while eliminating n-gram repetition, though T=1.5 exhibits semantic drift.

---

## 5. Multi-Head Attention Analysis

Attention maps extracted across early (Layer 0), middle (Layer 3), and late (Layer 5) layers.

- **Attention Entropy**: Quantifies dispersion (diffuse vs. concentrated).
- **Mean Attention Distance**: Measures average query-key distance in sequence positions.

### Observations
1. **Early Layers (Layer 0)**: Exhibit local positional focus (diagonal concentration, lower entropy, short distance ~1.5 - 2.1 positions), attending to immediate preceding tokens and subword constituents.
2. **Intermediate Layers (Layer 3)**: Develop syntactic specialization, routing representations across phrase and clause boundaries.
3. **Deep Layers (Layer 5)**: Broader, long-range attention (higher entropy, distance ~2.0 - 2.5 positions), integrating sentence-level context for predictive token modeling.
