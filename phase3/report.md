# Phase 3: Reasoning Finetuning, Attention Analysis, and Final Report

## 1. Synthetic Reasoning Dataset
We generated a custom synthetic reasoning dataset to finetune our models, strictly isolating the Hindi and Nepali data to avoid cross-lingual contamination. The dataset focuses on transitive and ordering comparisons (e.g., "If A is taller than B, and B is taller than C, who is tallest?"). 
- **Leakage Prevention**: We explicitly partitioned the entities (names, subjects) into disjoint sets: 70% Train, 15% Validation, 15% Test. This ensures that the exact names/relationships seen in training are never seen in testing.

## 2. Finetuning & Reasoning Accuracy Results
Both models were independently finetuned on their respective reasoning datasets.

| Metric | Model H (Hindi) | Model L (Nepali) |
| :--- | :--- | :--- |
| **Pretrained Baseline Accuracy** | 0.50% | 2.00% |
| **Finetuned Overall Accuracy** | 26.60% | 25.20% |
| **Finetuned Easy Prompt Accuracy**| 50.00% | - |

**Analysis**: Finetuning on structured data dramatically improved reasoning performance, taking the Hindi model from near 0% to 26.6% on the full mixed suite, and hitting 50.00% exact match accuracy on the purely "easy" prompt split. Nepali showed an almost identical absolute gain, proving that the underlying logic circuits can form regardless of the specific language embedding.

## 3. Attention Analysis (Pretrained vs. Post-trained)
We compared the Layer 5 (late layer) attention maps for a reasoning prompt ("यदि राम की आयु 25 साल है..."):
- **Pretrained**: Attention is relatively uniform and syntactically driven (attending closely to immediate neighbors).
- **Finetuned**: Attention heatmaps reveal significant head specialization. Certain attention heads "learn" to look specifically at the comparison entities (e.g., "राम" and "श्याम") and the comparative operative words (e.g., "कम"). The attention becomes sparser and much more long-range, skipping over boilerplate grammar words to directly link the subject to its relation. 

![Pretrained vs Finetuned Attention](../report/phase3_plots/plot_3.png)

---

## 4. Final Scientific Comparison (Addressing Key Questions)

**1. How did data scale and quality differ between Model H and Model L?**
Model H (Hindi) benefited from a massive and diverse corpus (OSCAR, Wikipedia, News), requiring heavy subsampling to fit the bounds. Model L (Nepali) was severely constrained, requiring us to scrape diverse but noisy secondary sources (news sites, raw dumps) to assemble a viable corpus. 

**2. How do language-modeling and reasoning results compare across the two resource tiers?**
Despite the gap in raw data volume, Nepali achieved a comparable (and actually slightly better) BPB (0.3939 vs 0.5131) due to structural efficiencies in its tokenizer. Interestingly, reasoning results were almost identical (26.60% vs 25.20%). The structural logic of the transformer scaled perfectly to the low-resource language once the synthetic finetuning task was mapped correctly.

**3. What tokenizer / corpus factors most affected the lower-resource model?**
Because the Nepali corpus was smaller, our custom BPE tokenizer required a slightly higher Byte-Per-Token compression to hit the target vocab size, resulting in longer average sequence lengths (13.16 bytes/token vs 10.31 bytes/token for Hindi). This means the Nepali model has to look back further in the context window to glean the same amount of semantic meaning.

**4. What evidence explains the observed differences?**
The repetition rate metrics in Phase 2 (Nepali Greedy Repetition = 67.12% vs Hindi 26.59%) highlight how the lower data diversity in Nepali led to faster degeneration during unbounded greedy generation. However, because both models share the exact same parameter count and hidden dimensions, they performed equally well on structured classification/reasoning (Phase 3), where generation length is constrained and logical structure dominates.

---

# Bonus: Ablation Study (No Positional Embeddings)

To isolate the impact of position information, we retrained the Hindi model with absolute positional embeddings entirely removed from the architecture.

## Ablation Results

| Metric | Standard Hindi | Ablated Hindi (No Positions) |
| :--- | :--- | :--- |
| **Validation Loss** | 3.6487 | 4.0697 |
| **Perplexity (PPL)** | 39.11 | **58.54** |
| **Bits-Per-Byte (BPB)**| 0.5131 | 0.5674 |
| **4-gram Repetition** | 26.59% | **68.52%** |
| **Distinct-1** | 0.4270 | 0.1703 |

## What breaks without position information?
1. **Severe Perplexity Degradation**: The model completely loses its inductive bias for sequence order. Sentences become purely a "bag of words" to the attention heads, driving Perplexity up from 39.11 to 58.54.
2. **Infinite Looping (Degeneration)**: Without position information, the model has no concept of "progress" through a sentence. As seen in the metrics, `Distinct-1` vocabulary usage drops to just 0.17, and the 4-gram repetition rate spikes to a massive **68.52%**. It gets stuck outputting the exact same high-probability token loops endlessly.
3. **Attention Collapse**: In the generated ablation attention heatmaps:

![Bonus Attention Collapse](../report/ablation/attn_entropy_distance.png)

the typical diagonal "local attention" lines completely vanish. Because every token lacks a position identifier, the attention heads smear their weights uniformly across the sequence or collapse entirely into static patterns, destroying the model's ability to track syntax.
