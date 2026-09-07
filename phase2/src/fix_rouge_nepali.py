"""
fix_rouge_nepali.py
-------------------
Re-computes ROUGE-L for Nepali generation samples using whitespace tokenisation.
The default rouge_score tokenizer doesn't split Devanagari correctly, producing
ROUGE-L = 0.0. This script uses a pure-Python LCS-based ROUGE-L (no external deps)
with whitespace splitting. Reads from reports/nepali/generation_samples.json and
patches reports/nepali/generation_quality.json in-place.
"""

import json
import os


def lcs_length(a, b):
    """Compute LCS length between two token lists using DP."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l_fmeasure(hypothesis: str, reference: str, beta: float = 1.2) -> float:
    """Compute ROUGE-L F-measure (whitespace-tokenised, no stemming)."""
    hyp_tokens = hypothesis.strip().split()
    ref_tokens = reference.strip().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    f = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    return f


def compute_rougeL(hyps, refs):
    pairs = [(h, r) for h, r in zip(hyps, refs) if h.strip() and r.strip()]
    if not pairs:
        return 0.0, 0
    scores = [rouge_l_fmeasure(h, r) for h, r in pairs]
    return sum(scores) / len(scores), len(pairs)


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'reports', 'nepali')
    samples_path = os.path.join(base, 'generation_samples.json')
    quality_path = os.path.join(base, 'generation_quality.json')

    with open(samples_path, encoding='utf-8') as f:
        samples = json.load(f)

    refs = [s.get('reference', '') for s in samples]
    keys = ['greedy', 'temp_0.5', 'temp_1.0', 'temp_1.5']


    print("Re-computing ROUGE-L with whitespace tokeniser (pure Python LCS)...\n")
    
    # Load existing quality file to preserve BLEU/chrF
    with open(quality_path, encoding='utf-8') as f:
        gen_quality = json.load(f)

    for key in keys:
        hyps = [s[key] for s in samples]
        rougeL, n = compute_rougeL(hyps, refs)
        gen_quality[key]['rougeL'] = round(rougeL, 6)
        gen_quality[key]['n_scored'] = n
        print(f"  {key}: ROUGE-L = {rougeL:.4f}  (n={n})")

    with open(quality_path, 'w', encoding='utf-8') as f:
        json.dump(gen_quality, f, indent=2, ensure_ascii=False)

    print(f"\nPatched {quality_path}")


if __name__ == '__main__':
    main()
