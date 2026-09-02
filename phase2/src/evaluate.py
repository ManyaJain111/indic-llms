import os
import json
import math
import argparse
import collections
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sacrebleu
from rouge_score import rouge_scorer
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Kaggle/headless
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from model import GPTLanguageModel
from tokenizer_wrapper import TokenizerWrapper
from dataset import get_dataloader

def evaluate_ppl_bpb(model, val_loader, device, tokenizer, num_batches=200):
    """Compute cross-entropy loss, perplexity, and bits-per-byte.
    
    BPB = CE_nats / (log(2) * avg_bytes_per_token)
    where avg_bytes_per_token is measured by decoding actual token bytes.
    This makes BPB comparable across tokenizers with different fertility.
    """
    model.eval()
    total_loss_nats = 0.0
    total_tokens = 0
    total_bytes = 0

    val_iter = iter(val_loader)
    with torch.no_grad():
        for i in range(num_batches):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break

            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            n_toks = y.numel()
            total_loss_nats += loss.item() * n_toks
            total_tokens += n_toks

            # Count UTF-8 bytes for every target token to get true BPB
            flat_ids = y.view(-1).cpu().tolist()
            decoded_text = tokenizer.decode(flat_ids)
            total_bytes += len(decoded_text.encode('utf-8'))

    avg_loss = total_loss_nats / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(min(avg_loss, 20))  # clamp to avoid overflow pre-convergence
    # True BPB: nats -> bits, then normalise by bytes per token
    avg_bytes_per_token = total_bytes / total_tokens if total_tokens > 0 else 1.0
    bpb = (avg_loss / math.log(2)) / avg_bytes_per_token

    return avg_loss, ppl, bpb, avg_bytes_per_token

def compute_repetition_rate(text, n=4):
    """Fraction of n-grams that are repeated (lower = more diverse)."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    repeated = len(ngrams) - len(set(ngrams))
    return repeated / len(ngrams) if ngrams else 0.0


def evaluate_generation(model, tokenizer, device, test_prefixes, references):
    """Generate with greedy decoding and temperatures 0.5 / 1.0 / 1.5.
    Compute BLEU-4, chrF, ROUGE-L, Distinct-1/2, and repetition rate for each.
    Spec requires greedy decoding explicitly alongside the three temperatures.
    """
    model.eval()
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    results = {}

    # --- Greedy decoding first (deterministic argmax, no temperature) ---
    greedy_generations = []
    for prefix in test_prefixes:
        enc = tokenizer.encode(prefix)
        if not enc:
            enc = [1]
        x = torch.tensor([enc], dtype=torch.long).to(device)
        with torch.no_grad():
            gen_idx = model.greedy_generate(x, max_new_tokens=50)
        greedy_generations.append(tokenizer.decode(gen_idx[0].cpu().tolist()))

    bleu_g = sacrebleu.corpus_bleu(greedy_generations, [references]).score
    chrf_g = sacrebleu.corpus_chrf(greedy_generations, [references]).score
    rougeL_g = sum(rouge.score(r, g)['rougeL'].fmeasure
                   for r, g in zip(references, greedy_generations)) / max(len(greedy_generations), 1)
    all_w_g = [w for g in greedy_generations for w in g.split()]
    dist1_g = len(set(all_w_g)) / len(all_w_g) if all_w_g else 0.0
    bg = list(zip(all_w_g, all_w_g[1:]))
    dist2_g = len(set(bg)) / len(bg) if bg else 0.0
    rep_g = sum(compute_repetition_rate(g, 4) for g in greedy_generations) / max(len(greedy_generations), 1)
    results['greedy'] = {
        'BLEU-4': round(bleu_g, 4), 'chrF': round(chrf_g, 4), 'ROUGE-L': round(rougeL_g, 4),
        'Distinct-1': round(dist1_g, 4), 'Distinct-2': round(dist2_g, 4),
        'repetition_rate_4gram': round(rep_g, 4), 'generations': greedy_generations,
    }

    # --- Temperature sampling ---
    for temp in [0.5, 1.0, 1.5]:
        generations = []
        for prefix in test_prefixes:
            enc = tokenizer.encode(prefix)
            if not enc:
                enc = [1]  # fallback to a safe token id
            x = torch.tensor([enc], dtype=torch.long).to(device)
            with torch.no_grad():
                gen_idx = model.generate(x, max_new_tokens=50, temperature=temp)
            gen_text = tokenizer.decode(gen_idx[0].cpu().tolist())
            generations.append(gen_text)

        # sacrebleu expects List[str] hypotheses and List[List[str]] references
        bleu = sacrebleu.corpus_bleu(generations, [references]).score
        chrf = sacrebleu.corpus_chrf(generations, [references]).score

        rougeL_scores = [rouge.score(ref, gen)['rougeL'].fmeasure
                         for ref, gen in zip(references, generations)]
        rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0

        # Diversity: Distinct-1, Distinct-2
        all_words = [w for gen in generations for w in gen.split()]
        total_w = len(all_words)
        dist1 = len(set(all_words)) / total_w if total_w > 0 else 0.0
        bigrams_all = list(zip(all_words, all_words[1:]))
        dist2 = len(set(bigrams_all)) / len(bigrams_all) if bigrams_all else 0.0

        # Repetition rate (4-gram)
        rep_rates = [compute_repetition_rate(gen, n=4) for gen in generations]
        rep_rate = sum(rep_rates) / len(rep_rates) if rep_rates else 0.0

        results[float(temp)] = {
            'BLEU-4': round(bleu, 4),
            'chrF': round(chrf, 4),
            'ROUGE-L': round(rougeL, 4),
            'Distinct-1': round(dist1, 4),
            'Distinct-2': round(dist2, 4),
            'repetition_rate_4gram': round(rep_rate, 4),
            'generations': generations,
        }
    return results

def compute_attention_stats(attention_maps):
    """Compute per-head attention entropy and mean attention distance.
    
    Returns:
        entropy[layer][head]: float
        distance[layer][head]: float  (mean position distance attended to)
    """
    entropy_table = []
    distance_table = []
    for layer_idx, attn in enumerate(attention_maps):
        # attn: (B, n_heads, T, T)
        attn_np = attn[0].detach().cpu().numpy()  # (n_heads, T, T)
        n_heads, T, _ = attn_np.shape
        ent_row, dist_row = [], []
        for h in range(n_heads):
            weights = attn_np[h]  # (T, T)
            # Entropy: H = -sum(w * log(w+eps)) averaged over queries
            eps = 1e-9
            ent = -np.sum(weights * np.log(weights + eps), axis=-1).mean()
            ent_row.append(float(ent))
            # Mean attention distance: for each query position q, sum_k |q-k| * w[q,k]
            positions = np.arange(T)
            dists = []
            for q in range(T):
                d = np.sum(np.abs(positions - q) * weights[q])
                dists.append(d)
            dist_row.append(float(np.mean(dists)))
        entropy_table.append(ent_row)
        distance_table.append(dist_row)
    return entropy_table, distance_table


def plot_attention(model, tokenizer, device, text, save_dir):
    """Plot attention heatmaps for early and late layers, all heads.
    Also saves attention entropy and distance summary."""
    model.eval()
    enc = tokenizer.encode(text)
    if not enc:
        print('[plot_attention] Could not encode text, skipping.')
        return
    x = torch.tensor([enc], dtype=torch.long).to(device)
    with torch.no_grad():
        _, _, attention_maps = model(x, return_attention=True)

    tokens = [tokenizer.decode([t]) for t in x[0].tolist()]
    n_layers = len(attention_maps)
    n_heads = attention_maps[0].shape[1]

    # --- Heatmaps: early layer, middle layer, late layer ---
    layer_indices = sorted(set([0, n_layers // 2, n_layers - 1]))
    for layer_idx in layer_indices:
        attn = attention_maps[layer_idx][0].detach().cpu().numpy()  # (n_heads, T, T)
        cols = min(n_heads, 4)
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
        if cols == 1:
            axes = [axes]
        for h in range(cols):
            ax = axes[h]
            sns.heatmap(attn[h], ax=ax, cmap='viridis',
                        xticklabels=tokens, yticklabels=tokens,
                        vmin=0, vmax=attn[h].max())
            ax.set_title(f'Layer {layer_idx} Head {h}')
            ax.set_xlabel('Key position')
            ax.set_ylabel('Query position')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'attn_layer{layer_idx}.png'), dpi=150)
        plt.close()

    # --- Entropy and distance tables ---
    entropy_table, distance_table = compute_attention_stats(attention_maps)

    # Plot entropy heatmap (layers x heads)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.heatmap(np.array(entropy_table), ax=axes[0], cmap='magma',
                xticklabels=[f'H{h}' for h in range(n_heads)],
                yticklabels=[f'L{l}' for l in range(n_layers)],
                annot=True, fmt='.2f')
    axes[0].set_title('Attention Entropy (nats)')
    axes[0].set_xlabel('Head'); axes[0].set_ylabel('Layer')

    sns.heatmap(np.array(distance_table), ax=axes[1], cmap='viridis',
                xticklabels=[f'H{h}' for h in range(n_heads)],
                yticklabels=[f'L{l}' for l in range(n_layers)],
                annot=True, fmt='.1f')
    axes[1].set_title('Mean Attention Distance (tokens)')
    axes[1].set_xlabel('Head'); axes[1].set_ylabel('Layer')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attn_entropy_distance.png'), dpi=150)
    plt.close()

    return {
        'entropy_per_layer_head': entropy_table,
        'distance_per_layer_head': distance_table,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='reports')
    parser.add_argument('--language', type=str, default='hindi',
                        choices=['hindi', 'nepali'],
                        help='Which language model to evaluate (selects test prefixes)')
    parser.add_argument('--num_batches', type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    tokenizer = TokenizerWrapper(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']

    model = GPTLanguageModel(**config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

    # --- PPL / BPB ---
    val_loader = get_dataloader(args.dataset, tokenizer, config['context_len'],
                                split='test', batch_size=4)
    loss, ppl, bpb, avg_bytes_per_tok = evaluate_ppl_bpb(
        model, val_loader, device, tokenizer, num_batches=args.num_batches)
    print(f'\nTest Loss: {loss:.4f} nats | PPL: {ppl:.2f} | BPB: {bpb:.4f} | '
          f'avg bytes/tok: {avg_bytes_per_tok:.2f}')

    # --- Language-specific test prefixes ---
    if args.language == 'hindi':
        prefixes = ['मैं आज', 'भारत के', 'वह एक', 'सरकार ने', 'देश में']
        references = ['मैं आज बाजार गया था।', 'भारत के प्रधानमंत्री ने कहा',
                      'वह एक बहुत ही अच्छा इंसान है।',
                      'सरकार ने नई नीति की घोषणा की।',
                      'देश में लोकतंत्र की स्थापना हुई।']
        attn_text = 'मैं आज बाजार गया था।'
    else:  # nepali
        prefixes = ['म आज', 'नेपालको', 'उनले भने', 'सरकारले', 'देशमा']
        references = ['म आज बजार गएको थिएँ।', 'नेपालको प्रधानमन्त्रीले भने',
                      'उनले भने कि यो राम्रो छ।',
                      'सरकारले नयाँ नीति घोषणा गर्यो।',
                      'देशमा प्रजातन्त्रको स्थापना भयो।']
        attn_text = 'म आज बजार गएको थिएँ।'

    # --- Generation metrics ---
    gen_results = evaluate_generation(model, tokenizer, device, prefixes, references)
    print('\nGeneration Metrics:')
    for temp, res in gen_results.items():
        print(f'  Temp {temp}: BLEU={res["BLEU-4"]:.2f} chrF={res["chrF"]:.2f} '
              f'ROUGE-L={res["ROUGE-L"]:.4f} Dist-1={res["Distinct-1"]:.4f} '
              f'Dist-2={res["Distinct-2"]:.4f} rep={res["repetition_rate_4gram"]:.4f}')
        print(f'    Sample generation: {res["generations"][0][:120]}')

    # --- Attention analysis ---
    attn_stats = plot_attention(model, tokenizer, device, attn_text, args.output_dir)

    # --- Save all results ---
    serialisable_gen = {}
    for temp, res in gen_results.items():
        r = dict(res)
        r.pop('generations', None)  # strip raw text from main JSON, save separately
        serialisable_gen[str(temp)] = r

    # Save generated samples separately
    with open(os.path.join(args.output_dir, 'generated_samples.json'), 'w', encoding='utf-8') as f:
        json.dump({str(t): res['generations'] for t, res in gen_results.items()},
                  f, ensure_ascii=False, indent=2)

    results = {
        'language': args.language,
        'checkpoint': args.checkpoint,
        'loss': loss,
        'ppl': ppl,
        'bpb': bpb,
        'avg_bytes_per_token': avg_bytes_per_tok,
        'generation_metrics': serialisable_gen,
        'attention_stats': attn_stats,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
