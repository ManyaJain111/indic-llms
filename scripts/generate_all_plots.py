import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from fix import load_sd
from phase3_lib_fixed import GPTLanguageModel, TokenizerWrapper, TransformerBlock

# Configure font
font_path = "NotoSansDevanagari-Regular.ttf"
devanagari_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']

class GPTLanguageModelAblated(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=1536, context_len=512, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, context_len, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
    def forward(self, idx, targets=None, return_attention=False):
        x = self.drop(self.token_emb(idx))
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            if return_attention: attention_maps.append(attn_weights)
        x = self.ln_f(x)
        logits = self.head(x)
        if return_attention: return logits, None, attention_maps
        return logits, None

def plot_attention_comparison_multihead(tok, pre_model, ft_model, text, lang, layer_idx=-1, is_early=False):
    tokens = tok.encode(text)
    input_ids = torch.tensor([tokens])
    pre_model.eval()
    ft_model.eval()
    with torch.no_grad():
        _, _, pre_attn = pre_model(input_ids, return_attention=True)
        _, _, ft_attn = ft_model(input_ids, return_attention=True)
    
    pre_attn = pre_attn[layer_idx][0].cpu().numpy()
    ft_attn = ft_attn[layer_idx][0].cpu().numpy()
    
    tokens_str = [tok.decode([t]) for t in tokens]
    layer_str = "Layer 0 (Early)" if is_early else f"Layer {layer_idx if layer_idx >= 0 else len(pre_model.blocks)+layer_idx} (Late)"
    
    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(5*cols, 9))
    for h in range(cols):
        sns.heatmap(pre_attn[h], xticklabels=tokens_str, yticklabels=tokens_str, cmap='Blues', ax=axes[0, h], cbar=False)
        axes[0, h].set_title(f'Pretrained - Head {h}', fontsize=12, fontweight='bold')
        for tick in axes[0, h].get_xticklabels():
            tick.set_fontproperties(devanagari_prop)
            tick.set_rotation(90)
        for tick in axes[0, h].get_yticklabels():
            tick.set_fontproperties(devanagari_prop)
            tick.set_rotation(0)
        
        sns.heatmap(ft_attn[h], xticklabels=tokens_str, yticklabels=tokens_str, cmap='Blues', ax=axes[1, h], cbar=False)
        axes[1, h].set_title(f'Finetuned - Head {h}', fontsize=12, fontweight='bold')
        for tick in axes[1, h].get_xticklabels():
            tick.set_fontproperties(devanagari_prop)
            tick.set_rotation(90)
        for tick in axes[1, h].get_yticklabels():
            tick.set_fontproperties(devanagari_prop)
            tick.set_rotation(0)
        
    plt.suptitle(f'{lang} {layer_str} Attention: Pretrained vs Finetuned', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = f"phase3_{lang.lower()}_attn_compare_layer{'0' if is_early else '5'}.png"
    os.makedirs('report/plots', exist_ok=True)
    plt.savefig(f'report/plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved report/plots/{filename}")

def plot_attention_multihead(tok, model, text, layer_idx=0, prefix='phase2_hindi', title_prefix='Standard Hindi'):
    tokens = tok.encode(text)
    input_ids = torch.tensor([tokens])
    model.eval()
    with torch.no_grad():
        _, _, attn = model(input_ids, return_attention=True)
        
    attn = attn[layer_idx][0].cpu().numpy()
    tokens_str = [tok.decode([t]) for t in tokens]
    
    cols = 4
    fig, axes = plt.subplots(1, cols, figsize=(5*cols, 4.5))
    for h in range(cols):
        sns.heatmap(attn[h], xticklabels=tokens_str, yticklabels=tokens_str, cmap='Blues', ax=axes[h], cbar=False)
        axes[h].set_title(f'{title_prefix} - Head {h}', fontsize=12, fontweight='bold')
        for tick in axes[h].get_xticklabels():
            tick.set_fontproperties(devanagari_prop)
            tick.set_rotation(90)
        for tick in axes[h].get_yticklabels():
            tick.set_fontproperties(devanagari_prop)
            tick.set_rotation(0)
        
    plt.suptitle(f'{title_prefix} Layer {layer_idx}', fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    filename = f"{prefix}_attn_layer{layer_idx}.png"
    os.makedirs('report/plots', exist_ok=True)
    plt.savefig(f'report/plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved report/plots/{filename}")

def calc_entropy_distance(model, tok, text):
    tokens = tok.encode(text)
    input_ids = torch.tensor([tokens])
    model.eval()
    with torch.no_grad():
        _, _, attn = model(input_ids, return_attention=True)
        
    seq_len = len(tokens)
    n_layers = len(attn)
    n_heads = attn[0].shape[1]
    
    entropy = np.zeros((n_layers, n_heads))
    distance = np.zeros((n_layers, n_heads))
    dist_matrix = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])
    
    for l in range(n_layers):
        a = attn[l][0].cpu().numpy()
        for h in range(n_heads):
            p = np.clip(a[h], 1e-10, 1.0)
            h_entropy = -np.sum(p * np.log2(p), axis=-1).mean()
            entropy[l, h] = h_entropy
            h_dist = np.sum(p * dist_matrix, axis=-1).mean()
            distance[l, h] = h_dist
            
    return entropy, distance

def plot_entropy_distance(entropy, distance, filename='phase2_hindi_attn_entropy_distance.png', prefix='Standard Hindi'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(entropy, annot=True, fmt=".2f", cmap="viridis", ax=ax1)
    ax1.set_title(f'{prefix} Model: Attention Entropy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Head', fontsize=11)
    ax1.set_ylabel('Layer', fontsize=11)
    
    sns.heatmap(distance, annot=True, fmt=".2f", cmap="magma", ax=ax2)
    ax2.set_title(f'{prefix} Model: Avg Attention Distance', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Head', fontsize=11)
    ax2.set_ylabel('Layer', fontsize=11)
    
    plt.tight_layout()
    os.makedirs('report/plots', exist_ok=True)
    plt.savefig(f'report/plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved report/plots/{filename}")

def create_model_from_sd(sd):
    vocab_size, d_model = sd['token_emb.weight'].shape
    d_ff = sd['blocks.0.ffn.linear1.weight'].shape[0]
    try:
        context_len = sd['pos_emb.weight'].shape[0]
        return GPTLanguageModel(vocab_size=vocab_size, d_model=d_model, n_heads=8, n_layers=6, d_ff=d_ff, context_len=context_len)
    except KeyError:
        return GPTLanguageModelAblated(vocab_size=vocab_size, d_model=d_model, n_heads=8, n_layers=6, d_ff=d_ff, context_len=256)

def main():
    print("--- 1. Hindi Models ---")
    tok_h = TokenizerWrapper('checkpoints/phase3/checkpoints/pretrained/hindi/tokenizer.json')
    sd_h_pre = load_sd('checkpoints/phase3/checkpoints/pretrained/hindi/model_final.pt', map_location='cpu')
    model_h_pre = create_model_from_sd(sd_h_pre)
    model_h_pre.load_state_dict(sd_h_pre)
    
    sd_h_ft = load_sd('checkpoints/phase3/checkpoints/finetuned/hindi/model_finetuned.pt', map_location='cpu')
    model_h_ft = create_model_from_sd(sd_h_ft)
    model_h_ft.load_state_dict(sd_h_ft)
    
    sample_text_h = "तार्किक क्रम के अनुसार, मान लीजिए ओडिशा की आय अधिक है। किसकी आय अधिक है? उत्तर: ओडिशा"
    
    plot_attention_comparison_multihead(tok_h, model_h_pre, model_h_ft, sample_text_h, "Hindi", layer_idx=0, is_early=True)
    plot_attention_comparison_multihead(tok_h, model_h_pre, model_h_ft, sample_text_h, "Hindi", layer_idx=5, is_early=False)
    
    plot_attention_multihead(tok_h, model_h_pre, sample_text_h, layer_idx=0, prefix='phase2_hindi', title_prefix='Standard Hindi')
    plot_attention_multihead(tok_h, model_h_pre, sample_text_h, layer_idx=3, prefix='phase2_hindi', title_prefix='Standard Hindi')
    plot_attention_multihead(tok_h, model_h_pre, sample_text_h, layer_idx=5, prefix='phase2_hindi', title_prefix='Standard Hindi')
    ent_std_h, dist_std_h = calc_entropy_distance(model_h_pre, tok_h, sample_text_h)
    plot_entropy_distance(ent_std_h, dist_std_h, filename='phase2_hindi_attn_entropy_distance.png', prefix='Standard Hindi')
    
    print("\n--- 2. Nepali Models ---")
    tok_n = TokenizerWrapper('checkpoints/phase3/checkpoints/pretrained/nepali/tokenizer.json')
    sd_n_pre = load_sd('checkpoints/phase3/checkpoints/pretrained/nepali/model_final.pt', map_location='cpu')
    model_n_pre = create_model_from_sd(sd_n_pre)
    model_n_pre.load_state_dict(sd_n_pre)
    
    sd_n_ft = load_sd('checkpoints/phase3/checkpoints/finetuned/nepali/model_finetuned.pt', map_location='cpu')
    model_n_ft = create_model_from_sd(sd_n_ft)
    model_n_ft.load_state_dict(sd_n_ft)
    
    sample_text_n = "तार्किक सम्बन्ध अनुसार, मानौं वीरगन्ज को प्रतिव्यक्ति आय 352 हजार रुपैयाँ छ र भक्तपुर को प्रतिव्यक्ति आय 226 हजार रुपैयाँ छ। बढी प्रतिव्यक्ति आय कसको छ?\nउत्तर: वीरगन्ज"
    plot_attention_comparison_multihead(tok_n, model_n_pre, model_n_ft, sample_text_n, "Nepali", layer_idx=0, is_early=True)
    plot_attention_comparison_multihead(tok_n, model_n_pre, model_n_ft, sample_text_n, "Nepali", layer_idx=5, is_early=False)
    
    plot_attention_multihead(tok_n, model_n_pre, sample_text_n, layer_idx=0, prefix='phase2_nepali', title_prefix='Standard Nepali')
    plot_attention_multihead(tok_n, model_n_pre, sample_text_n, layer_idx=3, prefix='phase2_nepali', title_prefix='Standard Nepali')
    plot_attention_multihead(tok_n, model_n_pre, sample_text_n, layer_idx=5, prefix='phase2_nepali', title_prefix='Standard Nepali')
    ent_std_n, dist_std_n = calc_entropy_distance(model_n_pre, tok_n, sample_text_n)
    plot_entropy_distance(ent_std_n, dist_std_n, filename='phase2_nepali_attn_entropy_distance.png', prefix='Standard Nepali')
    
    print("\n--- 3. Ablated Model ---")
    sd_ablated = load_sd('checkpoints/bonus/bonus_ablation_output/model_ablated.pt', map_location='cpu')
    model_ablated = create_model_from_sd(sd_ablated)
    model_ablated.load_state_dict(sd_ablated)
    
    plot_attention_multihead(tok_h, model_ablated, sample_text_h, layer_idx=0, prefix='bonus_ablation', title_prefix='Ablated Hindi')
    plot_attention_multihead(tok_h, model_ablated, sample_text_h, layer_idx=5, prefix='bonus_ablation', title_prefix='Ablated Hindi')
    
    ent_abl, dist_abl = calc_entropy_distance(model_ablated, tok_h, sample_text_h)
    plot_entropy_distance(ent_abl, dist_abl, filename='bonus_ablation_attn_entropy_distance.png', prefix='Ablated Hindi')

if __name__ == "__main__":
    main()

