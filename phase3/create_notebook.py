import json

def read_file(path):
    with open(path) as f:
        return f.read()

model_code = read_file("../phase2/src/model.py")
tok_code = read_file("../phase2/src/tokenizer_wrapper.py")

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    lines = [line + "\n" for line in text.split("\n")]
    if lines and lines[-1] == "\n":
        lines = lines[:-1]
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": lines})

add_md("# Phase 3: Reasoning Finetuning\n\nThis notebook fine-tunes Model H and Model L on the synthetic comparative reasoning datasets.")

add_code("!pip install -q huggingface_hub seaborn tqdm")

add_code("""import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import collections
import re
import string
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)""")

add_code("""# Download Pretrained Checkpoints from HF
repo_id = "manyajain111/lma-phase2"
os.makedirs("checkpoints/pretrained/hindi", exist_ok=True)
os.makedirs("checkpoints/pretrained/nepali", exist_ok=True)

files_to_download = [
    ("hindi/model_final.pt", "checkpoints/pretrained/hindi"),
    ("hindi/config.json", "checkpoints/pretrained/hindi"),
    ("hindi/tokenizer.json", "checkpoints/pretrained/hindi"),
    ("nepali/model_final.pt", "checkpoints/pretrained/nepali"),
    ("nepali/config.json", "checkpoints/pretrained/nepali"),
    ("nepali/tokenizer.json", "checkpoints/pretrained/nepali")
]

for f, out_dir in files_to_download:
    if not os.path.exists(os.path.join(out_dir, os.path.basename(f))):
        print(f"Downloading {f}...")
        hf_hub_download(repo_id=repo_id, filename=f, local_dir="checkpoints/pretrained")
print("Downloads complete.")""")

add_code("# --- INLINED DEPENDENCIES (Tokenizer and Model) ---\n\n" + tok_code + "\n\n" + model_code)

add_code("""class ReasoningDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                prompt = d['prompt']
                answer = d['answer']
                
                # Format: {prompt}\\nउत्तर: {answer}
                full_text = f"{prompt}\\nउत्तर: {answer}"
                
                tokens = tokenizer.encode(full_text)
                
                # We need to compute loss only on the answer part.
                # Find length of prompt
                prompt_text = f"{prompt}\\nउत्तर: "
                prompt_len = len(tokenizer.encode(prompt_text))
                
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                
                targets = list(tokens)
                # Mask out prompt tokens
                for i in range(min(prompt_len, len(targets))):
                    targets[i] = -100
                
                pad_len = max_len - len(tokens)
                if pad_len > 0:
                    tokens.extend([0] * pad_len) # assuming pad=0
                    targets.extend([-100] * pad_len)
                
                self.data.append((torch.tensor(tokens, dtype=torch.long), torch.tensor(targets, dtype=torch.long)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]""")

add_code("""def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for idx, targets in tqdm(dataloader, desc="Training"):
        idx, targets = idx.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(idx, targets=targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, targets in dataloader:
            idx, targets = idx.to(device), targets.to(device)
            logits, loss = model(idx, targets=targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def load_model_and_tokenizer(lang_full_name, checkpoint_path):
    with open(f"checkpoints/pretrained/{lang_full_name}/config.json", 'r') as f:
        config = json.load(f)
        
    tokenizer = TokenizerWrapper(f"checkpoints/pretrained/{lang_full_name}/tokenizer.json")
    
    model = GPTLanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        context_len=config['context_len'],
        dropout=0.1,
        pad_token_id=0
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.to(device)
    return model, tokenizer""")

add_code("""# SET YOUR DATASET PATH HERE! 
# In Kaggle, datasets extracted from zip are usually in a folder inside the input.
# For example, if you uploaded phase3_kaggle_data.zip, it might be in '/kaggle/input/phase3-kaggle-data/data'
DATA_DIR = "/kaggle/input/phase3-kaggle-data/data" # UPDATE THIS IF NEEDED

def run_finetuning(lang_short, lang_full, epochs=3, batch_size=32, lr=5e-5):
    print(f"\\n{'='*40}\\nFinetuning {lang_full.upper()}\\n{'='*40}")
    model, tokenizer = load_model_and_tokenizer(lang_full, f"checkpoints/pretrained/{lang_full}/model_final.pt")
    
    train_ds = ReasoningDataset(os.path.join(DATA_DIR, f"{lang_short}/train.jsonl"), tokenizer)
    val_ds = ReasoningDataset(os.path.join(DATA_DIR, f"{lang_short}/val.jsonl"), tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    out_dir = f"checkpoints/finetuned/{lang_full}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/model_finetuned.pt")
    print(f"Saved finetuned model to {out_dir}/model_finetuned.pt")
    
    return model, tokenizer""")

add_code("""# Run Finetuning! 
try:
    model_h_ft, tok_h = run_finetuning("hi", "hindi", epochs=3)
    model_l_ft, tok_l = run_finetuning("ne", "nepali", epochs=3)
except FileNotFoundError as e:
    print("Dataset not found! Please ensure you added the Kaggle dataset and updated DATA_DIR.", e)""")

add_code("""# Evaluation on Test Set (Exact Match)
def evaluate_exact_match(model, tokenizer, jsonl_path, num_samples=900):
    model.eval()
    correct = 0
    total = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:num_samples]
        
    with torch.no_grad():
        for line in tqdm(lines, desc="Evaluating"):
            d = json.loads(line)
            prompt = d['prompt']
            label = d['label'] # Exact entity name
            
            # Context
            context_str = f"{prompt}\\nउत्तर: "
            idx = torch.tensor([tokenizer.encode(context_str)], dtype=torch.long).to(device)
            
            # Generate up to 20 tokens (sufficient for the answer sentence)
            out_idx = model.greedy_generate(idx, max_new_tokens=20)
            generated_str = tokenizer.decode(out_idx[0].tolist())
            
            # The model output includes the prompt, so isolate the generated part
            gen_answer = generated_str[len(context_str):]
            
            # Exact Match: is the correct label in the generated answer?
            if label in gen_answer:
                correct += 1
            total += 1
            
    return correct / total * 100

print("\\n--- EXACT MATCH ACCURACY ---")
try:
    print("Loading Pretrained Models for Baseline...")
    model_h_pre, _ = load_model_and_tokenizer("hindi", "checkpoints/pretrained/hindi/model_final.pt")
    model_l_pre, _ = load_model_and_tokenizer("nepali", "checkpoints/pretrained/nepali/model_final.pt")
    
    print("\\nEvaluating Pretrained (Baseline)...")
    acc_h_pre = evaluate_exact_match(model_h_pre, tok_h, os.path.join(DATA_DIR, "hi/test.jsonl"), num_samples=200)
    acc_l_pre = evaluate_exact_match(model_l_pre, tok_l, os.path.join(DATA_DIR, "ne/test.jsonl"), num_samples=200)
    print(f"HINDI Pretrained Accuracy:  {acc_h_pre:.2f}%")
    print(f"NEPALI Pretrained Accuracy: {acc_l_pre:.2f}%")
    
    print("\\nEvaluating Finetuned...")
    acc_h_ft = evaluate_exact_match(model_h_ft, tok_h, os.path.join(DATA_DIR, "hi/test.jsonl"), num_samples=200)
    acc_l_ft = evaluate_exact_match(model_l_ft, tok_l, os.path.join(DATA_DIR, "ne/test.jsonl"), num_samples=200)
    print(f"HINDI Finetuned Accuracy:  {acc_h_ft:.2f}%")
    print(f"NEPALI Finetuned Accuracy: {acc_l_ft:.2f}%")
except Exception as e:
    print(f"Evaluation failed: {e}")
""")

add_code("""# Attention Visualization
def plot_attention_heatmaps(model, tokenizer, text, layer_idx=0, head_idx=0):
    model.eval()
    tokens = tokenizer.encode(text)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits, _, attention_maps = model(idx, return_attention=True)
    
    attn = attention_maps[layer_idx][0, head_idx].cpu().numpy()
    
    tok_strs = []
    for t in tokens:
        s = tokenizer.decode([t])
        tok_strs.append(s)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tok_strs, yticklabels=tok_strs, cmap='viridis')
    plt.title(f"Layer {layer_idx}, Head {head_idx}")
    plt.show()

try:
    sample_text = "यदि राम की आयु 25 साल है और श्याम की आयु 30 साल है, तो किसकी आयु कम है?\\nउत्तर: राम"
    print("Pretrained Hindi Model Attention (Layer 5, Head 0):")
    plot_attention_heatmaps(model_h_pre, tok_h, sample_text, layer_idx=5, head_idx=0)
    print("Finetuned Hindi Model Attention (Layer 5, Head 0):")
    plot_attention_heatmaps(model_h_ft, tok_h, sample_text, layer_idx=5, head_idx=0)
except Exception as e:
    print(f"Plotting failed: {e}")
""")

nb = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("phase3_reasoning_finetuning.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
