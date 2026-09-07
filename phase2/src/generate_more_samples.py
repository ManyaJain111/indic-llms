import json
import torch
import torch.nn.functional as F
from model import GPTLanguageModel
from tokenizer_wrapper import TokenizerWrapper

def generate_continuation(model, tokenizer, device, prefix_text, max_new_tokens=50, temperature=None):
    model.eval()
    enc = tokenizer.encode(prefix_text)
    if not enc:
        enc = [1]
    x = torch.tensor([enc], dtype=torch.long, device=device)
    
    with torch.no_grad():
        if temperature is None:
            gen_idx = model.greedy_generate(x, max_new_tokens=max_new_tokens)
        else:
            gen_idx = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)
            
    full_text = tokenizer.decode(gen_idx[0].cpu().tolist())
    return full_text

def run_sampling_for_language(lang, ckpt_path, tok_path, prompts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{lang.upper()}] Loading model from {ckpt_path} on {device}...")
    tokenizer = TokenizerWrapper(tok_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    
    model = GPTLanguageModel(**config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    samples = []
    temps = [('greedy', None), ('temp_0.5', 0.5), ('temp_1.0', 1.0), ('temp_1.5', 1.5)]
    
    for prompt in prompts:
        print(f"Generating for prompt: '{prompt}'")
        sample_entry = {
            "prefix": prompt,
        }
        for name, t in temps:
            gen_text = generate_continuation(model, tokenizer, device, prompt, max_new_tokens=50, temperature=t)
            sample_entry[name] = gen_text
        samples.append(sample_entry)
        
    return samples

if __name__ == '__main__':
    hindi_prompts = [
        "मौसम विभाग के अनुसार",
        "शिक्षा के क्षेत्र में",
        "भारतीय क्रिकेट टीम ने",
        "स्वास्थ्य विशेषज्ञों का कहना है कि",
        "प्रधानमंत्री ने आज नई दिल्ली में"
    ]
    
    nepali_prompts = [
        "मौसम पूर्वानुमान महाशाखाका अनुसार",
        "शिक्षाको क्षेत्रमा सुधार गर्न",
        "नेपाली राष्ट्रिय क्रिकेट टोलीले",
        "स्वास्थ्य मन्त्रालयका अनुसार",
        "काठमाडौं उपत्यकामा आज बिहान"
    ]
    
    # Generate for Hindi
    h_samples = run_sampling_for_language(
        'hindi',
        'checkpoints/hindi/model_final.pt',
        'tokenizers/hindi_tokenizer_16000.json',
        hindi_prompts
    )
    
    # Generate for Nepali
    n_samples = run_sampling_for_language(
        'nepali',
        'checkpoints/nepali/model_final.pt',
        'tokenizers/nepali_tokenizer_16000.json',
        nepali_prompts
    )
    
    with open('reports/hindi/more_samples.json', 'w', encoding='utf-8') as f:
        json.dump(h_samples, f, ensure_ascii=False, indent=2)
    print("Saved reports/hindi/more_samples.json")
    
    with open('reports/nepali/more_samples.json', 'w', encoding='utf-8') as f:
        json.dump(n_samples, f, ensure_ascii=False, indent=2)
    print("Saved reports/nepali/more_samples.json")
