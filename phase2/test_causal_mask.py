import torch
import json
from src.model import GPTLanguageModel

def test_causal_masking():
    print("Running Causal Mask Verification Test...")
    with open("config/model_h_config.json", "r") as f:
        config = json.load(f)
        
    # Initialize a small test model using the config
    torch.manual_seed(42)
    model = GPTLanguageModel(**config)
    model.eval()
    
    # Create two sequences that differ only at the last position
    # T = 10
    seq1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    seq2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 99]])
    
    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)
        
    # The logits for positions 0 through 8 should be EXACTLY identical
    # because they cannot attend to position 9 (the last token)
    diff = torch.max(torch.abs(logits1[0, :9, :] - logits2[0, :9, :]))
    
    print(f"Max difference in logits before changed token: {diff.item()}")
    assert diff.item() < 1e-6, "Causal mask failed! The model is looking ahead."
    
    # The logits for position 9 should be DIFFERENT
    diff_last = torch.max(torch.abs(logits1[0, 9, :] - logits2[0, 9, :]))
    print(f"Max difference in logits at changed token: {diff_last.item()}")
    assert diff_last.item() > 0, "Logits at the changed token should be different."
    
    print("✅ Causal masking successfully verified! The model cannot see the future.")

if __name__ == "__main__":
    test_causal_masking()
