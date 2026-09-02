import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset

class LMAIterableDataset(IterableDataset):
    def __init__(self, hf_dataset_path, tokenizer, context_len, split="train", max_tokens=None, batch_size=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.max_tokens = max_tokens
        
        # Load dataset in streaming mode
        self.dataset = load_dataset(hf_dataset_path, split=split, streaming=True)
        
    def __iter__(self):
        buffer = []
        tokens_yielded = 0

        for item in self.dataset:
            # We assume the text is in the 'text' column
            text = item.get('text', '')
            if not text:
                continue

            tokens = self.tokenizer.encode(text)
            # BPE returns -1 for unknown tokens; clamp to 0 (padding idx) so
            # nn.Embedding does not receive an out-of-range index and crash.
            tokens = [max(t, 0) for t in tokens]

            # Separate documents with a space token (id of b' ' = 32 in byte-BPE)
            # to prevent the model from learning across document boundaries.
            sep = self.tokenizer.encode(' ')
            tokens += [max(s, 0) for s in sep] if sep else []

            buffer.extend(tokens)
            
            # Yield chunks of (context_len + 1)
            while len(buffer) >= self.context_len + 1:
                chunk = buffer[:self.context_len + 1]
                buffer = buffer[self.context_len + 1:] # non-overlapping chunks
                
                # Input and Target
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                
                yield x, y
                
                tokens_yielded += self.context_len
                if self.max_tokens is not None and tokens_yielded >= self.max_tokens:
                    return

def get_dataloader(hf_dataset_path, tokenizer, context_len, split="train", batch_size=8, max_tokens=None):
    dataset = LMAIterableDataset(hf_dataset_path, tokenizer, context_len, split=split, max_tokens=max_tokens)
    # Using DataLoader without shuffling since it's an IterableDataset
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
