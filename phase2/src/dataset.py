import os, numpy as np, torch
from torch.utils.data import IterableDataset

BIN_MAP = {
    "manyajain111/hindi-lma-corpus":  {"train":"/kaggle/working/hindi_train.bin","validation":"/kaggle/working/hindi_val.bin","test":"/kaggle/working/hindi_test.bin"},
    "manyajain111/nepali-lma-corpus": {"train":"/kaggle/working/nepali_train.bin","validation":"/kaggle/working/nepali_val.bin","test":"/kaggle/working/nepali_test.bin"},
}

class LMAIterableDataset(IterableDataset):
    def __init__(self, hf_dataset_path, tokenizer, context_len, split="train", max_tokens=None, batch_size=1):
        super().__init__()
        self.context_len = context_len
        self.max_tokens = max_tokens
        path = BIN_MAP[hf_dataset_path][split]
        print(f"[Dataset] Memory-mapping {path}...")
        self.tokens = np.memmap(path, dtype=np.uint16, mode='r')
        print(f"[Dataset] {len(self.tokens):,} tokens ready.")

    def __iter__(self):
        n, i, yielded = len(self.tokens), 0, 0
        while i + self.context_len + 1 <= n:
            chunk = self.tokens[i:i+self.context_len+1]
            x = torch.from_numpy(chunk[:-1].astype(np.int64))
            y = torch.from_numpy(chunk[1:].astype(np.int64))
            yield x, y
            i += self.context_len + 1
            yielded += self.context_len
            if self.max_tokens and yielded >= self.max_tokens:
                return

def get_dataloader(hf_dataset_path, tokenizer, context_len, split="train", batch_size=8, max_tokens=None):
    ds = LMAIterableDataset(hf_dataset_path, tokenizer, context_len, split=split, max_tokens=max_tokens)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
