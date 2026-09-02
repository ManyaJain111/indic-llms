import os
import tarfile
import torch
from torch.utils.data import IterableDataset

# ---------------------------------------------------------------------------
# Map HuggingFace repo IDs to the actual filename inside each repo.
# These are the tar.gz files you uploaded in Phase 1.
# ---------------------------------------------------------------------------
HF_ARCHIVE_MAP = {
    'manyajain111/hindi-lma-corpus':  'hindi_submission_phase1.tar.gz',
    'manyajain111/nepali-lma-corpus': 'nepali_submission_phase1.tar.gz',
}


def _iter_texts_from_tar(tar_path):
    """Yield (doc_index, text) for every .txt file inside the tar archive."""
    with tarfile.open(tar_path, 'r:*') as tar:
        for idx, member in enumerate(tar):
            if not member.isfile():
                continue
            name_lower = member.name.lower()
            # Accept .txt files and any file without a known binary extension
            if name_lower.endswith('.txt') or (
                not any(name_lower.endswith(ext) for ext in
                        ['.py', '.json', '.csv', '.gz', '.zip',
                         '.jpg', '.png', '.pdf', '.bin', '.pt'])
            ):
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    text = f.read().decode('utf-8', errors='ignore').strip()
                except Exception:
                    continue
                if text:
                    yield idx, text


class LMAIterableDataset(IterableDataset):
    """
    Streams text from a HuggingFace-hosted .tar.gz archive.

    The archive is downloaded once via hf_hub_download (cached by HF).
    Train / validation / test splits are simulated:
        - validation: docs where doc_index % 10 == 0  (~10%)
        - test:       docs where doc_index % 10 == 1  (~10%)
        - train:      all remaining docs              (~80%)
    """

    def __init__(self, hf_dataset_path, tokenizer, context_len,
                 split='train', max_tokens=None, batch_size=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.max_tokens = max_tokens
        self.split = split

        # Resolve archive filename
        if hf_dataset_path not in HF_ARCHIVE_MAP:
            raise ValueError(
                f"Unknown dataset '{hf_dataset_path}'. "
                f"Add it to HF_ARCHIVE_MAP in dataset.py."
            )
        archive_filename = HF_ARCHIVE_MAP[hf_dataset_path]

        # Download and cache the tar.gz (this is fast on Kaggle; cached on disk)
        print(f"[Dataset] Resolving {archive_filename} from {hf_dataset_path} ...")
        from huggingface_hub import hf_hub_download
        self.tar_path = hf_hub_download(
            repo_id=hf_dataset_path,
            filename=archive_filename,
            repo_type='dataset',
        )
        print(f"[Dataset] Archive ready at: {self.tar_path}")

    def _doc_belongs_to_split(self, idx):
        if self.split == 'validation':
            return idx % 10 == 0
        elif self.split == 'test':
            return idx % 10 == 1
        else:  # train
            return idx % 10 >= 2

    def __iter__(self):
        buffer = []
        tokens_yielded = 0

        for doc_idx, text in _iter_texts_from_tar(self.tar_path):
            if not self._doc_belongs_to_split(doc_idx):
                continue

            tokens = self.tokenizer.encode(text)
            # Clamp any -1 unknown tokens to 0 (padding idx) so nn.Embedding is safe
            tokens = [max(t, 0) for t in tokens]

            # Append a space-token separator between documents
            sep = self.tokenizer.encode(' ')
            tokens += [max(s, 0) for s in sep] if sep else []

            buffer.extend(tokens)

            # Yield non-overlapping windows of (context_len + 1) tokens
            while len(buffer) >= self.context_len + 1:
                chunk = buffer[:self.context_len + 1]
                buffer = buffer[self.context_len + 1:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                yield x, y

                tokens_yielded += self.context_len
                if self.max_tokens is not None and tokens_yielded >= self.max_tokens:
                    return


def get_dataloader(hf_dataset_path, tokenizer, context_len,
                   split='train', batch_size=8, max_tokens=None):
    dataset = LMAIterableDataset(
        hf_dataset_path, tokenizer, context_len,
        split=split, max_tokens=max_tokens
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
