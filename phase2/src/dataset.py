import os
import tarfile
import glob
import torch
from torch.utils.data import IterableDataset

# ---------------------------------------------------------------------------
# Map HuggingFace repo IDs to the actual filename inside each repo.
# ---------------------------------------------------------------------------
HF_ARCHIVE_MAP = {
    'manyajain111/hindi-lma-corpus':  'hindi_submission_phase1.tar.gz',
    'manyajain111/nepali-lma-corpus': 'nepali_submission_phase1.tar.gz',
}

# Where to extract the archives on Kaggle (persistent across notebook cells)
EXTRACT_BASE = '/kaggle/working/lma_extracted'


def _extract_tar_once(tar_path, extract_dir):
    """Extract tar.gz to extract_dir only if not already extracted."""
    done_flag = os.path.join(extract_dir, '.extracted')
    if os.path.exists(done_flag):
        print(f'[Dataset] Already extracted at: {extract_dir}')
        return

    print(f'[Dataset] Extracting tar.gz to {extract_dir} (one-time, ~1-2 min) ...')
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

    # Write done flag so we skip extraction on subsequent runs
    with open(done_flag, 'w') as f:
        f.write('done')
    print(f'[Dataset] Extraction complete.')


def _collect_text_files(extract_dir):
    """
    Collect all .txt files under extract_dir, sorted for determinism.
    Falls back to all files without known binary extensions if no .txt found.
    """
    txt_files = sorted(glob.glob(os.path.join(extract_dir, '**', '*.txt'), recursive=True))
    if txt_files:
        return txt_files

    # Fallback: any file that looks like text
    skip_exts = {'.py', '.json', '.csv', '.gz', '.zip', '.jpg', '.png',
                 '.pdf', '.bin', '.pt', '.pyc', '.tar', '.sh', '.gitattributes'}
    all_files = sorted(glob.glob(os.path.join(extract_dir, '**', '*'), recursive=True))
    return [f for f in all_files
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() not in skip_exts
            and not os.path.basename(f).startswith('.')]


class LMAIterableDataset(IterableDataset):
    """
    Streams CLM training windows from extracted text files.

    Extraction is done once on first use (cached via .extracted flag).
    Train / val / test splits are simulated by file index:
        - validation: files where file_idx % 10 == 0  (~10%)
        - test:       files where file_idx % 10 == 1  (~10%)
        - train:      all remaining files              (~80%)
    """

    def __init__(self, hf_dataset_path, tokenizer, context_len,
                 split='train', max_tokens=None, batch_size=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.max_tokens = max_tokens
        self.split = split

        if hf_dataset_path not in HF_ARCHIVE_MAP:
            raise ValueError(f"Unknown dataset '{hf_dataset_path}'.")
        archive_filename = HF_ARCHIVE_MAP[hf_dataset_path]

        # Unique extract dir per dataset
        lang_tag = hf_dataset_path.split('/')[-1]  # e.g. hindi-lma-corpus
        self.extract_dir = os.path.join(EXTRACT_BASE, lang_tag)

        # Download the tar.gz (cached by HF after first download)
        print(f'[Dataset] Resolving {archive_filename} ...')
        from huggingface_hub import hf_hub_download
        tar_path = hf_hub_download(
            repo_id=hf_dataset_path,
            filename=archive_filename,
            repo_type='dataset',
        )
        print(f'[Dataset] Archive at: {tar_path}')

        # Extract once
        _extract_tar_once(tar_path, self.extract_dir)

        # Collect file list
        all_files = _collect_text_files(self.extract_dir)
        print(f'[Dataset] Found {len(all_files)} text files after extraction.')

        # Split by file index
        if split == 'validation':
            self.files = [f for i, f in enumerate(all_files) if i % 10 == 0]
        elif split == 'test':
            self.files = [f for i, f in enumerate(all_files) if i % 10 == 1]
        else:  # train
            self.files = [f for i, f in enumerate(all_files) if i % 10 >= 2]

        print(f'[Dataset] Split "{split}": {len(self.files)} files.')

    def __iter__(self):
        buffer = []
        tokens_yielded = 0

        for fpath in self.files:
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
            except Exception:
                continue
            if not text:
                continue

            tokens = [max(t, 0) for t in self.tokenizer.encode(text)]
            sep = self.tokenizer.encode(' ')
            tokens += [max(s, 0) for s in sep] if sep else []
            buffer.extend(tokens)

            while len(buffer) >= self.context_len + 1:
                chunk = buffer[:self.context_len + 1]
                buffer = buffer[self.context_len + 1:]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:],  dtype=torch.long))
                tokens_yielded += self.context_len
                if self.max_tokens and tokens_yielded >= self.max_tokens:
                    return


def get_dataloader(hf_dataset_path, tokenizer, context_len,
                   split='train', batch_size=8, max_tokens=None):
    dataset = LMAIterableDataset(
        hf_dataset_path, tokenizer, context_len,
        split=split, max_tokens=max_tokens
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
