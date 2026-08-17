import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

class MemmapDataset(Dataset):
    def __init__(self, bin_path, seq_len=4096):
        self.bin_path = bin_path
        self.seq_len = seq_len
        # Lazy initialization for memmap
        self.data = None
        self._len = os.path.getsize(bin_path) // 4 # uint32 is 4 bytes
        
    def _lazy_init(self):
        if self.data is None:
            self.data = np.memmap(self.bin_path, dtype=np.uint32, mode='r')

    def __len__(self):
        # Valid windows need start + seq_len + 1 <= _len, i.e.
        # idx * seq_len + seq_len + 1 <= _len  =>  idx < (_len - 1) / seq_len.
        # (The previous formula dropped one valid sample per dataset.)
        return max(0, (self._len - 1) // self.seq_len)

    def __getitem__(self, idx):
        self._lazy_init()
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

def create_dataloader(bin_path, batch_size, seq_len, is_distributed=True, is_train=True):
    dataset = MemmapDataset(bin_path, seq_len)
    sampler = DistributedSampler(dataset, shuffle=is_train) if is_distributed else None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=0, # Must be 0 to prevent Kaggle /dev/shm deadlock
        pin_memory=False,
        drop_last=True
    )
    return loader, sampler
