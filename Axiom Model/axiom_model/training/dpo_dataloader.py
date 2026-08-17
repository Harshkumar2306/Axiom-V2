import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict

class DPODataset(Dataset):
    def __init__(self, data_path: str):
        try:
            self.data = torch.load(data_path, map_location='cpu', weights_only=False)
        except TypeError:
            self.data = torch.load(data_path, map_location='cpu')
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def dpo_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Dynamically pads the chosen and rejected sequences independently within the batch.
    This saves massive amounts of GPU memory compared to padding everything to 4096.
    """
    # Kaggle T4 OOM Safety: Hard limit DPO sequences to 512 tokens.
    MAX_DPO_SEQ_LEN = 512
    
    chosen_len = min(MAX_DPO_SEQ_LEN, max(len(item["chosen_ids"]) for item in batch))
    rejected_len = min(MAX_DPO_SEQ_LEN, max(len(item["rejected_ids"]) for item in batch))
    max_len = max(chosen_len, rejected_len)
    
    # cl100k_base eot_token is 100257. We use this as pad_id to prevent embedding noise.
    pad_id = 100257
    ignore_index = -100
    
    batch_chosen_ids = []
    batch_chosen_labels = []
    batch_rejected_ids = []
    batch_rejected_labels = []
    
    for item in batch:
        c_ids = item["chosen_ids"][:MAX_DPO_SEQ_LEN]
        c_labels = item["chosen_labels"][:MAX_DPO_SEQ_LEN]
        r_ids = item["rejected_ids"][:MAX_DPO_SEQ_LEN]
        r_labels = item["rejected_labels"][:MAX_DPO_SEQ_LEN]
        
        # Pad chosen sequences to max_len
        c_pad = max_len - len(c_ids)
        if c_pad > 0:
            c_ids = torch.cat([c_ids, torch.full((c_pad,), pad_id, dtype=torch.long)])
            c_labels = torch.cat([c_labels, torch.full((c_pad,), ignore_index, dtype=torch.long)])
            
        # Pad rejected sequences to max_len
        r_pad = max_len - len(r_ids)
        if r_pad > 0:
            r_ids = torch.cat([r_ids, torch.full((r_pad,), pad_id, dtype=torch.long)])
            r_labels = torch.cat([r_labels, torch.full((r_pad,), ignore_index, dtype=torch.long)])
            
        batch_chosen_ids.append(c_ids)
        batch_chosen_labels.append(c_labels)
        batch_rejected_ids.append(r_ids)
        batch_rejected_labels.append(r_labels)
        
    return {
        "chosen_ids": torch.stack(batch_chosen_ids),
        "chosen_labels": torch.stack(batch_chosen_labels),
        "rejected_ids": torch.stack(batch_rejected_ids),
        "rejected_labels": torch.stack(batch_rejected_labels)
    }

def create_dpo_dataloader(data_path: str, batch_size: int, is_distributed: bool, is_train: bool = True):
    dataset = DPODataset(data_path)
    
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(is_train and not is_distributed),
        sampler=sampler,
        collate_fn=dpo_collate_fn,
        pin_memory=True,
        num_workers=0, # Set to 0 to prevent /dev/shm deadlocks on Kaggle multi-GPU
        drop_last=is_train
    )
    
    return dataloader, sampler
