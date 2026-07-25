import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class SFTDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, map_location='cpu')
        self.input_ids = data['input_ids']
        self.labels = data['labels']
        
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        # We need to shift labels in the dataloader or the model?
        # Actually, in causal LM, logits are shifted by 1 relative to labels.
        # F.cross_entropy expects logits of shape (B, T, V) and labels of shape (B, T).
        # We usually shift them inside the loss function. 
        # But for simplicity, we provide x and y directly here.
        x = self.input_ids[idx][:-1]
        y = self.labels[idx][1:]
        return x, y

def create_sft_dataloader(data_path, batch_size, is_distributed=True, is_train=True):
    dataset = SFTDataset(data_path)
    sampler = DistributedSampler(dataset, shuffle=is_train) if is_distributed else None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if torch.cuda.is_available() else None,
        drop_last=True
    )
    return loader, sampler
