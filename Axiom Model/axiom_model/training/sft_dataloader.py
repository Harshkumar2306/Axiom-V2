import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import itertools

from axiom_model.training.dataloader import MemmapDataset

class SFTDataset(Dataset):
    def __init__(self, data_path):
        try:
            data = torch.load(data_path, map_location='cpu', weights_only=False)
        except TypeError:
            data = torch.load(data_path, map_location='cpu')
        self.input_ids = data['input_ids']
        self.labels = data['labels']
        
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        x = self.input_ids[idx][:-1]
        y = self.labels[idx][1:]
        return x, y

class FastForwardSampler:
    def __init__(self, base_sampler, skip_items):
        self.base_sampler = base_sampler
        # CRITICAL BUG FIX: Modulo arithmetic prevents out-of-bounds on multi-epoch resume!
        self.skip_items = skip_items % len(base_sampler) if len(base_sampler) > 0 else 0
        
    def __iter__(self):
        it = iter(self.base_sampler)
        list(itertools.islice(it, self.skip_items))
        return it
        
    def __len__(self):
        return len(self.base_sampler) - self.skip_items
        
    def set_epoch(self, epoch):
        if hasattr(self.base_sampler, 'set_epoch'):
            self.base_sampler.set_epoch(epoch)

class BaseSFTLoader:
    def fast_forward(self, batches_to_skip):
        raise NotImplementedError

class StandardSFTLoader(BaseSFTLoader):
    def __init__(self, dataset, batch_size, is_distributed, is_train):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = DistributedSampler(dataset, shuffle=is_train) if is_distributed else None
        self.is_train = is_train
        
        self.loader = self._build_loader(self.sampler)
        
    def _build_loader(self, sampler):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None and self.is_train),
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )

    def fast_forward(self, batches_to_skip):
        if batches_to_skip == 0 or self.sampler is None: return
        items_to_skip = batches_to_skip * self.batch_size
        ff_sampler = FastForwardSampler(self.sampler, items_to_skip)
        self.loader = self._build_loader(ff_sampler)

    def set_epoch(self, epoch):
        if self.sampler: self.sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

class MixedReplayDataloader(BaseSFTLoader):
    def __init__(self, sft_path, pretrain_path, batch_size, seq_len=4096, is_distributed=True, is_train=True):
        self.sft_engine = StandardSFTLoader(SFTDataset(sft_path), batch_size, is_distributed, is_train)
        
        pretrain_dataset = MemmapDataset(pretrain_path, seq_len)
        self.pretrain_sampler = DistributedSampler(pretrain_dataset, shuffle=is_train) if is_distributed else None
        self.pretrain_loader = DataLoader(
            pretrain_dataset, batch_size=batch_size, sampler=self.pretrain_sampler,
            shuffle=(self.pretrain_sampler is None and is_train),
            num_workers=0, pin_memory=False, drop_last=True
        )
        self.pretrain_batch_size = batch_size
        
        self.num_batches = len(self.sft_engine)
        # Ratio is implicit: Every 10th batch is Pretrain, 9 are SFT (90% SFT / 10% Replay)
        
    def fast_forward(self, batches_to_skip):
        if batches_to_skip == 0: return
        
        # Calculate exactly how many batches belong to SFT vs Pretrain
        skipped_pretrain = batches_to_skip // 10
        skipped_sft = batches_to_skip - skipped_pretrain
        
        self.sft_engine.fast_forward(skipped_sft)
        
        if self.pretrain_sampler:
            items_to_skip = skipped_pretrain * self.pretrain_batch_size
            ff_pretrain = FastForwardSampler(self.pretrain_sampler, items_to_skip)
            self.pretrain_loader = DataLoader(
                self.pretrain_loader.dataset, batch_size=self.pretrain_batch_size,
                sampler=ff_pretrain, num_workers=0, pin_memory=False, drop_last=True
            )

    def set_epoch(self, epoch):
        self.sft_engine.set_epoch(epoch)
        if self.pretrain_sampler: self.pretrain_sampler.set_epoch(epoch)

    def __iter__(self):
        sft_iter = iter(self.sft_engine)
        pretrain_iter = iter(self.pretrain_loader)
        
        for i in range(self.num_batches):
            if i % 10 == 0:
                try:
                    yield next(pretrain_iter)
                except StopIteration:
                    pretrain_iter = iter(self.pretrain_loader)
                    yield next(pretrain_iter)
            else:
                try:
                    yield next(sft_iter)
                except StopIteration:
                    sft_iter = iter(self.sft_engine)
                    yield next(sft_iter)

    def __len__(self):
        return self.num_batches

def create_sft_dataloader(data_path, batch_size, is_distributed=True, is_train=True, replay_data_path=None, seq_len=4096):
    if replay_data_path:
        engine = MixedReplayDataloader(data_path, replay_data_path, batch_size, seq_len, is_distributed, is_train)
    else:
        engine = StandardSFTLoader(SFTDataset(data_path), batch_size, is_distributed, is_train)
    
    return engine, engine
