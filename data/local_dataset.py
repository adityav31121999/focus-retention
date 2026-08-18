import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LocalMemmapDataset(Dataset):
    def __init__(self, memmap_path: str, seq_len: int = 4096, dtype=np.uint32):
        self.data = np.memmap(memmap_path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.num_samples = len(self.data) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {"input_ids": chunk, "labels": chunk.clone()}

def get_local_dataloader(dataset: LocalMemmapDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)