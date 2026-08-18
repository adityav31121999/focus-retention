from typing import Iterator
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from .tokeniser import TokenizerManager

class HuggingFaceStreamer(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: TokenizerManager,
        seq_len: int = 4096,
        dataset_config: str = None,
        split: str = "train",
        buffer_size: int = 20000
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict]:
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True
        ).shuffle(buffer_size=self.buffer_size)

        token_buffer = []
        for sample in dataset:
            text = sample.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text) + [self.tokenizer.tokenizer.eos_token_id]
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.seq_len:
                chunk = token_buffer[:self.seq_len]
                token_buffer = token_buffer[self.seq_len:]
                tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": tensor_chunk, "labels": tensor_chunk.clone()}

def get_hf_dataloader(streamer: HuggingFaceStreamer, batch_size: int) -> DataLoader:
    return DataLoader(streamer, batch_size=batch_size, pin_memory=True)