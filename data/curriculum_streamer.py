# data/curriculum_streamer.py
from typing import Iterator, Optional, Dict, Any, List, Union
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from .tokeniser import TokenizerManager


class CurriculumStreamer(IterableDataset):
    """
    Streams tokens across one or multiple datasets sequentially:
    - When a dataset is exhausted, automatically moves to the next one.
    - When all datasets finish, increments the epoch counter and restarts from the first dataset.
    - Tracks active dataset index, sample offset, epoch, and token buffer for seamless checkpoint resuming.
    """
    def __init__(
        self,
        tokenizer: TokenizerManager,
        dataset_name: Optional[Union[str, List[Dict[str, Any]]]] = "roneneldan/TinyStories",
        dataset_config: Optional[str] = None,
        datasets_list: Optional[List[Dict[str, Any]]] = None,
        initial_seq_len: int = 128,
        split: str = "train",
        buffer_size: int = 10000,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = initial_seq_len
        self.split = split
        self.buffer_size = buffer_size
        self.seed = seed
        self.eos_token_id = self.tokenizer.tokenizer.eos_token_id

        # Normalize datasets into a unified list of {"name": ..., "config": ...}
        if datasets_list is not None and len(datasets_list) > 0:
            self.datasets = datasets_list
        elif isinstance(dataset_name, list):
            self.datasets = dataset_name
        else:
            self.datasets = [{"name": dataset_name, "config": dataset_config}]

        # State tracking for checkpointing & resuming
        self.current_ds_idx: int = 0
        self.samples_seen_in_current_ds: int = 0
        self.epoch: int = 0
        self.token_buffer: List[int] = []

    def set_seq_len(self, new_seq_len: int):
        """Dynamically switch context length when transitioning stages."""
        print(f"[Curriculum Data] Switched sequence length to: {new_seq_len}")
        self.seq_len = new_seq_len

    def state_dict(self) -> Dict[str, Any]:
        """Returns streaming progress, active dataset index, epoch, and buffer state."""
        return {
            "current_ds_idx": self.current_ds_idx,
            "samples_seen_in_current_ds": self.samples_seen_in_current_ds,
            "epoch": self.epoch,
            "token_buffer": self.token_buffer,
            "seq_len": self.seq_len,
        }

    def load_state_dict(self, state_dict: Optional[Dict[str, Any]]):
        """Restores dataset index, sample offset, epoch, and token buffer from checkpoint."""
        if not state_dict:
            return
        self.current_ds_idx = state_dict.get("current_ds_idx", 0)
        self.samples_seen_in_current_ds = state_dict.get(
            "samples_seen_in_current_ds", state_dict.get("samples_seen", 0)
        )
        self.epoch = state_dict.get("epoch", 0)
        self.token_buffer = state_dict.get("token_buffer", [])
        self.seq_len = state_dict.get("seq_len", self.seq_len)

        current_ds_name = self.datasets[self.current_ds_idx]["name"]
        print(
            f"📑 [Data State Restored] Epoch: {self.epoch} | "
            f"Dataset [{self.current_ds_idx + 1}/{len(self.datasets)}]: '{current_ds_name}' | "
            f"Samples Seen: {self.samples_seen_in_current_ds:,} | "
            f"Buffer: {len(self.token_buffer)} tokens | Seq Len: {self.seq_len}"
        )

    def __iter__(self) -> Iterator[dict]:
        while True:
            ds_info = self.datasets[self.current_ds_idx]
            ds_name = ds_info["name"]
            ds_config = ds_info.get("config", None)

            print(
                f"\n📖 [Data Streamer] Streaming Dataset [{self.current_ds_idx + 1}/{len(self.datasets)}]: "
                f"'{ds_name}' (Config: {ds_config}) | Epoch: {self.epoch + 1}"
            )

            dataset = load_dataset(
                ds_name,
                ds_config,
                split=self.split,
                streaming=True
            )

            # Fast-forward past already-processed samples if resuming
            if self.samples_seen_in_current_ds > 0:
                print(f"⏩ [Data Streamer] Skipping first {self.samples_seen_in_current_ds:,} samples in '{ds_name}'...")
                dataset = dataset.skip(self.samples_seen_in_current_ds)

            # Shuffle using a seed varied by epoch for diverse ordering across cycles
            dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=self.seed + self.epoch)

            # Stream through current dataset
            for sample in dataset:
                self.samples_seen_in_current_ds += 1
                text = sample.get("text", "")
                if not text:
                    continue

                tokens = self.tokenizer.encode(text) + [self.eos_token_id]
                self.token_buffer.extend(tokens)

                while len(self.token_buffer) >= self.seq_len:
                    chunk = self.token_buffer[:self.seq_len]
                    self.token_buffer = self.token_buffer[self.seq_len:]
                    tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                    yield {"input_ids": tensor_chunk, "labels": tensor_chunk.clone()}

            # --- Current Dataset Exhausted ---
            print(f"\n✅ [Data Streamer] Finished dataset '{ds_name}' ({self.samples_seen_in_current_ds:,} samples).")
            self.current_ds_idx += 1
            self.samples_seen_in_current_ds = 0

            # --- All Datasets Completed: Increment Epoch & Loop Back to Start ---
            if self.current_ds_idx >= len(self.datasets):
                self.current_ds_idx = 0
                self.epoch += 1
                print(f"\n🎉 [Data Streamer] Completed Epoch {self.epoch}! Looping back to first dataset...\n")


def create_curriculum_dataloader(streamer: CurriculumStreamer, batch_size: int = 1) -> DataLoader:
    return DataLoader(streamer, batch_size=batch_size, pin_memory=True)