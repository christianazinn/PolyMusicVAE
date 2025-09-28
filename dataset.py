import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk, Dataset as HFDataset
from os import PathLike
from typing import Dict, List, Optional, Tuple
import json


class MusicDataset(Dataset):
    def __init__(
        self, 
        ds: HFDataset,
    ):
        self.ds = ds

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        s = self.ds[idx]["s"]
        return {
            'sequence': torch.tensor(s, dtype=torch.long),
            'length': len(s)
        }

def collate_fn(batch: List[Dict], pad_token_id: int = 0):
    sequences = [item['sequence'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    padded_sequences = pad_sequence(
        sequences, 
        batch_first=True, 
        padding_value=pad_token_id
    )
    
    return {
        'sequences': padded_sequences,
        'target_sequences': padded_sequences.clone(),  # teacher forcing
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


def create_splits(
    ds_path: PathLike,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[HFDataset, HFDataset, HFDataset, Dict]:
    dataset = load_from_disk(ds_path)
    if test_split > 0:
        train_val_dataset = dataset.train_test_split(
            test_size=test_split, 
            seed=seed
        )
        test_dataset = train_val_dataset['test']
        remaining_dataset = train_val_dataset['train']
        adjusted_val_split = val_split / (1 - test_split)
    else:
        test_dataset = None
        remaining_dataset = dataset
        adjusted_val_split = val_split
    
    if val_split > 0:
        train_val_split = remaining_dataset.train_test_split(
            test_size=adjusted_val_split,
            seed=seed
        )
        train_dataset = train_val_split['train']
        val_dataset = train_val_split['test']
    else:
        train_dataset = remaining_dataset
        val_dataset = None
    
    print(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}, Test: {len(test_dataset) if test_dataset else 0}")
    
    return train_dataset, val_dataset, test_dataset, json.loads(dataset.info.description)


def create_dataloaders(
    ds_path: PathLike,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
    train_hf, val_hf, test_hf, config = create_splits(
        ds_path, val_split, test_split, seed
    )
    
    train_dataset = MusicDataset(train_hf)
    val_dataset = MusicDataset(val_hf) if val_hf else None
    test_dataset = MusicDataset(test_hf) if test_hf else None
    
    collate_func = lambda batch: collate_fn(batch, config['pad_id'])
    dl_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': collate_func
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **dl_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dl_kwargs
    ) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dl_kwargs
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader, config
