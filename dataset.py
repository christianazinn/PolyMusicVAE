import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
from os import PathLike
from typing import Dict, List, Optional, Tuple
import json
import pickle
from functools import partial


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
        return {"sequence": torch.tensor(s, dtype=torch.long), "length": len(s)}


def collate_fn(batch: List[Dict], pad_token_id: int = 0):
    sequences = [item["sequence"] for item in batch]
    lengths = [item["length"] for item in batch]

    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=pad_token_id
    )

    return {
        "sequences": padded_sequences,
        "target_sequences": padded_sequences.clone(),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def create_splits(
    ds_path: PathLike, val_split: float = 0.1, test_split: float = 0.1, seed: int = 42
) -> Tuple[HFDataset, HFDataset, HFDataset, Dict]:
    try:
        dataset = load_from_disk(ds_path)
        did = dataset.info.description
    except Exception:
        dataset = load_dataset(ds_path)["train"]
        # stupid manual hack
        did = '{"num_bars": 1, "vocab_size": 284, "bar_id": 4, "bos_id": 1, "eos_id": 2, "pad_id": 0, "max_seq_len": 256}'
    if test_split > 0:
        train_val_dataset = dataset.train_test_split(test_size=test_split, seed=seed)
        test_dataset = train_val_dataset["test"]
        remaining_dataset = train_val_dataset["train"]
        adjusted_val_split = val_split / (1 - test_split)
    else:
        test_dataset = None
        remaining_dataset = dataset
        adjusted_val_split = val_split

    if val_split > 0:
        train_val_split = remaining_dataset.train_test_split(
            test_size=adjusted_val_split, seed=seed
        )
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
    else:
        train_dataset = remaining_dataset
        val_dataset = None

    print(
        f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}, Test: {len(test_dataset) if test_dataset else 0}"
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        json.loads(did),
    )


def create_length_weighted_sampler(dataset, power=0.0, max_samples=2**24 - 1):
    lengths = [len(dataset.ds[i]["s"]) for i in range(len(dataset))][:max_samples]
    weights = [length**power for length in lengths]

    num_samples = min(len(dataset), max_samples)

    return WeightedRandomSampler(
        weights=weights,
        # TODO cannot be over 2**24 - 1 due to some PyTorch limitation
        # which is apparently not documented anywhere (how delightful)
        num_samples=num_samples,
        replacement=True,
    )


def create_dataloaders(
    ds_path: PathLike,
    batch_size: int = 32,
    val_split: float = 0.0,
    test_split: float = 0.0,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
    sampler_power: float = 0.0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
    train_hf, val_hf, test_hf, config = create_splits(
        ds_path, val_split, test_split, seed
    )

    collate_func = partial(collate_fn, pad_token_id=config["pad_id"])
    dl_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_func,
    }

    # assert sampler_power > 0.0, "testing, this should be passed through"

    train_dataset = MusicDataset(train_hf)
    if sampler_power > 0.0:
        ppath = f"data/train_sampler_power_{sampler_power}.pkl"
        try:
            print(f"attempting to load sampler from {ppath}")
            with open(ppath, "rb") as f:
                train_sampler = pickle.load(f)
        except FileNotFoundError:
            print("didn't find it, prepare to wait (a lot)")
            train_sampler = create_length_weighted_sampler(train_dataset, sampler_power)
            with open(ppath, "wb") as f:
                pickle.dump(train_sampler, f)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, drop_last=True, **dl_kwargs
    )
    if val_hf:
        val_dataset = MusicDataset(val_hf)
        val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)
    else:
        val_loader = None
    if test_hf:
        test_dataset = MusicDataset(test_hf)
        test_loader = DataLoader(test_dataset, shuffle=False, **dl_kwargs)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader, config
