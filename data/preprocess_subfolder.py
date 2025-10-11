import numpy as np
import sys
import os
import json
from dataclasses import dataclass
from datasets import Dataset
from miditok import REMI, TokSequence
from pathlib import Path
from symusic import Score
from tqdm import tqdm


@dataclass
class PreprocessConfig:
    num_bars: int
    vocab_size: int
    bar_id: int
    bos_id: int
    eos_id: int
    pad_id: int
    max_seq_len: int


def preprocess_file(midi_file: str | Path, tokenizer: REMI, config: PreprocessConfig):
    score = Score.from_file(midi_file)
    tokens = tokenizer.encode(score)
    if type(tokens) is not list:
        tokens = [tokens]

    # extract all nonempty num_bars segments from the file
    for tokseq in tokens:
        assert type(tokseq) is TokSequence
        ids = np.array(tokseq.ids)
        bar_breaks = np.where(ids == config.bar_id)[0]

        for i in range(len(bar_breaks) - config.num_bars):
            start = bar_breaks[i] + 1  # do not include the opening bar token
            end = bar_breaks[i + config.num_bars]
            tolist = ids[start:end].tolist() + [config.eos_id]
            if (
                len(tolist) > config.num_bars + 1
                and len(tolist) <= config.max_seq_len - 1
            ):  # room for BOS
                yield tolist


def process(
    file_paths, output_path: str | Path, tokenizer: REMI, config: PreprocessConfig
):
    def chunk_generator_with_stats():
        for file_path in tqdm(file_paths, desc="Processing files", unit=" files"):
            try:
                for sample in preprocess_file(file_path, tokenizer, config):
                    yield {"s": sample}
            except Exception:
                continue

    dataset = Dataset.from_generator(chunk_generator_with_stats)
    dataset.info.description = json.dumps(config.__dict__)
    dataset.save_to_disk(output_path)  # type: ignore
    return dataset


# in lmd_full/e/ e59b70ca47d81f8f3507d9c421eabeb2 segfaults when processing and I can't figure out why
def sanitize(file_paths: list[Path]):
    for i, path in enumerate(file_paths):
        if "e59b70ca47d81f8f3507d9c421eabeb2" in str(path):
            return file_paths[:i] + file_paths[i + 1 :]
    return file_paths


def main():
    if len(sys.argv) == 4:
        _, split, num_bars_str, max_seq_len_str = sys.argv
        if split not in "abcdef0123456789":
            print("Split must be a valid hex")
            sys.exit(1)
        split = f"{split}"
    elif len(sys.argv) == 3:
        # this doesn't actually work yet: TBD
        _, num_bars_str, max_seq_len_str = sys.argv
        split = ""
    else:
        print("Usage: python preprocess.py <num_bars> <max_seq_len>")
        print("or for one split: python preprocess.py <split> <num_bars> <max_seq_len>")
        sys.exit(1)

    num_bars = int(num_bars_str)
    tokenizer = REMI()
    config = PreprocessConfig(
        num_bars=num_bars,
        vocab_size=len(tokenizer.vocab),
        bar_id=tokenizer.vocab["Bar_None"],
        bos_id=tokenizer.vocab["BOS_None"],
        eos_id=tokenizer.vocab["EOS_None"],
        pad_id=tokenizer.vocab["PAD_None"],
        max_seq_len=int(max_seq_len_str),
    )

    data_path = Path(f"/mnt/t/midi/lmd_full/{split}")
    # b/c of how it's launched, safer to abspath
    out_path = Path(f"/home/christian/vae/data_nb_{num_bars}/{split}")
    out_path.mkdir(exist_ok=True, parents=True)

    print("Globbing...")
    file_paths = list(data_path.rglob("*.mid"))
    before_len = len(file_paths)
    with open(os.path.dirname(__file__) + "/lakh_good_martin.json", "r") as f:
        good_files = set(json.load(f))
    file_paths = [fp for fp in file_paths if fp.name.rstrip(".mid") in good_files]
    print(f"Globbed {len(file_paths)} files (originally {before_len}). Processing...")

    if split in ["e", ""]:
        print("Bad file in list, removing...")
        file_paths = sanitize(file_paths)
        print("Removed bad file. Processing...")

    process(file_paths, out_path, tokenizer, config)


if __name__ == "__main__":
    main()
