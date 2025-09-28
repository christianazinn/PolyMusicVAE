from dataset import MusicDataset
from datasets import load_from_disk
from miditok import REMI

def main():
    # tok = REMI()
    # print(tok.vocab)
    ds = load_from_disk("/home/christian/vae/data_nb_1/a")
    # dataset = MusicDataset(ds["train"], num_bars=1)
    print(ds[0])

if __name__ == "__main__":
    main()