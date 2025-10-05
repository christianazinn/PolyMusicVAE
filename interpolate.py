import matplotlib.pyplot as plt
import torch
from miditok import REMI
from model import MusicVAE
from dataset import create_splits
import numpy as np


def test_interpolate_for(model: MusicVAE, tokenizer: REMI, ds, num_steps: int = 10, iiter: int = 1):
    lds = len(ds)
    for ii in range(iiter):
        tokens = [ds[idx]["s"] for idx in np.random.randint(0, lds, size=2)]
        # assume scores are 1 bar only
        tensors = [torch.tensor(t, dtype=torch.int32).unsqueeze(0).cuda() for t in tokens]
        interpolated: list[torch.Tensor] = model.interpolate(*tensors, num_steps)
        interpolated = [tensors[0]] + interpolated + [tensors[1]]
        scores = [tokenizer.decode(ids.cpu().numpy()).resample(tpq=4, min_dur=1) for ids in interpolated]
        piano_rolls = [
            score.pianoroll(
                modes=["frame", "onset"],
                pitch_range=[0, 128],
                encode_velocity=False
            ) for score in scores
        ]
        
        combined = np.concatenate([pr[0, 0] + pr[1, 0] for pr in piano_rolls], axis=1)
        
        h, w = combined.shape
        n_rolls = len(piano_rolls)
        
        # Track actual widths of each piano roll
        widths = [pr.shape[3] for pr in piano_rolls]
        cumulative_widths = np.cumsum([0] + widths)
        
        cmap = plt.cm.gnuplot
        adj = 30
        
        background = np.zeros((h, w, 3))
        for i in range(n_rolls):
            t = i / (n_rolls - 1 + adj)
            color = cmap(t)[:3]
            
            start = cumulative_widths[i]
            end = cumulative_widths[i + 1]
            background[:, start:end] = color
        
        final_img = background.copy()
        mask = combined > 0
        final_img[mask] = [1, 1, 1]
        
        fig, ax = plt.subplots(figsize=(20, 3))
        ax.imshow(final_img, origin='lower', aspect='auto', interpolation='nearest')
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_ylabel('Pitch')
        plt.tight_layout()
        plt.savefig(f'test/interpolated_{ii}.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    ds, _, _, _ = create_splits(ds_path="/home/christian/vae/data_nb_1/a", val_split=0.0, test_split=0.0)
    tokenizer = REMI()
    model = MusicVAE.load_from_checkpoint("checkpoints/eager-music-6/music-vae-epoch=02-val/total_loss=0.29.ckpt")
    # model = MusicVAE.load_from_checkpoint("checkpoints/last.ckpt")
    test_interpolate_for(model, tokenizer, ds, iiter=10)