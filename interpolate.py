import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from miditok import REMI
from miditok.utils import concat_scores
from symusic import dump_wav, Synthesizer, BuiltInSF3
from tqdm import tqdm
from dataset import create_splits
from model import MusicVAE


def test_interpolate_for(
    model: MusicVAE,
    tokenizer: REMI,
    ds,
    num_steps: int = 10,
    do_spherical: bool = False,
    iiter: int = 1,
    sample_rate: int = 48000,
):
    lds = len(ds)
    os.makedirs("test/pianoroll", exist_ok=True)
    os.makedirs("test/audio", exist_ok=True)

    synthesizer = Synthesizer(
        sf_path=BuiltInSF3.MuseScoreGeneral().path(download=True),
        sample_rate=sample_rate,
    )

    for ii in tqdm(range(iiter), desc="Interpolate and render"):
        tokens = [ds[idx]["s"] for idx in np.random.randint(0, lds, size=2)]
        # assume scores are 1 bar only
        tensors = [
            torch.tensor(t, dtype=torch.int32).unsqueeze(0).cuda() for t in tokens
        ]
        interpolated: list[torch.Tensor] = model.interpolate(*tensors, num_steps, do_spherical)
        interpolated = [tensors[0]] + interpolated + [tensors[1]]
        scores = [
            tokenizer.decode(ids.cpu().numpy()).resample(tpq=8, min_dur=1)
            for ids in interpolated
        ]

        # render to music
        # assuming 4/4 time, one bar at 8 ticks/quarter note is 32 ticks, so we assume end_ticks are just multiples of 32
        concat_score = concat_scores(
            scores, end_ticks=[(i + 1) * 32 for i in range(len(scores))]
        )
        audio_data = synthesizer.render(concat_score, stereo=True)
        dump_wav(f"test/audio/concatenated_{ii}.wav", audio_data, sample_rate)
        concat_pianoroll = concat_score.pianoroll(
            modes=["frame", "onset"], pitch_range=[0, 128], encode_velocity=False
        )

        plt.figure(figsize=(12, 6))
        plt.imshow(
            concat_pianoroll[0, 0] + concat_pianoroll[1, 0],
            origin="lower",
            aspect="auto",
            extent=[0, concat_pianoroll.shape[3], 0, 128],
        )
        plt.title("Piano Roll (Track 0)")
        plt.xlabel("Time (in ticks)")
        plt.ylabel("Pitch (MIDI note number)")
        # TODO: re-add the background gradient/a way to differentiate sections
        plt.tight_layout()
        plt.savefig(
            f"test/pianoroll/concatenated_{ii}_ct.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # FIXME empty space at the end of a section is a byproduct of training on REMI tokenized data that allows for notes to cross bar boundaries
        # which may or may not even be desirable at all.
        # TODO: ask martin?

"""
        # create and render piano roll (OLD)
        piano_rolls = [
            score.pianoroll(
                modes=["frame", "onset"], pitch_range=[0, 128], encode_velocity=False
            )
            for score in scores
        ]

        combined = np.concatenate([pr[0, 0] + pr[1, 0] for pr in piano_rolls], axis=1)

        h, w = combined.shape
        n_rolls = len(piano_rolls)

        # Track actual widths of each piano roll
        widths = [pr.shape[3] for pr in piano_rolls]
        cumulative_widths = np.cumsum([0] + widths)

        cmap = plt.cm.gnuplot
        adj = 100

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
        ax.imshow(final_img, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_ylabel("Pitch")
        plt.tight_layout()
        plt.savefig(
            f"test/pianoroll/interpolated_{ii}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
"""


if __name__ == "__main__":
    ds, _, _, _ = create_splits(
        ds_path="/home/christian/vae/data_nb_1/b", val_split=0.0, test_split=0.0
    )
    tokenizer = REMI()
    model = MusicVAE.load_from_checkpoint(
        "checkpoints/24_beta_0.2_1024d_free_bits_24_latent_1024d_lr_1e-5_20_epochs/last.ckpt"
    )
    model.eval()
    # model = MusicVAE.load_from_checkpoint("checkpoints/last.ckpt")
    test_interpolate_for(model, tokenizer, ds, iiter=10, do_spherical=False)
