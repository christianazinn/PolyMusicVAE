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


def interpolate_base(
    model: MusicVAE,
    tokenizer: REMI,
    tensors: list[torch.Tensor],
    synthesizer,
    num_steps: int = 10,
    do_spherical: bool = False,
    sample_rate: int = 48000,
    ii: int = -1,
):
    model.eval()
    interpolated: list[torch.Tensor] = model.interpolate(
        *tensors, num_steps, do_spherical
    )
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

    # Create figure and plot piano roll data
    h, w = concat_pianoroll[0, 0].shape
    n_sections = len(scores)

    # Create background with color gradient
    cmap = plt.cm.gnuplot
    adj = 100
    background = np.zeros((h, w, 3))

    # Calculate section widths (assuming equal spacing at 32 ticks each)
    section_width = w // n_sections
    for i in range(n_sections):
        t = i / (n_sections - 1 + adj)
        color = cmap(t)[:3]
        start = i * section_width
        end = (i + 1) * section_width if i < n_sections - 1 else w
        background[:, start:end] = color

    # Combine background with piano roll using different colors
    frame = concat_pianoroll[0, 0]  # sustained notes
    onset = concat_pianoroll[1, 0]  # note onsets

    final_img = background.copy()
    # Green/cyan for sustained notes
    sustained_mask = (frame > 0) & (onset == 0)
    final_img[sustained_mask] = [1, 1, 1]
    # Yellow for note onsets
    onset_mask = onset > 0
    final_img[onset_mask] = [0, 0.8, 0.6]

    plt.figure(figsize=(12, 6))
    plt.imshow(final_img, origin="lower", aspect="auto", interpolation="nearest")
    plt.title("Piano Roll")
    plt.xlabel("Time (ticks)")
    plt.ylabel("Pitch")
    plt.tight_layout()
    plt.savefig(
        f"test/pianoroll/concatenated_{ii}_ct.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


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
        interpolate_base(
            model,
            tokenizer,
            tensors,
            synthesizer,
            num_steps,
            do_spherical,
            sample_rate,
            ii,
        )

        # FIXME empty space at the end of a section is a byproduct of training on REMI tokenized data that allows for notes to cross bar boundaries
        # which may or may not even be desirable at all.
        # TODO: ask martin?


if __name__ == "__main__":
    ds, _, _, _ = create_splits(
        ds_path="/home/christian/vae/data_nb_1/b", val_split=0.0, test_split=0.0
    )
    tokenizer = REMI()
    model = MusicVAE.load_id(25)
    model.eval()
    # model = MusicVAE.load_from_checkpoint("checkpoints/last.ckpt")
    test_interpolate_for(model, tokenizer, ds, iiter=10, do_spherical=False)
