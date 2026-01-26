import matplotlib.pyplot as plt
import torch
from miditok import REMI, TokenizerConfig
from os import PathLike
from symusic import Score, Synthesizer, BuiltInSF3
from model import MusicVAE
from dataset import create_dataloaders
from interpolate import interpolate_base
from utils import extract_notes, compute_f1
import numpy as np
from tqdm import tqdm
import argparse


# interpolate between two given midi files and visualize the results
def test_interpolate(
    model: MusicVAE,
    tokenizer: REMI,
    path1: PathLike,
    path2: PathLike,
    num_steps: int = 10,
):
    # assume scores are 1 bar only
    tokens = [
        tokenizer.encode(Score.from_file(path))[0].ids[1:] for path in [path1, path2]
    ]
    tensors = [torch.tensor(t, dtype=torch.int32).unsqueeze(0).cuda() for t in tokens]

    synthesizer = Synthesizer(
        sf_path=BuiltInSF3.MuseScoreGeneral().path(download=True),
        sample_rate=48000,
    )
    interpolate_base(model, tokenizer, tensors, synthesizer, num_steps)


# does the decoder work fine with random noise? posterior collapse
def test_random_noise(model: MusicVAE, tokenizer: REMI, num_samples: int = 5):
    scores = []
    for _ in range(num_samples):
        z = torch.randn(1, model.hparams.latent_dim).cuda()
        ids = model.decode_autoregressive(z)
        scores.append(tokenizer.decode(ids.cpu().numpy()).resample(tpq=4, min_dur=1))
    piano_rolls = [
        score.pianoroll(
            modes=["frame", "onset"], pitch_range=[0, 128], encode_velocity=False
        )
        for score in scores
    ]
    _, axes = plt.subplots(len(piano_rolls), 1, figsize=(3, 3 * len(piano_rolls)))
    for i, pianoroll in enumerate(piano_rolls):
        axes[i].imshow(
            pianoroll[0, 0] + pianoroll[1, 0],
            origin="lower",
            aspect="auto",
            extent=[0, pianoroll.shape[3], 0, 128],
        )
    plt.tight_layout()
    plt.savefig("test/random_samples.png", dpi=300, bbox_inches="tight")
    plt.close()


# are dataset latents close to each other? posterior collapse
def test_latents(model: MusicVAE, num_samples: int = 20):
    loader, _, _, _ = create_dataloaders(ds_path="/home/christian/vae/data_nb_1/a")
    model.eval()
    latent_means = []

    with torch.no_grad():
        samples_collected = 0
        for batch in loader:
            if samples_collected >= num_samples:
                break
            sequences = batch["sequences"].to(model.device)
            lengths = batch.get("lengths", None)
            batch_size = min(sequences.shape[0], num_samples - samples_collected)
            sequences = sequences[:batch_size]
            if lengths is not None:
                lengths = lengths[:batch_size]
            latent_dist, _ = model.encode(sequences, lengths)
            latent_means.append(latent_dist.mean)
            print(latent_dist.mean.shape)
            samples_collected += batch_size

    all_means = torch.cat(latent_means, dim=0)[:num_samples]
    normalized = torch.nn.functional.normalize(all_means, dim=1)
    similarity_matrix = torch.mm(normalized, normalized.t())
    mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    similarities = similarity_matrix[mask]

    total_pairs = len(similarities)
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]

    print(f"Similarity analysis for {total_pairs:,} pairs:")
    print("Threshold | Count     | Percentage")
    print("-" * 35)

    for threshold in thresholds:
        count = (similarities > threshold).sum().item()
        percentage = (count / total_pairs) * 100
        print(f">  {threshold:.1f}    | {count:8,} | {percentage:8.3f}%")

    print(f"\nSummary statistics:")
    print(f"Mean: {similarities.mean():.4f}")
    print(f"Std:  {similarities.std():.4f}")
    print(f"Min:  {similarities.min():.4f}")
    print(f"Max:  {similarities.max():.4f}")


def test_reconstruction(model: MusicVAE, tokenizer: REMI, path: PathLike):
    model.eval()
    score = Score.from_file(path)
    tokenized = tokenizer.encode(score)[0].ids[1:]  # remove BOS
    print(tokenized)
    tensor = torch.tensor(tokenized, dtype=torch.int32).unsqueeze(0).cuda()
    with torch.no_grad():
        latent_dist, _ = model.encode(tensor)
        reconstructed_ids = model.decode_autoregressive(latent_dist.sample())
    reconstructed_score = tokenizer.decode(reconstructed_ids.cpu().numpy()).resample(
        tpq=8, min_dur=1
    )

    xscore = tokenizer.decode(tensor.cpu().numpy()).resample(tpq=8, min_dur=1)

    original_pianoroll = xscore.pianoroll(
        modes=["frame", "onset"], pitch_range=[0, 128], encode_velocity=False
    )
    reconstructed_pianoroll = reconstructed_score.pianoroll(
        modes=["frame", "onset"], pitch_range=[0, 128], encode_velocity=False
    )

    _, axes = plt.subplots(2, 1, figsize=(6, 6))
    axes[0].imshow(
        original_pianoroll[0, 0] + original_pianoroll[1, 0],
        origin="lower",
        aspect="auto",
        extent=[0, original_pianoroll.shape[3], 0, 128],
    )
    axes[0].set_title("Original")
    axes[1].imshow(
        reconstructed_pianoroll[0, 0] + reconstructed_pianoroll[1, 0],
        origin="lower",
        aspect="auto",
        extent=[0, reconstructed_pianoroll.shape[3], 0, 128],
    )
    axes[1].set_title("Reconstructed")
    plt.tight_layout()
    plt.savefig("test/reconstruction.png", dpi=300, bbox_inches="tight")
    plt.close()


def test_file_reconstruction(model: MusicVAE, tokenizer: REMI, path: PathLike):
    model.eval()
    score = Score.from_file(path)
    tokenized = tokenizer.encode(score)

    bar_start = 16
    num_bars = 8
    bar_id = tokenizer.vocab["Bar_None"]
    track_tokens = [[], [], []]
    reconst_tokens = [[], [], []]

    for j, track in enumerate(tokenized):
        real_track = np.array(track.ids[1:])  # remove BOS
        bar_breaks = np.where(real_track == bar_id)[0]
        for i in range(num_bars):
            start = bar_breaks[i + bar_start - 1] + 1
            end = bar_breaks[i + bar_start]
            bar_tokens = real_track[start:end].tolist()
            track_tokens[j].extend(bar_tokens)
            track_tokens[j].append(bar_id)

            if len(bar_tokens) > 0:
                tensor = torch.tensor(bar_tokens, dtype=torch.int32).unsqueeze(0).cuda()
                with torch.no_grad():
                    latent_dist, _ = model.encode(tensor)
                    reconstructed_ids = model.decode_autoregressive(
                        latent_dist.sample()
                    )
                    rccd = reconstructed_ids.cpu().numpy().tolist()[0]
                    reconst_tokens[j].extend(rccd)
                    reconst_tokens[j].append(bar_id)
            else:
                reconst_tokens[j].append(bar_id)

    original_semiscore = tokenizer.decode(track_tokens)
    original_semiscore.dump_midi("test/reconst.mid")
    reconst_score = tokenizer.decode(reconst_tokens)
    reconst_score.dump_midi("test/rec.mid")


# F1 functions moved to utils.py - use extract_notes and compute_f1 from there


def test_f1_scores(
    model: MusicVAE,
    tokenizer: REMI,
    path: PathLike | None = None,
    num_val_phrases: int | None = None,
    ds_path: str = "/home/christian/vae/data_nb_1_combined",
    mode: str = "op",
    use_mean: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Compute F1 reconstruction scores for onset+pitch (op) or onset+pitch+duration (opd).

    This follows the evaluation methodology from PhraseVAE where:
    - F1op: A note is correct if onset time and pitch match
    - F1opd: A note is correct if onset time, pitch, AND duration match

    Args:
        model: Trained MusicVAE model
        tokenizer: REMI tokenizer (should match model's training tokenizer)
        path: Path to a single MIDI file to evaluate (mutually exclusive with num_val_phrases)
        num_val_phrases: Number of phrases to evaluate from validation set
                        (mutually exclusive with path)
        ds_path: Dataset path for loading validation phrases
        mode: "op" for onset+pitch, "opd" for onset+pitch+duration
        use_mean: If True, use latent mean for reconstruction; if False, sample
        verbose: Print detailed results

    Returns:
        Dict with aggregate metrics: 'f1', 'precision', 'recall', 'num_phrases',
        and per-phrase 'scores' list
    """
    if path is None and num_val_phrases is None:
        raise ValueError("Must specify either 'path' or 'num_val_phrases'")
    if path is not None and num_val_phrases is not None:
        raise ValueError("Cannot specify both 'path' and 'num_val_phrases'")
    if mode not in ("op", "opd"):
        raise ValueError(f"mode must be 'op' or 'opd', got {mode}")

    model.eval()
    all_scores = []

    # Aggregate counts for micro-averaged F1
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        if path is not None:
            # Single file mode
            score = Score.from_file(path)
            tokenized = tokenizer.encode(score)[0].ids[1:]  # remove BOS

            if len(tokenized) == 0:
                raise ValueError(f"Empty tokenization for {path}")

            tensor = (
                torch.tensor(tokenized, dtype=torch.int32).unsqueeze(0).to(model.device)
            )
            latent_dist, _ = model.encode(tensor)
            z = latent_dist.mean if use_mean else latent_dist.sample()
            reconstructed_ids = model.decode_autoregressive(z)

            # Decode to scores
            original_score = tokenizer.decode([tokenized])
            reconstructed_score = tokenizer.decode(reconstructed_ids.cpu().numpy())

            # Extract notes and compute F1
            orig_notes = extract_notes(original_score, mode)
            recon_notes = extract_notes(reconstructed_score, mode)
            scores = compute_f1(orig_notes, recon_notes)

            all_scores.append(scores)
            total_tp += scores["tp"]
            total_fp += scores["fp"]
            total_fn += scores["fn"]

        else:
            # Validation set mode
            val_loader, _, _, _ = create_dataloaders(ds_path=ds_path)

            phrases_evaluated = 0
            with tqdm(total=num_val_phrases) as pbar:
                for batch in val_loader:
                    if phrases_evaluated >= num_val_phrases:
                        break

                    sequences = batch["sequences"].to(model.device)
                    batch_size = min(
                        sequences.shape[0], num_val_phrases - phrases_evaluated
                    )

                    for i in range(batch_size):
                        seq = sequences[i : i + 1]

                        # Remove padding for original
                        seq_np = seq.cpu().numpy()[0]
                        # Find actual length (up to first pad or end)
                        pad_id = model.pad_id
                        valid_mask = seq_np != pad_id
                        if valid_mask.sum() == 0:
                            continue
                        valid_tokens = seq_np[valid_mask].tolist()

                        # Encode and reconstruct
                        latent_dist, _ = model.encode(seq)
                        z = latent_dist.mean if use_mean else latent_dist.sample()
                        reconstructed_ids = model.decode_autoregressive(z)

                        # Decode to scores
                        try:
                            original_score = tokenizer.decode([valid_tokens])
                            recon_tokens = reconstructed_ids.cpu().numpy()[0].tolist()
                            # Remove BOS/EOS/PAD from reconstruction
                            recon_tokens = [
                                t for t in recon_tokens if t not in (0, 1, 2)
                            ]
                            reconstructed_score = tokenizer.decode([recon_tokens])

                            # Extract notes and compute F1
                            orig_notes = extract_notes(original_score, mode)
                            recon_notes = extract_notes(reconstructed_score, mode)
                            scores = compute_f1(orig_notes, recon_notes)

                            all_scores.append(scores)
                            total_tp += scores["tp"]
                            total_fp += scores["fp"]
                            total_fn += scores["fn"]

                        except Exception as e:
                            if verbose:
                                print(
                                    f"Warning: Failed to decode phrase {phrases_evaluated + i}: {e}"
                                )
                            continue

                    phrases_evaluated += batch_size
                    pbar.update(batch_size)

    # Compute aggregate metrics (micro-averaged)
    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    # Compute macro-averaged F1 (average of per-phrase F1s)
    macro_f1 = np.mean([s["f1"] for s in all_scores]) if all_scores else 0.0
    macro_precision = (
        np.mean([s["precision"] for s in all_scores]) if all_scores else 0.0
    )
    macro_recall = np.mean([s["recall"] for s in all_scores]) if all_scores else 0.0

    results = {
        "mode": mode,
        "num_phrases": len(all_scores),
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "scores": all_scores,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"F1 Scores (mode={mode})")
        print(f"{'='*50}")
        print(f"Phrases evaluated: {len(all_scores)}")
        print(f"\nMicro-averaged (note-level):")
        print(f"  Precision: {micro_precision:.4f}")
        print(f"  Recall:    {micro_recall:.4f}")
        print(f"  F1:        {micro_f1:.4f}")
        print(f"\nMacro-averaged (phrase-level):")
        print(f"  Precision: {macro_precision:.4f}")
        print(f"  Recall:    {macro_recall:.4f}")
        print(f"  F1:        {macro_f1:.4f}")
        print(f"\nNote counts:")
        print(f"  True Positives:  {total_tp:,}")
        print(f"  False Positives: {total_fp:,}")
        print(f"  False Negatives: {total_fn:,}")

        if all_scores:
            f1_values = [s["f1"] for s in all_scores]
            print(f"\nPer-phrase F1 distribution:")
            print(f"  Min:    {min(f1_values):.4f}")
            print(f"  Max:    {max(f1_values):.4f}")
            print(f"  Median: {np.median(f1_values):.4f}")
            print(f"  Std:    {np.std(f1_values):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int)
    args = parser.parse_args()
    tokenizer = REMI()
    # 53, 70, 90 are good
    model = MusicVAE.load_id(args.id)
    model.eval()
    # test_interpolate(
    #     model,
    #     tokenizer,
    #     "test/musicvae_melody_example_1.mid",
    #     "test/musicvae_melody_example_2.mid",
    # )
    # test_random_noise(model, tokenizer, num_samples=5)
    # print([param.dtype for param in model.parameters()])
    # print("\n")
    # test_latents(model, num_samples=1000)
    # test_reconstruction(model, tokenizer, "test/test.mid")
    # test_file_reconstruction(model, tokenizer, "test/001.mid")
    test_f1_scores(model, tokenizer, num_val_phrases=200, mode="opd")


if __name__ == "__main__":
    main()
    # tokenizer = REMI(TokenizerConfig(use_rests=True))
    # print(tokenizer.vocab)
