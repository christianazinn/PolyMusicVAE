import matplotlib.pyplot as plt
import torch
from miditok import REMI
from os import PathLike
from symusic import Score
from model import MusicVAE
from dataset import create_dataloaders

# interpolate between two given midi files and visualize the results
def test_interpolate(model: MusicVAE, tokenizer: REMI, path1: PathLike, path2: PathLike, num_steps: int = 10):
    # assume scores are 1 bar only
    tokens = [tokenizer.encode(Score.from_file(path))[0].ids[1:] for path in [path1, path2]]
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
    _, axes = plt.subplots(len(piano_rolls), 1, figsize=(3, 3*len(piano_rolls)))
    for i, pianoroll in enumerate(piano_rolls):
        axes[i].imshow(pianoroll[0, 0] + pianoroll[1, 0], 
                            origin='lower', aspect='auto',
                            extent=[0, pianoroll.shape[3], 0, 128])
    plt.tight_layout()
    plt.savefig('test/interpolated.png', dpi=300, bbox_inches='tight')
    plt.close()

# does the decoder work fine with random noise? posterior collapse
def test_random_noise(model: MusicVAE, tokenizer: REMI, num_samples: int = 5):
    scores = []
    for _ in range(num_samples):
        z = torch.randn(1, model.hparams.latent_dim).cuda()
        ids = model.decode_autoregressive(z)
        scores.append(tokenizer.decode(ids.cpu().numpy()).resample(tpq=4, min_dur=1))
    piano_rolls = [
        score.pianoroll(
            modes=["frame", "onset"],
            pitch_range=[0, 128],
            encode_velocity=False
        ) for score in scores
    ]
    _, axes = plt.subplots(len(piano_rolls), 1, figsize=(3, 3*len(piano_rolls)))
    for i, pianoroll in enumerate(piano_rolls):
        axes[i].imshow(pianoroll[0, 0] + pianoroll[1, 0], 
                            origin='lower', aspect='auto',
                            extent=[0, pianoroll.shape[3], 0, 128])
    plt.tight_layout()
    plt.savefig('test/random_samples.png', dpi=300, bbox_inches='tight')
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
    score = Score.from_file(path)
    tokenized = tokenizer.encode(score)[0].ids[1:]  # remove BOS
    tensor = torch.tensor(tokenized, dtype=torch.int32).unsqueeze(0).cuda()
    with torch.no_grad():
        latent_dist, _ = model.encode(tensor)
        reconstructed_ids = model.decode_autoregressive(latent_dist.mean)
    reconstructed_score = tokenizer.decode(reconstructed_ids.cpu().numpy()).resample(tpq=4, min_dur=1)
    
    original_pianoroll = score.pianoroll(
        modes=["frame", "onset"],
        pitch_range=[0, 128],
        encode_velocity=False
    )
    reconstructed_pianoroll = reconstructed_score.pianoroll(
        modes=["frame", "onset"],
        pitch_range=[0, 128],
        encode_velocity=False
    )
    
    _, axes = plt.subplots(2, 1, figsize=(6, 6))
    axes[0].imshow(original_pianoroll[0, 0] + original_pianoroll[1, 0], 
                        origin='lower', aspect='auto',
                        extent=[0, original_pianoroll.shape[3], 0, 128])
    axes[0].set_title('Original')
    axes[1].imshow(reconstructed_pianoroll[0, 0] + reconstructed_pianoroll[1, 0], 
                        origin='lower', aspect='auto',
                        extent=[0, reconstructed_pianoroll.shape[3], 0, 128])
    axes[1].set_title('Reconstructed')
    plt.tight_layout()
    plt.savefig('test/reconstruction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def main():
    tokenizer = REMI()
    model = MusicVAE.load_from_checkpoint("checkpoints/last.ckpt")
    # test_interpolate(model, tokenizer, "test/bar_9.mid", "test/bar_17.mid")
    # test_random_noise(model, tokenizer, num_samples=5)
    # test_latents(model, num_samples=1000)
    test_reconstruction(model, tokenizer, "test/bar_9.mid")

if __name__ == "__main__":
    main()