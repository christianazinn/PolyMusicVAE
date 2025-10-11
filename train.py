import sys
import shutil
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from model import MusicVAE, get_callbacks
from dataset import create_dataloaders
from config_loader import load_config, print_config_types
import wandb


def run_single_training(config_path: str):
    """Run a single training job from a config file."""
    wandb.finish()
    # Load config
    config = load_config(config_path)
    print("\n=== Config ===")
    print_config_types(config)
    print("===================\n")

    run_name = config["name"]
    print(f"\n{'='*60}")
    print(f"Starting training run: {run_name}")
    print(f"{'='*60}\n")

    # Create dataloaders
    train_loader, val_loader, _, config_data = create_dataloaders(**config["data"])

    # Create model
    model_config = {**config["model"], **config_data}
    model = MusicVAE(**model_config)

    # Setup trainer
    trainer_config = config["trainer"].copy()
    trainer_config["logger"] = WandbLogger(
        project="music-vae", name=run_name, log_model=True
    )
    trainer_config["callbacks"] = get_callbacks()

    import lightning as L

    trainer = L.Trainer(**trainer_config)

    trainer.fit(model, train_loader, val_loader)

    # Post-training cleanup
    checkpoint_dir = Path("checkpoints")
    run_dir = checkpoint_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Move last.ckpt to run-specific folder
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        shutil.move(str(last_ckpt), str(run_dir / "last.ckpt"))
        print(f"Moved last.ckpt to {run_dir}")

    # Remove intermediate checkpoints (keep only last.ckpt in run folder)
    for ckpt in checkpoint_dir.glob("music-vae-epoch=*-val"):
        ckpt.unlink()
        print(f"Removed intermediate checkpoint: {ckpt.name}")

    print(f"\nCompleted training run: {run_name}\n")
    wandb.finish()
    del trainer_config["logger"]


"""
TODO: see this passage from MusicVAE:
As a proof that modeling musical sequences with a recurrent
VAE is possible, we first tried modeling 2-bar (T = 32)
monophonic music sequences (melodies and drum patterns)
with a flat decoder. The model was given a tolerance of
48 free bits (≈33.3 nats) and had the KL cost weight, β,
annealed from 0.0 to 0.2 with exponential rate 0.99999.
Scheduled sampling was introduced with an inverse sigmoid
rate of 2000

TODO: exp annealing for beta
TODO: scheduled sampling: try inv sigmoid or cosine annealing
TODO: implement the stupid annealihng yourself for beta/lr/ss!

All models were trained using Adam (Kingma & Ba, 2014)
with a learning rate annealed from 10^-3 to 10^-5 with ex-
ponential decay rate 0.9999 and a batch size of 512. The
2- and 16-bar models were run for 50k and 100k gradient
updates, respectively. We used a cross-entropy loss against
the ground-truth output with scheduled sampling (Bengio
et al., 2015) for 2-bar models and teacher forcing for 16-bar
models.

bsz 512 * 50k gradient updates -> bsz 64 * 400k updates = 25.6M examples
TODO: adjust learning rate and maybe annealing method
TODO: may need to adjust for polyphonic music

TODO: try other methods of deduplicating data, such as sampling uniformly over seqlen
TODO: relatedly, you may need to reduce epoch length b/c you have 24M samples
"""


def main(config_files: list[str]):
    """Run multiple training jobs in sequence."""
    print(f"Queued {len(config_files)} training runs")

    for i, config_path in enumerate(config_files, 1):
        print(f"\n[{i}/{len(config_files)}] Processing {config_path}")
        try:
            run_single_training(config_path)
        except Exception as e:
            print(f"ERROR in {config_path}: {e}")
            print("Continuing to next run...")
            continue

    print("\n" + "=" * 60)
    print("All training runs completed!")
    print("=" * 60)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python train.py <start> <end>"
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    configs = []
    for config_file in sorted(Path("configs/runs").glob("*.yaml")):
        num = int(config_file.stem.split("_")[0])
        if start <= num <= end:
            configs.append(str(config_file))

    main(configs)
