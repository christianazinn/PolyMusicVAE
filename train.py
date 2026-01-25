import os
import sys
import shutil
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from model import MusicVAE, get_callbacks
from dataset import create_dataloaders
from config_loader import load_config, print_config_types
import lightning as L
import wandb
import torch

try:
    from miditok import REMI, TokenizerConfig
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


def get_num_gpus() -> int:
    """Detect number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def run_single_training(config_path: str):
    wandb.finish()
    config = load_config(config_path)

    # Detect GPUs and scale config accordingly
    num_gpus = get_num_gpus()
    if num_gpus > 1:
        print(f"\n=== MULTI-GPU DETECTED: {num_gpus} GPUs ===")
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(broadcast_buffers=False)
        config["trainer"]["strategy"] = strategy
        # Scale devices
        config["trainer"]["devices"] = num_gpus
        # Scale num_workers (more workers to feed multiple GPUs)
        if "num_workers" in config["data"]:
            original_workers = config["data"]["num_workers"]
            config["data"]["num_workers"] = original_workers * num_gpus
            print(f"Scaled num_workers: {original_workers} -> {config['data']['num_workers']}")
        print(f"Set devices: {num_gpus}")
        print("=" * 40 + "\n")

    print("\n======= Config =======")
    print_config_types(config)
    print("===================\n")

    run_name = config["name"]
    print(f"{'='*60}")
    print(f"Starting training run: {run_name}")
    print(f"{'='*60}\n")

    try:
        train_loader, val_loader, _, config_data = create_dataloaders(**config["data"])
    except Exception:
        config["data"]["ds_path"] = "/app/data"
        train_loader, val_loader, _, config_data = create_dataloaders(**config["data"])

    model_config = {**config["model"], **config_data}

    # Check for staged training configuration
    staged_config = config.get("staged_training")
    if staged_config and staged_config.get("enabled", False):
        print("\n=== STAGED TRAINING MODE ===")
        source_run_id = staged_config.get("source_run_id")
        source_checkpoint = staged_config.get("source_checkpoint")

        if source_run_id:
            checkpoint_dir = Path("checkpoints")
            matching = list(checkpoint_dir.glob(f"{source_run_id}_*"))
            if not matching:
                raise FileNotFoundError(f"No checkpoint folder found for run_id {source_run_id}")
            source_checkpoint = str(matching[0] / "last.ckpt")
            print(f"Loading from run {source_run_id}: {source_checkpoint}")
        elif source_checkpoint:
            print(f"Loading from checkpoint: {source_checkpoint}")
        else:
            raise ValueError("staged_training requires either source_run_id or source_checkpoint")

        # Build stage kwargs from config
        stage_kwargs = {}
        stage_fields = [
            "new_latent_dim", "new_bottleneck_dim", "new_num_queries",
            "training_mode", "freeze_encoder", "freeze_decoder", "freeze_bottleneck",
            "kl_reduction", "learning_rate", "beta_start", "beta_end",
            "beta_warmup_steps", "free_bits",
        ]
        for field in stage_fields:
            if field in staged_config:
                stage_kwargs[field] = staged_config[field]

        model = MusicVAE.load_for_stage(source_checkpoint, **stage_kwargs)
        print(f"=== END STAGED TRAINING SETUP ===\n")
    else:
        model = MusicVAE(**model_config)

    # Set up tokenizer for F1 evaluation during validation
    if TOKENIZER_AVAILABLE:
        ds_path = config["data"].get("ds_path", "")
        use_rests = "rests" in str(ds_path).lower()
        tokenizer = REMI(TokenizerConfig(use_rests=use_rests))
        model.set_tokenizer(tokenizer)
        print(f"Tokenizer set for F1 evaluation (use_rests={use_rests})")

    trainer_config = config["trainer"].copy()
    trainer_config["logger"] = WandbLogger(
        project="music-vae", name=run_name, log_model=True
    )
    trainer_config["callbacks"] = get_callbacks()

    trainer = L.Trainer(use_distributed_sampler=False, **trainer_config)

    trainer.fit(model, train_loader, val_loader)

    checkpoint_dir = Path("checkpoints")
    run_dir = checkpoint_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # move last.ckpt to run-specific folder
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        shutil.move(str(last_ckpt), str(run_dir / "last.ckpt"))
        print(f"Moved last.ckpt to {run_dir}")

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
    print(f"Queued {len(config_files)} training runs")

    for i, config_path in enumerate(config_files, 1):
        print(f"\n[{i}/{len(config_files)}] Processing {config_path}")
        try:
            run_single_training(config_path)
        except Exception as e:
            raise e
            print(f"ERROR in {config_path}: {e}")
            print("Continuing to next run...")
            continue

if __name__ == "__main__":
    if len(sys.argv) == 3:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        is_remote = False
    else:
        try:
            start = int(os.environ.get('START_RUN'))
            end = int(os.environ.get('END_RUN'))
            is_remote = True
        except Exception as e:
            print(e)  # debug
            print("Usage: python train.py <start> <end>")
            raise e

    configs = []
    for config_file in sorted(Path("configs/runs").glob("*.yaml")):
        if (is_remote and not "rem" in config_file.stem.split("_")[0]) \
            or (not is_remote and "rem" in config_file.stem.split("_")[0]):
            continue
        # denote remote runs eg 1rem
        num = int(config_file.stem.split("_")[0].removesuffix("rem"))
        if start <= num <= end:
            configs.append(str(config_file))

    main(configs)
