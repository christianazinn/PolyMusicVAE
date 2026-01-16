import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import math
from pathlib import Path

torch.set_float32_matmul_precision("medium")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).contiguous()
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :].clone()
        return self.dropout(x)


class MusicVAE(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        latent_dim: int = 512,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        beta_warmup_steps: int = 10000,
        beta_decay_rate: float = 0.99999,
        free_bits: int | None = None,
        lr_schedule: str = "cosine",
        warmup_steps: int = 4000,
        input_dropout: float = 0.0,
        steps_per_epoch_est: int = 341523,  # this number is for data_nb_1b_combined
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        # for now...
        assert self.max_seq_len == 256
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_warmup_steps = beta_warmup_steps
        self.beta_decay_rate = beta_decay_rate  # only used if beta_warmup_steps=0
        self.free_bits = free_bits
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.input_dropout = input_dropout
        self.steps_per_epoch_est = steps_per_epoch_est

        # tracking
        self._val_latent_means = []
        self._val_latent_vars = []

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        self.encoder_to_latent = nn.Linear(d_model, latent_dim * 2)  # mu and logvar
        self.latent_to_decoder = nn.Linear(latent_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()

        self.training_step_count = 0

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def get_beta(self):
        if self.beta_warmup_steps == 0:
            # exp annealing
            return self.beta_end * (1 - self.beta_decay_rate**self.training_step_count)

        progress = min(1.0, self.training_step_count / self.beta_warmup_steps)
        return self.beta_start + (self.beta_end - self.beta_start) * progress

    def create_padding_mask(self, sequences):
        return sequences == self.pad_id

    def create_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(self.device)

    def apply_input_dropout(self, tokens):
        if not self.training or self.input_dropout <= 0:
            return tokens

        mask = torch.rand_like(tokens.float()) < self.input_dropout
        mask[:, 0] = False  # keep BOS
        dropped = tokens.clone()
        dropped[mask] = self.pad_id
        return dropped

    def encode(self, sequences, lengths=None):
        padding_mask = self.create_padding_mask(sequences)

        embedded = self.token_embedding(sequences).transpose(0, 1)
        embedded = self.pos_encoding(embedded)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)

        if lengths is not None:
            pooled = []
            for i, length in enumerate(lengths):
                seq_repr = encoded[:length, i].mean(dim=0)
                pooled.append(seq_repr)
            pooled = torch.stack(pooled)
        else:
            mask = padding_mask.transpose(0, 1).unsqueeze(-1)
            masked_encoded = encoded.masked_fill(mask, 0)
            valid_lengths = (
                (~padding_mask).sum(dim=1, keepdim=True).float().transpose(0, 1)
            )
            pooled = masked_encoded.sum(dim=0) / valid_lengths.squeeze(0)

        latent_params = self.encoder_to_latent(pooled)
        mu, logvar = latent_params.chunk(2, dim=-1)

        std = torch.exp(0.5 * logvar)
        dist = Normal(mu, std)

        return dist, encoded

    def decode_teacher_forcing(self, z, target_sequences):
        batch_size, seq_len = target_sequences.shape

        decoder_input = torch.cat(
            [
                torch.full((batch_size, 1), self.bos_id, device=self.device),
                target_sequences[:, :-1],
            ],
            dim=1,
        )

        decoder_input = self.apply_input_dropout(decoder_input)

        padding_mask = self.create_padding_mask(decoder_input)
        causal_mask = self.create_causal_mask(seq_len)

        embedded = self.token_embedding(decoder_input).transpose(0, 1)
        embedded = self.pos_encoding(embedded)

        memory = self.latent_to_decoder(z).unsqueeze(0)

        decoded = self.decoder(
            embedded, memory, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask
        )

        logits = self.output_projection(decoded).transpose(0, 1)

        return logits

    def decode_autoregressive(self, z, max_length=None, temperature=1.0):
        if max_length is None:
            max_length = self.max_seq_len

        batch_size = z.shape[0]

        generated = torch.full((batch_size, 1), self.bos_id, device=self.device)

        memory = self.latent_to_decoder(z).unsqueeze(0)

        for step in range(max_length):
            seq_len = generated.shape[1]
            causal_mask = self.create_causal_mask(seq_len)

            embedded = self.token_embedding(generated).transpose(0, 1)
            embedded = self.pos_encoding(embedded)

            decoded = self.decoder(embedded, memory, tgt_mask=causal_mask)

            next_token_logits = self.output_projection(decoded[-1]) / temperature

            if temperature > 0:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)
            else:
                next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tokens], dim=1)

            if (next_tokens.squeeze(-1) == self.eos_id).all():
                break

        return generated

    def generate(self, batch_size=1, max_length=None, temperature=1.0):
        self.eval()
        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            generated = self.decode_autoregressive(z, max_length, temperature)
        return generated

    def interpolate(self, seq1, seq2, num_steps=10, do_spherical=False):
        self.eval()
        with torch.no_grad():
            z1 = self.encode(seq1)[0].mean
            z2 = self.encode(seq2)[0].mean

            # TODO: is this right? it sucks
            if do_spherical:
                alphas = torch.linspace(0, math.pi / 2, num_steps, device=self.device)
                interpolated = []
                for alpha in alphas:
                    z_interp = (
                        torch.cos(alpha) * z1 + torch.sin(alpha) * z2
                    ) / math.sqrt(2)
                    generated = self.decode_autoregressive(z_interp)  # .unsqueeze(0))
                    interpolated.append(generated)
            else:
                alphas = torch.linspace(0, 1, num_steps, device=self.device)
            interpolated = []

            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                generated = self.decode_autoregressive(z_interp)  # .unsqueeze(0))
                interpolated.append(generated)

        return interpolated

    def forward(self, sequences, target_sequences=None, lengths=None, sample=True):
        latent_dist, encoded = self.encode(sequences, lengths)

        z = latent_dist.rsample() if sample else latent_dist.mean

        if target_sequences is not None:
            logits = self.decode_teacher_forcing(z, target_sequences)
            return {
                "logits": logits,
                "latent_dist": latent_dist,
                "z": z,
                "encoded": encoded,
            }
        else:
            generated = self.decode_autoregressive(z)
            return {
                "generated": generated,
                "latent_dist": latent_dist,
                "z": z,
                "encoded": encoded,
            }

    def compute_loss(self, logits, targets, latent_dist, beta=None):
        if beta is None:
            beta = self.get_beta()

        reconstruction_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_id,
            reduction="mean",
        )

        prior = Normal(
            torch.zeros_like(latent_dist.mean), torch.ones_like(latent_dist.stddev)
        )

        kl_per_example = (
            torch.distributions.kl_divergence(latent_dist, prior).sum(dim=-1).mean()
        )
        if hasattr(self, "log"):
            self.log("debug/kl_total_per_example", kl_per_example.item(), on_step=True)
        kl_per_dim = torch.distributions.kl_divergence(latent_dist, prior).mean()

        kl_before_free_bits = kl_per_dim.item()

        if self.free_bits is not None:
            free_bits = self.free_bits / self.latent_dim
            kl_loss = torch.max(
                kl_per_dim - free_bits, torch.zeros_like(kl_per_dim)
            ).mean()

            # dude actually what the fuck
            kl_after_free_bits = kl_loss.item()
            free_bits_threshold = free_bits
            reduction_pct = 100 * (
                1 - kl_after_free_bits / (kl_before_free_bits + 1e-8)
            )

            if hasattr(self, "log"):
                self.log("debug/kl_before_free_bits", kl_before_free_bits, on_step=True)
                self.log("debug/kl_after_free_bits", kl_after_free_bits, on_step=True)
                self.log("debug/free_bits_threshold", free_bits_threshold, on_step=True)
                self.log("debug/kl_reduction_pct", reduction_pct, on_step=True)
        else:
            kl_loss = kl_per_dim

        total_loss = reconstruction_loss + beta * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def training_step(self, batch, batch_idx):
        sequences = batch["sequences"]
        target_sequences = batch.get("target_sequences", sequences)
        lengths = batch.get("lengths", None)

        outputs = self.forward(sequences, target_sequences, lengths, sample=True)

        total_loss, recon_loss, kl_loss = self.compute_loss(
            outputs["logits"], target_sequences, outputs["latent_dist"]
        )

        beta = self.get_beta()
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)
        self.log("train/reconstruction_loss", recon_loss, on_step=True, on_epoch=True)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True)
        self.log("trainer/beta", beta, on_step=True)

        self.training_step_count += 1

        return total_loss

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequences"]
        target_sequences = batch.get("target_sequences", sequences)
        lengths = batch.get("lengths", None)

        outputs = self.forward(sequences, target_sequences, lengths, sample=False)

        total_loss, recon_loss, kl_loss = self.compute_loss(
            outputs["logits"], target_sequences, outputs["latent_dist"]
        )

        # primary val metrics
        self.log("val/total_loss", total_loss, on_epoch=True, sync_dist=True)
        self.log("val/reconstruction_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("val/kl_loss", kl_loss, on_epoch=True, sync_dist=True)

        # compare reconstruction with real latent vs zero latent
        # (make sure latents are used)
        z_zero = torch.zeros_like(outputs["z"])
        logits_zero = self.decode_teacher_forcing(z_zero, target_sequences)
        loss_zero = F.cross_entropy(
            logits_zero.reshape(-1, self.vocab_size),
            target_sequences.reshape(-1),
            ignore_index=self.pad_id,
            reduction="mean",
        )

        self.log("val/loss_with_latent", recon_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss_with_zero_latent", loss_zero, on_epoch=True, sync_dist=True)
        self.log(
            "val/latent_benefit_to_reconstruction",
            loss_zero - recon_loss,
            on_epoch=True,
            sync_dist=True,
        )

        # compare reconstruction with real latent vs boosted latent
        z_boosted = outputs["z"] * 10.0
        logits_boosted = self.decode_teacher_forcing(z_boosted, target_sequences)
        loss_boosted = F.cross_entropy(
            logits_boosted.reshape(-1, self.vocab_size),
            target_sequences.reshape(-1),
            ignore_index=self.pad_id,
            reduction="mean",
        )

        self.log(
            "val/loss_with_boosted_latent", loss_boosted, on_epoch=True, sync_dist=True
        )

        latent_mean = outputs["latent_dist"].mean
        latent_std = outputs["latent_dist"].stddev

        # latent stats
        self.log(
            "val/latent_mean_abs_mean",
            latent_mean.abs().mean(),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/latent_std_mean", latent_std.mean(), on_epoch=True, sync_dist=True
        )
        self.log(
            "val/latent_mean_abs_max",
            latent_mean.abs().max(),
            on_epoch=True,
            sync_dist=True,
        )

        logits = outputs["logits"]
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_sequences.reshape(-1),
            ignore_index=self.pad_id,
            reduction="none",
        ).reshape(target_sequences.shape)

        mask = target_sequences != self.pad_id
        masked_loss = per_token_loss * mask
        position_loss = masked_loss.sum(dim=0) / mask.sum(dim=0).clamp(min=1)

        # positional losses
        seq_len = position_loss.shape[0]
        if seq_len > 50:
            self.log(
                "val/loss_pos_50", position_loss[49], on_epoch=True, sync_dist=True
            )
        if seq_len > 100:
            self.log(
                "val/loss_pos_100", position_loss[99], on_epoch=True, sync_dist=True
            )
        if seq_len > 150:
            self.log(
                "val/loss_pos_150", position_loss[149], on_epoch=True, sync_dist=True
            )
        if seq_len > 200:
            self.log(
                "val/loss_pos_200", position_loss[199], on_epoch=True, sync_dist=True
            )
        if seq_len > 250:
            self.log(
                "val/loss_pos_250", position_loss[199], on_epoch=True, sync_dist=True
            )

        # store latent means for similarity analysis
        if len(self._val_latent_means) < 100:
            self._val_latent_means.append(outputs["latent_dist"].mean.detach().clone())
            self._val_latent_vars.append(
                outputs["latent_dist"].variance.detach().clone()
            )

        return total_loss

    def test_step(self, batch, batch_idx):
        sequences = batch["sequences"]
        target_sequences = batch.get("target_sequences", sequences)
        lengths = batch.get("lengths", None)

        outputs = self.forward(sequences, target_sequences, lengths, sample=False)

        total_loss, recon_loss, kl_loss = self.compute_loss(
            outputs["logits"], target_sequences, outputs["latent_dist"]
        )

        self.log("test/total_loss", total_loss)
        self.log("test/reconstruction_loss", recon_loss)
        self.log("test/kl_loss", kl_loss)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.lr_schedule == "cosine":
            # TODO: dumb hack, hardcoded lol 72162 or 341523
            est_num_steps = self.trainer.max_epochs * self.steps_per_epoch_est
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=est_num_steps,
                eta_min=self.learning_rate * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def on_validation_epoch_end(self):
        if not hasattr(self, "_val_latent_means") or len(self._val_latent_means) == 0:
            return

        all_means = torch.cat(self._val_latent_means, dim=0)

        # determine pairwise similarity between latents
        normalized = F.normalize(all_means, dim=1)
        similarity_matrix = torch.mm(normalized, normalized.t())

        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
        similarities = similarity_matrix[mask]

        self.log("val_sim/mean", similarities.mean(), sync_dist=True)
        self.log("val_sim/std", similarities.std(), sync_dist=True)
        self.log("val_sim/min", similarities.min(), sync_dist=True)
        self.log("val_sim/max", similarities.max(), sync_dist=True)

        total_pairs = len(similarities)
        pct_above_90 = (similarities > 0.9).sum().float() / total_pairs * 100
        pct_above_75 = (similarities > 0.75).sum().float() / total_pairs * 100

        self.log("val_sim/above_0.9_pct", pct_above_90, sync_dist=True)
        self.log("val_sim/above_0.75_pct", pct_above_75, sync_dist=True)

        vars = torch.cat(self._val_latent_vars, dim=0)
        self.log(
            "val_latent/active_units_0.1",
            (vars.mean(0) > 0.1).sum().to(torch.float32),
            sync_dist=True,
        )
        self.log(
            "val_latent/active_units_0.01",
            (vars.mean(0) > 0.01).sum().to(torch.float32),
            sync_dist=True,
        )

        self._val_latent_means = []
        self._val_latent_vars = []

    @classmethod
    def load_id(cls, run_id: int | str, checkpoints_dir: str = "checkpoints"):
        """Load checkpoint by run ID number."""
        checkpoint_path = Path(checkpoints_dir)
        matching = list(checkpoint_path.glob(f"{run_id}_*"))

        if not matching:
            raise FileNotFoundError(f"No checkpoint folder found for run_id {run_id}")
        if len(matching) > 1:
            raise ValueError(f"Multiple folders found for run_id {run_id}: {matching}")

        ckpt_file = matching[0] / "last.ckpt"
        return cls.load_from_checkpoint(str(ckpt_file))


def get_callbacks():
    callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            dirpath="checkpoints/",
            filename="music-vae-{epoch:02d}-{val/total_loss:.2f}",
            save_top_k=5,
            mode="min",
            save_last=True,
        ),
        EarlyStopping(monitor="val/total_loss", patience=10, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    return callbacks
