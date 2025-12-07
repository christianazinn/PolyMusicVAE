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
    """Positional encoding for transformer sequences."""

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
        """x: (seq_len, batch_size, d_model)"""
        x = x + self.pe[: x.size(0), :].clone()
        return self.dropout(x)


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm with zero-initialized output scaling.

    Replaces standard LayerNorm, conditioning the normalization on latent z.
    adaLN(x, z) = γ(z) * LayerNorm(x) + β(z)

    The α parameter gates the residual connection and is zero-initialized
    for stable training.
    """

    def __init__(self, d_model, latent_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Project latent to scale, shift, and residual gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(latent_dim, 3 * d_model)
        )
        # Zero-initialize the final linear layer for stability
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, z):
        """
        x: (seq_len, batch, d_model) or (batch, seq_len, d_model)
        z: (batch, latent_dim)
        Returns: normalized x, alpha for residual gating
        """
        # Get modulation parameters from z
        shift, scale, alpha = self.adaLN_modulation(z).chunk(3, dim=-1)

        # Handle both seq-first and batch-first formats
        if x.dim() == 3 and x.shape[1] != z.shape[0]:
            # x is (seq_len, batch, d_model), need to broadcast
            shift = shift.unsqueeze(0)  # (1, batch, d_model)
            scale = scale.unsqueeze(0)
            alpha = alpha.unsqueeze(0)
        elif x.dim() == 3 and x.shape[0] == z.shape[0]:
            # x is (batch, seq_len, d_model)
            shift = shift.unsqueeze(1)  # (batch, 1, d_model)
            scale = scale.unsqueeze(1)
            alpha = alpha.unsqueeze(1)

        x_norm = self.norm(x)
        x_mod = x_norm * (1 + scale) + shift
        return x_mod, alpha


class AdaLNTransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with adaLN-Zero conditioning.

    Replaces standard LayerNorm with adaptive normalization conditioned on z.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, latent_dim):
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )

        # Cross attention to memory
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # adaLN for each sub-layer
        self.adaln1 = AdaLNZero(d_model, latent_dim)
        self.adaln2 = AdaLNZero(d_model, latent_dim)
        self.adaln3 = AdaLNZero(d_model, latent_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, z, tgt_mask=None, tgt_key_padding_mask=None):
        """
        tgt: (seq_len, batch, d_model)
        memory: (mem_len, batch, d_model)
        z: (batch, latent_dim)
        """
        # Self attention with adaLN
        tgt_norm, alpha1 = self.adaln1(tgt, z)
        self_attn_out, _ = self.self_attn(
            tgt_norm,
            tgt_norm,
            tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + alpha1 * self.dropout(self_attn_out)

        # Cross attention with adaLN
        tgt_norm, alpha2 = self.adaln2(tgt, z)
        cross_attn_out, _ = self.cross_attn(tgt_norm, memory, memory)
        tgt = tgt + alpha2 * self.dropout(cross_attn_out)

        # FFN with adaLN
        tgt_norm, alpha3 = self.adaln3(tgt, z)
        ffn_out = self.ffn(tgt_norm)
        tgt = tgt + alpha3 * ffn_out

        return tgt


class AdaLNTransformerDecoder(nn.Module):
    """Stack of AdaLN decoder layers."""

    def __init__(
        self, d_model, nhead, dim_feedforward, dropout, latent_dim, num_layers
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AdaLNTransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout, latent_dim
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tgt, memory, z, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, z, tgt_mask, tgt_key_padding_mask)
        return output


class ChunkedLocalDecoderLayer(nn.Module):
    """Decoder layer with chunk-local self-attention.

    Self-attention is restricted to within-chunk only (block-diagonal mask),
    forcing global/cross-chunk information to flow through conductor embeddings.
    Cross-attention attends to per-chunk conductor embeddings.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, conductor_embeds, chunk_mask, tgt_key_padding_mask=None):
        """
        tgt: (seq_len, batch, d_model)
        conductor_embeds: (seq_len, batch, d_model) - per-position conductor embeddings
        chunk_mask: (seq_len, seq_len) - block-diagonal + causal mask
        """
        # Self attention (chunk-local only due to mask)
        tgt_norm = self.norm1(tgt)
        self_attn_out, _ = self.self_attn(
            tgt_norm,
            tgt_norm,
            tgt_norm,
            attn_mask=chunk_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout(self_attn_out)

        # Cross attention to conductor embeddings
        # Each position attends to its corresponding conductor embedding
        tgt_norm = self.norm2(tgt)
        cross_attn_out, _ = self.cross_attn(
            tgt_norm, conductor_embeds, conductor_embeds
        )
        tgt = tgt + self.dropout(cross_attn_out)

        # FFN
        tgt_norm = self.norm3(tgt)
        ffn_out = self.ffn(tgt_norm)
        tgt = tgt + ffn_out

        return tgt


class ChunkedLocalDecoder(nn.Module):
    """Hierarchical decoder with chunk-local self-attention.

    Implements the key insight from MusicVAE: break the information pathway
    so that global context MUST flow through the latent -> conductor -> decoder path.

    For a sequence of length T split into num_chunks chunks:
    1. Conductor maps z -> num_chunks embeddings (one per chunk)
    2. Each token can only self-attend within its chunk (block-diagonal mask)
    3. Each token cross-attends to its chunk's conductor embedding

    This forces the model to encode global structure in z, since the decoder
    cannot directly "see" tokens outside its local chunk.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        latent_dim,
        num_layers,
        chunk_size=16,
        max_seq_len=256,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.max_chunks = (max_seq_len + chunk_size - 1) // chunk_size

        # Conductor: z -> sequence of chunk embeddings
        # Using a small transformer to generate chunk embeddings autoregressively
        # (non-autoregressive also works but this is closer to MusicVAE's conductor RNN)
        self.conductor_embed = nn.Parameter(
            torch.randn(self.max_chunks, d_model) * 0.02
        )
        self.conductor_proj = nn.Linear(latent_dim, d_model)
        self.conductor_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=False,
            ),
            num_layers=2,  # Lightweight conductor
        )
        self.conductor_out = nn.Linear(d_model, d_model)

        # Main decoder layers with chunk-local attention
        self.layers = nn.ModuleList(
            [
                ChunkedLocalDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        # Cache for chunk masks
        self._chunk_mask_cache = {}

    def create_chunk_causal_mask(self, seq_len, device):
        """Create block-diagonal causal mask for chunk-local attention.

        Tokens can only attend to:
        1. Tokens within the same chunk
        2. Tokens that come before them (causal)

        Returns mask where True = blocked, False = allowed (PyTorch convention)
        """
        cache_key = (seq_len, device)
        if cache_key in self._chunk_mask_cache:
            return self._chunk_mask_cache[cache_key]

        # Start with all blocked
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        # Unblock within-chunk attention
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            mask[chunk_start:chunk_end, chunk_start:chunk_end] = False

        # Apply causal masking (block future tokens)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )
        mask = mask | causal

        self._chunk_mask_cache[cache_key] = mask
        return mask

    def get_conductor_embeddings(self, z, seq_len):
        """Generate per-chunk conductor embeddings from latent z.

        z: (batch, latent_dim)
        Returns: (seq_len, batch, d_model) - embeddings broadcast to each position
        """
        batch_size = z.shape[0]
        device = z.device

        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        # Project z and add to learnable chunk position embeddings
        z_proj = self.conductor_proj(z)  # (batch, d_model)
        chunk_embeds = self.conductor_embed[:num_chunks].unsqueeze(
            1
        )  # (num_chunks, 1, d_model)
        chunk_embeds = chunk_embeds.expand(
            -1, batch_size, -1
        )  # (num_chunks, batch, d_model)
        chunk_embeds = chunk_embeds + z_proj.unsqueeze(0)  # Add z to each chunk embed

        # Pass through conductor transformer
        chunk_embeds = self.conductor_layers(
            chunk_embeds
        )  # (num_chunks, batch, d_model)
        chunk_embeds = self.conductor_out(chunk_embeds)  # (num_chunks, batch, d_model)

        # Expand to per-position: each position gets its chunk's embedding
        position_embeds = []
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start
            # Repeat this chunk's embedding for all positions in the chunk
            expanded = chunk_embeds[chunk_idx : chunk_idx + 1].expand(chunk_len, -1, -1)
            position_embeds.append(expanded)

        position_embeds = torch.cat(position_embeds, dim=0)  # (seq_len, batch, d_model)
        return position_embeds

    def forward(self, tgt, z, tgt_key_padding_mask=None):
        """
        tgt: (seq_len, batch, d_model)
        z: (batch, latent_dim)
        """
        seq_len = tgt.shape[0]
        device = tgt.device

        # Get chunk-local causal mask
        chunk_mask = self.create_chunk_causal_mask(seq_len, device)

        # Get per-position conductor embeddings
        conductor_embeds = self.get_conductor_embeddings(z, seq_len)

        # Add conductor embeddings directly to input (also available via cross-attention)
        output = tgt + conductor_embeds

        # Pass through decoder layers
        for layer in self.layers:
            output = layer(output, conductor_embeds, chunk_mask, tgt_key_padding_mask)

        return output


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
        free_bits: float | None = None,
        lr_schedule: str = "cosine",
        warmup_steps: int = 4000,
        scheduled_sampling: bool = False,
        scheduled_sampling_rate: int = 2000,
        # New parameters for posterior collapse prevention
        use_adaln: bool = True,
        num_memory_tokens: int = 8,
        input_dropout: float = 0.0,
        cyclical_annealing: bool = False,
        n_cycles: int = 4,
        # Hierarchical/chunked decoder (MusicVAE-style)
        use_chunked_decoder: bool = False,
        chunk_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model parameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_warmup_steps = beta_warmup_steps
        self.beta_decay_rate = beta_decay_rate
        self.free_bits = free_bits
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.scheduled_sampling = scheduled_sampling
        self.scheduled_sampling_rate = scheduled_sampling_rate

        # Posterior collapse prevention
        self.use_adaln = use_adaln
        self.num_memory_tokens = num_memory_tokens
        self.input_dropout = input_dropout
        self.cyclical_annealing = cyclical_annealing
        self.n_cycles = n_cycles
        self.use_chunked_decoder = use_chunked_decoder
        self.chunk_size = chunk_size

        # tracking
        self._val_latent_means = []
        self._val_latent_vars = []

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder
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

        # Multi-token memory: expand latent to multiple memory tokens
        self.latent_to_memory = nn.Linear(latent_dim, num_memory_tokens * d_model)

        # Also keep direct latent projection for adaLN
        self.latent_to_decoder = nn.Linear(latent_dim, d_model)

        # Decoder - choose architecture based on flags
        # Priority: chunked > adaln > standard
        if use_chunked_decoder:
            self.decoder = ChunkedLocalDecoder(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                latent_dim=latent_dim,
                num_layers=n_decoder_layers,
                chunk_size=chunk_size,
                max_seq_len=max_seq_len,
            )
        elif use_adaln:
            self.decoder = AdaLNTransformerDecoder(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                latent_dim=latent_dim,
                num_layers=n_decoder_layers,
            )
        else:
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

        # Training step counter for beta scheduling
        self.training_step_count = 0

        # Will be set during training
        self._total_training_steps = None

    def _init_weights(self):
        """Initialize model weights."""
        for name, module in self.named_modules():
            # Skip adaLN modulation layers (already zero-initialized)
            if "adaLN_modulation" in name:
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def get_beta(self):
        """Get current beta value for KL loss weighting."""
        if self.cyclical_annealing and self._total_training_steps is not None:
            # Cyclical annealing: repeat beta warmup n_cycles times
            steps_per_cycle = self._total_training_steps // self.n_cycles
            if steps_per_cycle == 0:
                steps_per_cycle = 1
            cycle_position = self.training_step_count % steps_per_cycle
            progress = cycle_position / steps_per_cycle
            return self.beta_start + (self.beta_end - self.beta_start) * progress
        elif self.beta_warmup_steps == 0:
            # Exp annealing
            return self.beta_end * (1 - self.beta_decay_rate**self.training_step_count)
        else:
            # Linear warmup
            progress = min(1.0, self.training_step_count / self.beta_warmup_steps)
            return self.beta_start + (self.beta_end - self.beta_start) * progress

    def get_sampling_probability(self):
        """Get probability of using ground truth (inverse sigmoid schedule)."""
        if not self.scheduled_sampling or self.scheduled_sampling_rate == 0:
            return 1.0

        k = self.scheduled_sampling_rate
        i = self.training_step_count
        epsilon = k / (k + math.exp(i / k))
        return epsilon

    def create_padding_mask(self, sequences):
        """Create padding mask for sequences."""
        return sequences == self.pad_id

    def create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(self.device)

    def apply_input_dropout(self, tokens):
        """Apply word dropout to decoder inputs during training.

        Randomly replaces tokens with pad_id to force reliance on latent.
        """
        if not self.training or self.input_dropout <= 0:
            return tokens

        # Create dropout mask (don't drop BOS token at position 0)
        mask = torch.rand_like(tokens.float()) < self.input_dropout
        mask[:, 0] = False  # Keep BOS

        # Replace with pad_id (or could use a special <unk> token)
        dropped = tokens.clone()
        dropped[mask] = self.pad_id
        return dropped

    def encode(self, sequences, lengths=None):
        """Encode sequences to latent space."""
        batch_size, seq_len = sequences.shape

        # Create padding mask
        padding_mask = self.create_padding_mask(sequences)

        # Embed and add positional encoding
        embedded = self.token_embedding(sequences).transpose(0, 1)
        embedded = self.pos_encoding(embedded)

        # Encode
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)

        # Pool to fixed-size representation
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

        # Project to latent space
        latent_params = self.encoder_to_latent(pooled)
        mu, logvar = latent_params.chunk(2, dim=-1)

        std = torch.exp(0.5 * logvar)
        dist = Normal(mu, std)

        return dist, encoded

    def create_memory(self, z):
        """Create multi-token memory from latent vector.

        Expands z into num_memory_tokens separate memory vectors for richer
        cross-attention conditioning.
        """
        batch_size = z.shape[0]
        # Project to multiple memory tokens
        memory = self.latent_to_memory(z)  # (batch, num_tokens * d_model)
        memory = memory.view(batch_size, self.num_memory_tokens, self.d_model)
        memory = memory.transpose(0, 1)  # (num_tokens, batch, d_model)
        return memory

    def decode_teacher_forcing(self, z, target_sequences):
        """Decode with teacher forcing for training."""
        batch_size, seq_len = target_sequences.shape

        # Prepare decoder input (shift target sequences by one)
        decoder_input = torch.cat(
            [
                torch.full((batch_size, 1), self.bos_id, device=self.device),
                target_sequences[:, :-1],
            ],
            dim=1,
        )

        # Apply input dropout for decoder weakening
        decoder_input = self.apply_input_dropout(decoder_input)

        # Create masks
        padding_mask = self.create_padding_mask(decoder_input)

        # Embed decoder input
        embedded = self.token_embedding(decoder_input).transpose(0, 1)
        embedded = self.pos_encoding(embedded)

        # Decode based on architecture
        if self.use_chunked_decoder:
            # Chunked decoder handles its own masking and latent conditioning
            decoded = self.decoder(embedded, z, tgt_key_padding_mask=padding_mask)
        elif self.use_adaln:
            # adaLN decoder needs memory, z, and causal mask
            memory = self.create_memory(z)
            causal_mask = self.create_causal_mask(seq_len)
            decoded = self.decoder(
                embedded,
                memory,
                z,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
            )
        else:
            # Standard decoder
            memory = self.create_memory(z)
            causal_mask = self.create_causal_mask(seq_len)
            decoded = self.decoder(
                embedded,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
            )

        # Project to vocabulary
        logits = self.output_projection(decoded).transpose(0, 1)

        return logits

    def decode_scheduled_sampling(self, z, target_sequences):
        """Decode with scheduled sampling - gradually mix ground truth with predictions."""
        batch_size, seq_len = target_sequences.shape

        sampling_prob = self.get_sampling_probability()

        use_ground_truth = (
            torch.rand(batch_size, seq_len, device=self.device) < sampling_prob
        )

        decoder_input_tokens = torch.full(
            (batch_size, seq_len), self.pad_id, device=self.device
        )
        decoder_input_tokens[:, 0] = target_sequences[:, 0]

        # For non-chunked decoders, create memory once
        if not self.use_chunked_decoder:
            memory = self.create_memory(z)

        for t in range(1, seq_len):
            current_input = torch.cat(
                [
                    torch.full((batch_size, 1), self.bos_id, device=self.device),
                    decoder_input_tokens[:, :t],
                ],
                dim=1,
            )

            embedded = self.token_embedding(current_input).transpose(0, 1)
            embedded = self.pos_encoding(embedded)

            curr_len = current_input.shape[1]

            with torch.no_grad():
                if self.use_chunked_decoder:
                    decoded = self.decoder(embedded, z)
                elif self.use_adaln:
                    causal_mask = self.create_causal_mask(curr_len)
                    decoded = self.decoder(embedded, memory, z, tgt_mask=causal_mask)
                else:
                    causal_mask = self.create_causal_mask(curr_len)
                    decoded = self.decoder(embedded, memory, tgt_mask=causal_mask)

                logits_t = self.output_projection(decoded[-1])
                probs = F.softmax(logits_t, dim=-1)
                predicted_token = torch.multinomial(probs, 1).squeeze(-1)

            ground_truth_token = target_sequences[:, t]
            next_token = torch.where(
                use_ground_truth[:, t], ground_truth_token, predicted_token
            )
            decoder_input_tokens[:, t] = next_token

        # Final forward pass
        decoder_input = torch.cat(
            [
                torch.full((batch_size, 1), self.bos_id, device=self.device),
                decoder_input_tokens[:, :-1],
            ],
            dim=1,
        )

        decoder_input = self.apply_input_dropout(decoder_input)

        padding_mask = self.create_padding_mask(decoder_input)

        embedded = self.token_embedding(decoder_input).transpose(0, 1)
        embedded = self.pos_encoding(embedded)

        if self.use_chunked_decoder:
            decoded = self.decoder(embedded, z, tgt_key_padding_mask=padding_mask)
        elif self.use_adaln:
            causal_mask = self.create_causal_mask(seq_len)
            decoded = self.decoder(
                embedded,
                memory,
                z,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
            )
        else:
            causal_mask = self.create_causal_mask(seq_len)
            decoded = self.decoder(
                embedded,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
            )

        logits = self.output_projection(decoded).transpose(0, 1)

        return logits

    def decode_autoregressive(self, z, max_length=None, temperature=1.0):
        """Decode autoregressively for inference."""
        if max_length is None:
            max_length = self.max_seq_len

        batch_size = z.shape[0]

        generated = torch.full((batch_size, 1), self.bos_id, device=self.device)

        # For non-chunked decoders, create memory once
        if not self.use_chunked_decoder:
            memory = self.create_memory(z)

        for step in range(max_length):
            seq_len = generated.shape[1]

            embedded = self.token_embedding(generated).transpose(0, 1)
            embedded = self.pos_encoding(embedded)

            if self.use_chunked_decoder:
                decoded = self.decoder(embedded, z)
            elif self.use_adaln:
                causal_mask = self.create_causal_mask(seq_len)
                decoded = self.decoder(embedded, memory, z, tgt_mask=causal_mask)
            else:
                causal_mask = self.create_causal_mask(seq_len)
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
        """Generate sequences from random latent vectors."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            generated = self.decode_autoregressive(z, max_length, temperature)
        return generated

    def interpolate(self, seq1, seq2, num_steps=10, do_spherical=False):
        """Interpolate between two sequences in latent space."""
        self.eval()
        with torch.no_grad():
            z1 = self.encode(seq1)[0].mean
            z2 = self.encode(seq2)[0].mean

            if do_spherical:
                # Spherical interpolation (slerp)
                z1_norm = F.normalize(z1, dim=-1)
                z2_norm = F.normalize(z2, dim=-1)
                omega = torch.acos(
                    (z1_norm * z2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
                )

                alphas = torch.linspace(0, 1, num_steps, device=self.device)
                interpolated = []
                for alpha in alphas:
                    if omega.abs() < 1e-6:
                        z_interp = (1 - alpha) * z1 + alpha * z2
                    else:
                        z_interp = (
                            torch.sin((1 - alpha) * omega) * z1
                            + torch.sin(alpha * omega) * z2
                        ) / torch.sin(omega)
                    generated = self.decode_autoregressive(z_interp)
                    interpolated.append(generated)
            else:
                alphas = torch.linspace(0, 1, num_steps, device=self.device)
                interpolated = []
                for alpha in alphas:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    generated = self.decode_autoregressive(z_interp)
                    interpolated.append(generated)

        return interpolated

    def forward(self, sequences, target_sequences=None, lengths=None, sample=True):
        """Full forward pass."""
        latent_dist, encoded = self.encode(sequences, lengths)

        if sample:
            z = latent_dist.rsample()
        else:
            z = latent_dist.mean

        if target_sequences is not None:
            if self.scheduled_sampling and self.training:
                logits = self.decode_scheduled_sampling(z, target_sequences)
            else:
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
        """Compute VAE loss (reconstruction + KL divergence)."""
        if beta is None:
            beta = self.get_beta()

        # Reconstruction loss
        reconstruction_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_id,
            reduction="mean",
        )

        # KL divergence loss
        prior = Normal(
            torch.zeros_like(latent_dist.mean), torch.ones_like(latent_dist.stddev)
        )
        kl_per_dim = torch.distributions.kl_divergence(latent_dist, prior)

        if self.free_bits is not None:
            # Free bits per dimension
            free_bits_per_dim = self.free_bits / self.latent_dim
            # Clamp KL per dimension, then sum
            kl_loss = (
                torch.clamp(kl_per_dim - free_bits_per_dim, min=0).sum(dim=-1).mean()
            )
        else:
            kl_loss = kl_per_dim.sum(dim=-1).mean()

        total_loss = reconstruction_loss + beta * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def training_step(self, batch, batch_idx):
        """Training step."""
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
        if self.scheduled_sampling:
            self.log(
                "trainer/sampling_prob", self.get_sampling_probability(), on_step=True
            )

        self.training_step_count += 1

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        sequences = batch["sequences"]
        target_sequences = batch.get("target_sequences", sequences)
        lengths = batch.get("lengths", None)

        outputs = self.forward(sequences, target_sequences, lengths, sample=False)

        total_loss, recon_loss, kl_loss = self.compute_loss(
            outputs["logits"], target_sequences, outputs["latent_dist"]
        )

        self.log("val/total_loss", total_loss, on_epoch=True, sync_dist=True)
        self.log("val/reconstruction_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("val/kl_loss", kl_loss, on_epoch=True, sync_dist=True)

        latent_mean = outputs["latent_dist"].mean
        latent_std = outputs["latent_dist"].stddev

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

        # Per-position cross entropy
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

        if len(self._val_latent_means) < 100:
            self._val_latent_means.append(outputs["latent_dist"].mean.detach().clone())
            self._val_latent_vars.append(
                outputs["latent_dist"].variance.detach().clone()
            )

        return total_loss

    def test_step(self, batch, batch_idx):
        """Test step."""
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

    def on_train_start(self):
        """Called at the start of training to compute total steps for cyclical annealing."""
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            self._total_training_steps = self.trainer.max_steps
        else:
            # Estimate from epochs
            steps_per_epoch = len(self.trainer.train_dataloader)
            self._total_training_steps = steps_per_epoch * self.trainer.max_epochs

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.lr_schedule == "cosine":
            # Use max_steps if available, otherwise estimate
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                total_steps = self.trainer.max_steps
            else:
                # Estimate - this will be approximate
                total_steps = self.trainer.max_epochs * 1000  # fallback estimate

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # This was the bug - needs to be step for step-level decay
                    "frequency": 1,
                },
            }
        elif self.lr_schedule == "warmup_cosine":
            # Get total steps
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                total_steps = self.trainer.max_steps
            else:
                total_steps = self.trainer.max_epochs * 1000

            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / max(1, self.warmup_steps)
                else:
                    progress = (step - self.warmup_steps) / max(
                        1, total_steps - self.warmup_steps
                    )
                    return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
        """Compute and log latent similarity metrics at the end of validation."""
        if not hasattr(self, "_val_latent_means") or len(self._val_latent_means) == 0:
            return

        all_means = torch.cat(self._val_latent_means, dim=0)

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
    def load_id(cls, run_id: int, checkpoints_dir: str = "checkpoints"):
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
    """Get standard callbacks for training."""
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
