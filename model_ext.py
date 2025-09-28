import torch
from .model import MusicVAE


class MusicVAEExt(MusicVAE):
    def generate(self, batch_size=1, max_length=None, temperature=1.0):
        """Generate sequences from random latent vectors."""
        self.eval()
        with torch.no_grad():
            # Sample random latent vectors
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Generate sequences
            generated = self.decode_autoregressive(z, max_length, temperature)
        return generated
    
    def interpolate(self, seq1, seq2, num_steps=10):
        """Interpolate between two sequences in latent space."""
        self.eval()
        with torch.no_grad():
            # Encode both sequences
            z1 = self.encode(seq1)[0].mean
            z2 = self.encode(seq2)[0].mean
            
            # Interpolate in latent space
            alphas = torch.linspace(0, 1, num_steps, device=self.device)
            interpolated = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                generated = self.decode_autoregressive(z_interp.unsqueeze(0))
                interpolated.append(generated)
        
        return interpolated