"""
Bolt-GAN Trainer
Training loop with  KL-divergence feedback mechanism

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import stats
from tqdm import tqdm
import os

from .generator import BoltGANGenerator
from .discriminator import BoltGANDiscriminator


class BoltGANTrainer:
    """
    Bolt-GAN Training with Feedback Mechanism (Corrected)
    
    Implements Algorithm 1 from the paper with proper bolt feedback every
    feedback_interval epochs.
    
    Args:
        seq_len: Sequence length
        feature_dim: Number of features
        latent_dim: Latent dimension (default: 100)
        lstm_hidden: LSTM hidden size (default: 308)
        lstm_layers: Number of LSTM layers (default: 3)
        dropout: Dropout rate (default: 0.3)
        lr: Learning rate (default: 0.0002)
        beta1: Adam beta1 (default: 0.5)
        beta2: Adam beta2 (default: 0.999)
        num_epochs: Total training epochs (default: 2500)
        feedback_interval: Apply bolt every N epochs (default: 500, i.e., 20%)
        feedback_epochs: Train for N epochs after bolt (default: 100)
        batch_size: Batch size (default: 32)
        device: Device to train on (default: 'cuda' if available)
    """
    
    def __init__(self, seq_len, feature_dim, latent_dim=100, 
                 lstm_hidden=308, lstm_layers=3, dropout=0.3,
                 lr=0.0002, beta1=0.5, beta2=0.999,
                 num_epochs=2500, feedback_interval=500, feedback_epochs=100,
                 batch_size=32, device=None):
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.feedback_interval = feedback_interval
        self.feedback_epochs = feedback_epochs
        self.batch_size = batch_size
        
        # Initialize models
        self.generator = BoltGANGenerator(
            latent_dim=latent_dim,
            seq_len=seq_len,
            feature_dim=feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout
        ).to(self.device)
        
        self.discriminator = BoltGANDiscriminator(
            seq_len=seq_len,
            feature_dim=feature_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout
        ).to(self.device)
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), 
                                      lr=lr, betas=(beta1, beta2))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), 
                                      lr=lr, betas=(beta1, beta2))
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Normalization parameters (will be set during training)
        self.data_mean = None
        self.data_std = None
        
        # Feedback storage
        self.use_feedback = False
        self.feedback_vector = None
        
    def compute_kl_divergence(self, p_samples, q_samples):
        """
        Compute KL divergence between two sets of samples using KDE
        
        Args:
            p_samples: Samples from distribution P [N, D]
            q_samples: Samples from distribution Q [M, D]
            
        Returns:
            kl_divergence: Scalar KL divergence value
        """
        # Flatten to 1D for each dimension and average
        p_flat = p_samples.reshape(p_samples.shape[0], -1)
        q_flat = q_samples.reshape(q_samples.shape[0], -1)
        
        kl_sum = 0.0
        valid_dims = 0
        
        for dim in range(min(p_flat.shape[1], 100)):  # Limit to 100 dims for speed
            try:
                # Estimate densities using KDE
                p_kde = stats.gaussian_kde(p_flat[:, dim])
                q_kde = stats.gaussian_kde(q_flat[:, dim])
                
                # Sample points for integration
                x_min = min(p_flat[:, dim].min(), q_flat[:, dim].min())
                x_max = max(p_flat[:, dim].max(), q_flat[:, dim].max())
                x_points = np.linspace(x_min, x_max, 100)
                
                # Evaluate densities
                p_vals = p_kde(x_points)
                q_vals = q_kde(x_points)
                
                # Add epsilon for numerical stability
                eps = 1e-10
                p_vals = p_vals + eps
                q_vals = q_vals + eps
                
                # Compute KL divergence
                kl = np.sum(p_vals * np.log(p_vals / q_vals)) * (x_max - x_min) / 100
                kl_sum += kl
                valid_dims += 1
            except:
                # Skip if KDE fails
                continue
        
        return kl_sum / max(valid_dims, 1)
    
    def apply_bolt_feedback(self, real_data_batch):
        """
        Apply bolt feedback mechanism (Equation 5 from paper)
        
        Computes: Z_new = KL(X||X̂) * Z + KL(X̂||Z) * X̂
        
        Args:
            real_data_batch: Batch of real data [batch_size, seq_len, feature_dim]
            
        Returns:
            feedback_noise: Enhanced noise [batch_size, latent_dim]
        """
        self.generator.eval()
        
        with torch.no_grad():
            batch_size = real_data_batch.size(0)
            
            # Generate samples with original noise
            z_original = torch.randn(batch_size, self.latent_dim).to(self.device)
            x_generated = self.generator(z_original)
            
            # Move to numpy for KL computation
            x_real_np = real_data_batch.cpu().numpy()
            x_gen_np = x_generated.cpu().numpy()
            z_np = z_original.cpu().numpy()
            
            # Compute KL divergences
            # KL(X||X̂): How different is generated from real
            kl_real_gen = self.compute_kl_divergence(x_real_np, x_gen_np)
            
            # KL(X̂||Z): How far has generator moved from noise
            # Note: We flatten generated data to compare with noise
            x_gen_flat = x_gen_np.reshape(batch_size, -1)
            z_expanded = z_np  # Keep noise as is for comparison
            kl_gen_noise = self.compute_kl_divergence(x_gen_flat[:, :self.latent_dim], z_expanded)
            
            # Compute feedback according to Equation 5
            # Z_new = KL(X||X̂) * Z + KL(X̂||Z) * X̂_collapsed
            # Collapse X̂ to latent dim by taking mean over time
            x_gen_collapsed = torch.tensor(x_gen_np.mean(axis=1).mean(axis=1, keepdims=True), 
                                          dtype=torch.float32, device=self.device)
            x_gen_collapsed = x_gen_collapsed.repeat(1, self.latent_dim)  # Expand to latent_dim
            
            feedback_noise = kl_real_gen * z_original + kl_gen_noise * x_gen_collapsed
            
            print(f"  Feedback KL(X||X̂)={kl_real_gen:.4f}, KL(X̂||Z)={kl_gen_noise:.4f}")
        
        self.generator.train()
        return feedback_noise
    
    def train(self, data, save_dir='./results'):
        """
        Train Bolt-GAN with feedback mechanism
        
        Args:
            data: Training data [N, seq_len, feature_dim] numpy array
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Normalize data
        data_flat = data.reshape(-1, self.feature_dim)
        self.data_mean = data_flat.mean(axis=0)
        self.data_std = data_flat.std(axis=0) + 1e-6
        
        data_normalized = (data - self.data_mean) / self.data_std
        
        # Create DataLoader
        dataset = TensorDataset(torch.FloatTensor(data_normalized))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        print("\n" + "="*70)
        print("Starting Bolt-GAN training with feedback mechanism...")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Feedback interval: Every {self.feedback_interval} epochs")
        print(f"Feedback training: {self.feedback_epochs} epochs")
        print("="*70)
        
        g_losses = []
        d_losses = []
        
        for epoch in range(self.num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            # Check if we should apply bolt feedback
            if (epoch + 1) % self.feedback_interval == 0 and epoch > 0:
                print(f"\n{'='*70}")
                print(f"⚡ BOLT FEEDBACK at epoch {epoch+1}")
                print(f"{'='*70}")
                self.use_feedback = True
                
                # Compute feedback from a sample batch
                sample_batch = next(iter(dataloader))[0].to(self.device)
                self.feedback_vector = self.apply_bolt_feedback(sample_batch)
                
                # Train for feedback_epochs with modified noise
                print(f"Training with feedback for {self.feedback_epochs} epochs...")
                for feedback_ep in range(self.feedback_epochs):
                    for batch_idx, (real_seq,) in enumerate(dataloader):
                        batch_size = real_seq.size(0)
                        real_seq = real_seq.to(self.device)
                        
                        real_labels = torch.ones(batch_size, 1).to(self.device)
                        fake_labels = torch.zeros(batch_size, 1).to(self.device)
                        
                        # Train Discriminator
                        self.d_optimizer.zero_grad()
                        real_validity = self.discriminator(real_seq)
                        d_real_loss = self.criterion(real_validity, real_labels)
                        
                        # Use feedback noise
                        if self.feedback_vector is not None:
                            z_base = torch.randn(batch_size, self.latent_dim).to(self.device)
                            z_feedback = self.feedback_vector[:batch_size] if batch_size <= self.feedback_vector.size(0) else z_base
                            z = z_base + 0.1 * z_feedback  # Mix original and feedback
                        else:
                            z = torch.randn(batch_size, self.latent_dim).to(self.device)
                        
                        fake_seq = self.generator(z).detach()
                        fake_validity = self.discriminator(fake_seq)
                        d_fake_loss = self.criterion(fake_validity, fake_labels)
                        
                        d_loss = (d_real_loss + d_fake_loss) / 2
                        d_loss.backward()
                        self.d_optimizer.step()
                        
                        # Train Generator
                        self.g_optimizer.zero_grad()
                        fake_seq = self.generator(z)
                        fake_validity = self.discriminator(fake_seq)
                        g_loss = self.criterion(fake_validity, real_labels)
                        g_loss.backward()
                        self.g_optimizer.step()
                
                self.use_feedback = False
                print(f"Feedback training complete. Resuming normal training...\n")
            
            # Normal training
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch_idx, (real_seq,) in enumerate(pbar):
                batch_size = real_seq.size(0)
                real_seq = real_seq.to(self.device)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                real_validity = self.discriminator(real_seq)
                d_real_loss = self.criterion(real_validity, real_labels)
                
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_seq = self.generator(z).detach()
                fake_validity = self.discriminator(fake_seq)
                d_fake_loss = self.criterion(fake_validity, fake_labels)
                
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_seq = self.generator(z)
                fake_validity = self.discriminator(fake_seq)
                
                g_loss = self.criterion(fake_validity, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                # Track losses
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
                pbar.set_postfix({'D_loss': f'{d_loss.item():.4f}', 
                                 'G_loss': f'{g_loss.item():.4f}'})
            
            # Average losses
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}] D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 500 == 0 or epoch == 0 or (epoch + 1) == self.num_epochs:
                checkpoint_path = os.path.join(save_dir, f'boltgan_epoch{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'g_optimizer': self.g_optimizer.state_dict(),
                    'd_optimizer': self.d_optimizer.state_dict(),
                    'data_mean': self.data_mean,
                    'data_std': self.data_std,
                    'g_losses': g_losses,
                    'd_losses': d_losses
                }, checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        print("\n✅ Training complete!")
        return g_losses, d_losses
    
    def generate(self, num_samples):
        """
        Generate synthetic samples
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples (denormalized) [num_samples, seq_len, feature_dim]
        """
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.generator(z).cpu().numpy()
        
        # Denormalize
        if self.data_mean is not None:
            samples = samples * self.data_std + self.data_mean
        
        return samples
