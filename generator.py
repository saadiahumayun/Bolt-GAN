"""
Bolt-GAN Generator
LSTM-based generator for multivariate time series generation
"""

import torch
import torch.nn as nn


class BoltGANGenerator(nn.Module):
    """
    LSTM-based Generator for Bolt-GAN
    
    Args:
        latent_dim: Dimension of noise input (e.g., 100)
        seq_len: Length of generated sequences (e.g., 144)
        feature_dim: Number of features (e.g., 29 for Energy)
        lstm_hidden: Hidden dimension for LSTM layers (default: 308)
        lstm_layers: Number of LSTM layers (default: 3)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(self, latent_dim, seq_len, feature_dim, 
                 lstm_hidden=308, lstm_layers=3, dropout=0.3):
        super(BoltGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.lstm_hidden = lstm_hidden
        
        # Input projection layer
        self.input_proj = nn.Linear(latent_dim, lstm_hidden)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output projection layer
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden, feature_dim),
            nn.Tanh()  # Normalize output to [-1, 1]
        )
        
    def forward(self, z):
        """
        Generate synthetic time series
        
        Args:
            z: Noise input [batch_size, latent_dim]
            
        Returns:
            Generated sequences [batch_size, seq_len, feature_dim]
        """
        batch_size = z.size(0)
        
        # Project noise to LSTM input dimension
        h = self.input_proj(z)  # [batch_size, lstm_hidden]
        
        # Repeat for sequence length
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, lstm_hidden]
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(h)  # [batch_size, seq_len, lstm_hidden]
        
        # Project to feature dimension
        output = self.output_proj(lstm_out)  # [batch_size, seq_len, feature_dim]
        
        return output
    
    def generate(self, num_samples, device='cpu'):
        """
        Generate synthetic samples
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on ('cpu' or 'cuda')
            
        Returns:
            Generated samples [num_samples, seq_len, feature_dim]
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.forward(z)
        return samples.cpu().numpy()


if __name__ == "__main__":
    # Test the generator
    print("Testing Bolt-GAN Generator...")
    
    # Hyperparameters
    BATCH_SIZE = 32
    LATENT_DIM = 100
    SEQ_LEN = 144
    FEATURE_DIM = 29
    
    # Initialize generator
    generator = BoltGANGenerator(
        latent_dim=LATENT_DIM,
        seq_len=SEQ_LEN,
        feature_dim=FEATURE_DIM
    )
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Test forward pass
    z = torch.randn(BATCH_SIZE, LATENT_DIM)
    output = generator(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test generation
    samples = generator.generate(num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
    
    print("✅ Generator test passed!")
