"""
Bolt-GAN Discriminator
LSTM-based discriminator for multivariate time series classification
"""

import torch
import torch.nn as nn


class BoltGANDiscriminator(nn.Module):
    """
    LSTM-based Discriminator for Bolt-GAN
    
    Args:
        seq_len: Length of input sequences (e.g., 144)
        feature_dim: Number of features (e.g., 29 for Energy)
        lstm_hidden: Hidden dimension for LSTM layers (default: 308)
        lstm_layers: Number of LSTM layers (default: 3)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(self, seq_len, feature_dim, 
                 lstm_hidden=308, lstm_layers=3, dropout=0.3):
        super(BoltGANDiscriminator, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.lstm_hidden = lstm_hidden
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * lstm_hidden, lstm_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1)
            # Note: No sigmoid here - we use BCEWithLogitsLoss
        )
        
    def forward(self, x):
        """
        Classify sequences as real or fake
        
        Args:
            x: Input sequences [batch_size, seq_len, feature_dim]
            
        Returns:
            Validity scores [batch_size, 1] (logits, not probabilities)
        """
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, lstm_hidden]
        
        # Flatten and classify
        validity = self.output_layers(lstm_out)  # [batch_size, 1]
        
        return validity


if __name__ == "__main__":
    # Test the discriminator
    print("Testing Bolt-GAN Discriminator...")
    
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 144
    FEATURE_DIM = 29
    
    # Initialize discriminator
    discriminator = BoltGANDiscriminator(
        seq_len=SEQ_LEN,
        feature_dim=FEATURE_DIM
    )
    
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM)
    output = discriminator(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test with sigmoid (for interpretation)
    probs = torch.sigmoid(output)
    print(f"Probability range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
    
    print("✅ Discriminator test passed!")
