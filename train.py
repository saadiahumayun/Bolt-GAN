"""
Simple training script for Bolt-GAN

Usage:
    python train.py --data path/to/data.npy --epochs 2500
"""

import argparse
import numpy as np
from boltgan import BoltGANTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Bolt-GAN')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (.npy file)')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=100,
                       help='Latent dimension (default: 100)')
    parser.add_argument('--lstm-hidden', type=int, default=308,
                       help='LSTM hidden size (default: 308)')
    parser.add_argument('--lstm-layers', type=int, default=3,
                       help='Number of LSTM layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=2500,
                       help='Number of training epochs (default: 2500)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate (default: 0.0002)')
    parser.add_argument('--feedback-interval', type=int, default=500,
                       help='Apply bolt every N epochs (default: 500)')
    parser.add_argument('--feedback-epochs', type=int, default=100,
                       help='Train for N epochs after bolt (default: 100)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    print(f"Data shape: {data.shape}")
    
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D data [samples, timesteps, features], got shape {data.shape}")
    
    num_samples, seq_len, feature_dim = data.shape
    print(f"Samples: {num_samples}, Sequence length: {seq_len}, Features: {feature_dim}")
    
    # Initialize trainer
    trainer = BoltGANTrainer(
        seq_len=seq_len,
        feature_dim=feature_dim,
        latent_dim=args.latent_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        lr=args.lr,
        num_epochs=args.epochs,
        feedback_interval=args.feedback_interval,
        feedback_epochs=args.feedback_epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Train
    print("\nStarting training...")
    g_losses, d_losses = trainer.train(data, save_dir=args.output)
    
    # Generate sample synthetic data
    print("\nGenerating 100 sample synthetic sequences...")
    synthetic_data = trainer.generate(num_samples=100)
    
    # Save synthetic data
    synthetic_path = f"{args.output}/synthetic_samples.npy"
    np.save(synthetic_path, synthetic_data)
    print(f"Saved synthetic data to {synthetic_path}")
    
    print("\n✅ Training complete!")
    print(f"Final G_loss: {g_losses[-1]:.4f}")
    print(f"Final D_loss: {d_losses[-1]:.4f}")
    print(f"Checkpoints saved in: {args.output}/")


if __name__ == '__main__':
    main()
