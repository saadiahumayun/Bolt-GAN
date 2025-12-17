# Bolt-GAN
# Bolt-GAN: Multivariate Time Series Data Generator

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange)](.)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official PyTorch implementation of "Bolt-GAN: A Multivariate Time-Series Data Generator"**

> 📝 **Status:** Paper under review  
> 🔗 **Preprint:** [Coming soon]  
> 📧 **Contact:** s.humayun.27269@khi.iba.edu.pk

---

## 🎯 Overview

Bolt-GAN introduces a **novel KL-divergence-based feedback mechanism** that periodically refines the generator during training, achieving superior synthetic data quality across diverse time-series domains.

### Key Innovation: The "Bolt" Feedback Mechanism

Unlike traditional GANs that rely solely on discriminator feedback, Bolt-GAN:
1. Trains normally for N epochs (e.g., 500 epochs = 20% of training)
2. **Computes distributional gaps** using KL divergence
3. **Applies "bolt" feedback** to adjust generator input
4. **Refines for M epochs** (e.g., 100 epochs) with enhanced signal
5. **Repeats** throughout training

This mechanism prevents mode collapse and improves distributional fidelity.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Bolt-GAN.git
cd Bolt-GAN

# Install dependencies
pip install -r requirements.txt
```

### Train Bolt-GAN

```python
from boltgan import BoltGANTrainer
import numpy as np

# Load your data (shape: [samples, timesteps, features])
data = np.load('your_data.npy')

# Initialize trainer
trainer = BoltGANTrainer(
    seq_len=data.shape[1],
    feature_dim=data.shape[2],
    latent_dim=100,
    num_epochs=2500,
    feedback_interval=500,  # Apply bolt every 500 epochs (20%)
    feedback_epochs=100     # Refine for 100 epochs after bolt
)

# Train with feedback mechanism
trainer.train(data, save_dir='./results')

# Generate synthetic data
synthetic_data = trainer.generate(num_samples=1000)
```

### Command Line

```bash
# Train on your dataset
python train.py --data your_data.npy --epochs 2500 --output ./results

# Quick test (fewer epochs for demo)
python train.py --data your_data.npy --epochs 100 --feedback-interval 20
```

---

## 🔬 The Feedback Mechanism Explained

### Mathematical Formulation

At each feedback interval (every 500 epochs), we compute:

```
Z_new = KL(X||X̂) × Z + KL(X̂||Z) × X̂_collapsed
```

Where:
- `X` = real data distribution
- `X̂` = generated data distribution
- `Z` = noise input distribution
- `KL(·||·)` = Kullback-Leibler divergence

### Why This Works

1. **KL(X||X̂)** measures how far generated data is from real data
   - High value → generator needs more guidance → increase noise influence
   
2. **KL(X̂||Z)** measures how far generator has moved from noise
   - High value → generator has learned structure → preserve learned patterns

3. **Adaptive adjustment** prevents:
   - Mode collapse (generator stuck on few patterns)
   - Training instability (generator oscillates)
   - Distributional drift (generator forgets earlier learning)

---

## 📁 Repository Structure

```
Bolt-GAN/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── boltgan/                  # Core package
│   ├── __init__.py
│   ├── generator.py          # LSTM-based generator
│   ├── discriminator.py      # LSTM-based discriminator
│   └── trainer.py            # Training with feedback mechanism
├── train.py                  # Command-line training script
└── example.ipynb             # Jupyter notebook demo
```

---

## 📦 Requirements

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.11.0         # For KL divergence computation
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 🎓 Key Hyperparameters

### Architecture
- `latent_dim`: Noise dimension (default: 100)
- `lstm_hidden`: LSTM hidden size (default: 308)
- `lstm_layers`: Number of LSTM layers (default: 3)
- `dropout`: Dropout rate (default: 0.3)

### Training
- `num_epochs`: Total training epochs (default: 2500)
- `batch_size`: Batch size (default: 32)
- `lr`: Learning rate (default: 0.0002)

### Feedback (Critical!)
- `feedback_interval`: Apply bolt every N epochs (default: 500)
  - Set to 20% of total epochs
  - Example: 2500 epochs → interval = 500
  
- `feedback_epochs`: Refine for N epochs after bolt (default: 100)
  - Additional training with enhanced noise
  - 4-5% of total epochs recommended

---

## 🔧 Advanced Usage

### Custom Feedback Schedule

```python
trainer = BoltGANTrainer(
    seq_len=144,
    feature_dim=29,
    num_epochs=5000,
    feedback_interval=1000,  # Every 1000 epochs
    feedback_epochs=200      # Refine for 200 epochs
)
```

### Load Pretrained Model

```python
checkpoint = torch.load('results/boltgan_epoch2500.pth')
trainer.generator.load_state_dict(checkpoint['generator'])
trainer.discriminator.load_state_dict(checkpoint['discriminator'])

# Generate samples
synthetic_data = trainer.generate(num_samples=1000)
```

---

## 📊 Evaluation Metrics

### Distributional Quality
- **Chi-square test** (χ²): p-value > 0.05 = statistically indistinguishable
- **Total Variation**: TV Complement > 0.85 = acceptable quality
- **KL Similarity**: exp(-KL divergence) > 0.80 = good match

### Downstream Utility
- **Regression**: R² score, RMSE, MAE
- **Classification**: AUROC, AUPRC (important for imbalanced data!)

---

## 🐛 Troubleshooting

### "RuntimeError: KDE failed"
- KL divergence computation uses scipy KDE which can fail on degenerate data
- Solution: Code automatically skips failed dimensions and averages over successful ones

### "Feedback makes training worse"
- Try reducing feedback influence: `z = z_base + 0.05 * feedback` (line 254 in trainer.py)
- Increase feedback_interval (e.g., 1000 instead of 500)

### "Out of memory"
- Reduce batch_size: `--batch-size 16`
- Reduce lstm_hidden: `--lstm-hidden 128`
- Reduce feedback computation: Limit to first 50 dimensions (already implemented)

---

## 📧 Contact

**Saadia Humayun**  
Institute of Business Administration, Karachi  
📧 s.humayun.27269@khi.iba.edu.pk

**Hira Zahid:** hzahid@iba.edu.pk  
**Tariq Mahmood:** tmahmood@iba.edu.pk

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Datasets used:
- Energy Appliances: UCI Machine Learning Repository
- Air Quality: Beijing Municipal Environmental Monitoring Center  
- CICIDS-2017: Canadian Institute for Cybersecurity
- PICU: Available upon request with ethics approval

---

**⭐ If you find this useful, please star the repo!**
