# Differentially Private Deep Learning with Weight Clipping

This repository contains the implementation for reproducing results on DP-SGD with weight clipping across MNIST, Fashion-MNIST, and CIFAR-10 datasets.

## Quick Start (One Command)

To reproduce all results with multiple seeds:

```bash
bash scripts/run_all_experiments.sh
```

This will:
- Run experiments on all three datasets (MNIST, Fashion-MNIST, CIFAR-10)
- Use 5 different random seeds for statistical significance
- Generate plots with error bars and standard deviations
- Create a comprehensive summary report

## 📁 Repository Structure

```
lip_dp_sgd/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── train.py                     # Main training script (enhanced)
├── config.yaml                  # Default configuration
├── configs/                     # Dataset-specific configurations
│   ├── mnist.yaml
│   ├── fashion_mnist.yaml
│   └── cifar10.yaml
├── scripts/                     # Automation and analysis scripts
│   ├── run_multiple_seeds.py    # Multi-seed experiment runner
│   ├── plot_results.py          # Results analysis and plotting
│   └── run_all_experiments.sh   # Master automation script
├── models.py                    # Neural network architectures
├── utils.py                     # Training utilities
├── augmult.py                   # Data augmentation
├── privacy.py                   # Privacy accounting
├── results/                     # Generated results
│   ├── experiments/             # Raw experiment data
│   └── plots/                   # Generated plots and summaries
└── logs/                        # Training logs
```

## 🛠️ Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/AntoineBarczewski/lip_dp_sgd.git
cd lip_dp_sgd

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda (recommended)

```bash
# Clone the repository
git clone https://github.com/AntoineBarczewski/lip_dp_sgd.git
cd lip_dp_sgd

# Create conda environment
conda create -n dp_sgd python=3.11
conda activate dp_sgd

# Install JAX (adjust for your CUDA version if using GPU)
pip install jax[cpu]  # For CPU
# OR
pip install jax[cuda12_pip]  # For CUDA 12
# OR  
pip install jax[cuda11_pip]  # For CUDA 11

# Install remaining dependencies
pip install -r requirements.txt
```

## 📊 Expected Results

When you run the full experiment suite, you should expect results similar to:

| Dataset | Best Accuracy (mean ± std) | Privacy Loss (ε) |
|---------|----------------------------|-------------------|
| MNIST | 0.995 ± 0.003 | ~5.5 |
| Fashion-MNIST | 0.935 ± 0.005 | ~4.0 |
| CIFAR-10 | 0.863 ± 0.008 | ~5.1 |

*Note: Actual results may vary slightly due to hardware differences and random initialization.*

## 🎯 Usage Examples

### Run All Experiments (Recommended for Reviewers)

```bash
# Run complete reproducibility study (may take several hours)
bash scripts/run_all_experiments.sh

# Or with custom number of seeds
bash scripts/run_all_experiments.sh 10  # Use seeds 1-10
```

### Run Single Dataset with Multiple Seeds

```bash
# MNIST with 5 seeds
python scripts/run_multiple_seeds.py \
    --config configs/mnist.yaml \
    --seeds 42 123 456 789 999 \
    --output_dir results/mnist_study

# CIFAR-10 with 3 seeds (faster for testing)
python scripts/run_multiple_seeds.py \
    --config configs/cifar10.yaml \
    --seeds 42 123 456 \
    --output_dir results/cifar10_test
```

### Run Single Experiment

```bash
# Single run with specific seed
python train.py --config configs/mnist.yaml --seed 42 --output_dir results/single_run
```

### Generate Plots from Existing Results

```bash
python scripts/plot_results.py \
    --results_dir results/experiments \
    --output_dir results/plots \
    --format png
```

## ⚙️ Configuration

### Dataset-Specific Configurations

Each dataset has optimized hyperparameters in `configs/`:

- **MNIST**: 50 epochs, batch size 64→256, no augmentation multiplicity
- **Fashion-MNIST**: 50 epochs, batch size 64→256, no augmentation multiplicity 3  
- **CIFAR-10**: 120 epochs, batch size 32→512, augmentation multiplicity 4

### Key Configuration Parameters

```yaml
# Privacy settings
noise_std: 1.0              # Noise multiplier
delta: 1e-5                 # Privacy parameter
max_epsilon: 10.0           # Stop training when ε exceeds this

# Training settings
learning_rate: 0.001
effective_batch_size: 512   # Logical batch size
batch_size: 32              # Physical batch size (for memory)

# Techniques
use_augmentation: true      # Enable data augmentation
use_ema: true              # Enable exponential moving averages
```

## 📈 Generated Outputs

After running experiments, you'll find:

### Results Directory Structure
```
results/
├── experiments/           # Raw experiment data
│   ├── mnist/
│   │   ├── seed_42/
│   │   │   ├── config.yaml
│   │   │   ├── final_metrics.yaml
│   │   │   ├── train_training.log
│   │   │   └── best_model_*.pkl
│   │   └── experiment_summary.yaml
│   ├── fashion_mnist/
│   └── cifar10/
└── plots/                 # Analysis and visualizations
    ├── accuracy_comparison.png
    ├── accuracy_comparison_ema.png
    ├── privacy_accuracy_tradeoff.png
    ├── training_comparison.png
    ├── results_summary.csv
    └── all_results.csv
```

### Key Output Files

1. **accuracy_comparison.png** - Bar chart comparing datasets with error bars
2. **privacy_accuracy_tradeoff.png** - Privacy-utility tradeoff analysis
3. **results_summary.csv** - Statistical summary table
4. **all_results.csv** - Complete raw data for further analysis

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size in config files
   batch_size: 16  # Instead of 32
   effective_batch_size: 256  # Keep same logical size
   ```

2. **JAX/CUDA Issues**
   ```bash
   # Reinstall JAX for your CUDA version
   pip uninstall jax jaxlib
   pip install jax[cuda12_pip]  # or cuda11_pip
   ```

3. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install autodp  # For privacy accounting
   ```

4. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x scripts/run_all_experiments.sh
   ```

### Debug Mode

For debugging individual runs:

```bash
# Enable debug logging
python train.py --config configs/mnist.yaml --seed 42 --output_dir debug_run
# Check logs in debug_run/train_training.log
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{barczewski2025dp,
  author = {Antoine Barczewski},
  title = {Differentially Private Deep Learning with Weight Clipping},
  year = {2025},
  url = {https://github.com/AntoineBarczewski/lip_dp_sgd}
}
```

## 📧 Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [Create an issue](https://github.com/AntoineBarczewski/lip_dp_sgd/issues)
- **Email**: [Your email if you want to provide it]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- JAX and Flax teams for the deep learning framework
- TensorFlow Datasets for easy dataset access
- AutoDP library for privacy accounting
- The differential privacy research community

---

## 🔄 For Reviewers: Quick Verification

To quickly verify our claims:

1. **Clone and setup** (5 minutes):
   ```bash
   git clone https://github.com/AntoineBarczewski/lip_dp_sgd.git
   cd lip_dp_sgd
   pip install -r requirements.txt
   ```

2. **Quick test run** (10 minutes):
   ```bash
   # Test on MNIST with 2 seeds and fewer epochs
   python scripts/run_multiple_seeds.py \
       --config configs/mnist.yaml \
       --seeds 42 123 \
       --output_dir quick_test
   # Modify configs/mnist.yaml: num_epochs: 5 for ultra-fast testing
   ```

3. **View results**:
   ```bash
   python scripts/plot_results.py --results_dir quick_test --output_dir quick_plots
   ```

4. **Full reproduction** (if desired):
   ```bash
   bash scripts/run_all_experiments.sh
   ```

The repository is designed to make reproduction as straightforward as possible while maintaining scientific rigor.
