# Differentially Private Deep Learning with Weight Clipping

Code for reproducing the experiments in *[paper title]* ([arXiv:2310.18001](https://arxiv.org/abs/2310.18001)).
We train deep networks with DP-SGD while enforcing Lipschitz constraints via spectral normalization, weight standardization, and max-input-norm tracking, yielding tighter per-sample gradient bounds without explicit per-sample clipping.

## Method overview

- **Lipschitz-constrained architectures.** Convolutional and dense layers are wrapped with spectral normalization and a `MaxInputNormLayer` that tracks activation norms at each layer. Convolutions use Scaled Weight Standardization (`WSConv2D`) with GroupNorm, following the NFNet recipe.
- **Privacy accounting.** We compose subsampled Gaussian mechanisms via Renyi Differential Privacy (autodp). Training halts automatically when the cumulative epsilon exceeds a configurable budget.
- **Augmentation multiplicity.** For CIFAR-10 we apply K independent augmentations per sample and average their per-sample gradients before noise addition, improving the signal-to-noise ratio at no additional privacy cost.
- **Exponential moving average (EMA).** Model weights are averaged with a warmup schedule; evaluation is performed on the EMA parameters.

## Results

Reported over 5 seeds. Training uses SGD with gradient accumulation (physical batch -> effective batch).

| Dataset       | Accuracy (mean +/- std) | epsilon (delta=1e-5) | Model    | Epochs |
|---------------|-------------------------|----------------------|----------|--------|
| MNIST         | 0.991 +/- 0.0003       | ~4.1                 | CNN      | 100    |
| Fashion-MNIST | 0.917 +/- 0.0005       | ~4.1                 | CNN      | 100    |
| CIFAR-10      | 0.861 +/- 0.0014       | ~7.5                 | ResNet18 | 120    |

## Installation

```bash
git clone https://github.com/AntoineBarczewski/lip_dp_sgd.git
cd lip_dp_sgd
pip install -r requirements.txt
```

JAX defaults to CPU. For GPU support, install the appropriate JAX variant first:

```bash
pip install jax[cuda12_pip]   # CUDA 12
pip install jax[cuda11_pip]   # CUDA 11
```

## Reproducing all experiments

```bash
bash scripts/run_all_experiments.sh        # 3 datasets x 5 seeds
bash scripts/run_all_experiments.sh 10     # use 10 seeds instead
```

Results are written to `results/experiments/` (per-run metrics and checkpoints) and `results/plots/` (figures and CSV summaries).

## Running individual experiments

Single run:

```bash
python train.py --config configs/mnist.yaml --seed 42 --output_dir results/single_run
```

Multiple seeds on one dataset:

```bash
python scripts/run_multiple_seeds.py \
    --config configs/cifar10.yaml \
    --seeds 42 123 456 789 999 \
    --output_dir results/cifar10_study
```

Generate plots from existing results:

```bash
python scripts/plot_results.py \
    --results_dir results/experiments \
    --output_dir results/plots
```

## Configuration

Dataset-specific YAML files live in `configs/`. Key parameters:

| Parameter              | Description                                      |
|------------------------|--------------------------------------------------|
| `noise_std`            | Gaussian noise multiplier (sigma)                |
| `delta`                | Privacy parameter delta                          |
| `max_epsilon`          | Stop training when epsilon exceeds this          |
| `batch_size`           | Physical batch size (fits in memory)             |
| `effective_batch_size` | Logical batch size (via gradient accumulation)   |
| `use_augmentation`     | Enable augmentation multiplicity                 |
| `augment_mult`         | Number of augmented copies per sample            |
| `use_ema`              | Enable exponential moving average of weights     |

## Repository structure

```
train.py                  Main training loop
models.py                 CNN (MNIST/FMNIST), ResNet18/34 (CIFAR-10)
utils.py                  Train state, EMA, gradient accumulation
privacy.py                RDP-based privacy accounting, noise injection
augmult.py                Augmentation multiplicity
configs/                  Per-dataset hyperparameter configs
scripts/
  run_multiple_seeds.py   Multi-seed runner
  plot_results.py         Plotting and summary statistics
  run_all_experiments.sh  End-to-end automation
```

## Quick verification for reviewers

1. Install dependencies (~2 min):
   ```bash
   pip install -r requirements.txt
   ```

2. Fast sanity check on MNIST (~10 min on CPU):
   ```bash
   python scripts/run_multiple_seeds.py \
       --config configs/mnist.yaml \
       --seeds 42 123 \
       --output_dir quick_test
   ```

3. Inspect results:
   ```bash
   python scripts/plot_results.py --results_dir quick_test --output_dir quick_plots
   ```

## Citation

```bibtex
@article{barczewski2025dp,
  author  = {Antoine Barczewski},
  title   = {Differentially Private Deep Learning with Weight Clipping},
  year    = {2025},
  url     = {https://github.com/AntoineBarczewski/lip_dp_sgd}
}
```

## License

MIT. See [LICENSE](LICENSE).
