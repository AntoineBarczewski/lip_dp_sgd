#!/bin/bash

# Run all experiments with multiple seeds for all datasets
# Usage: bash scripts/run_all_experiments.sh [number_of_seeds]

set -e  # Exit on any error

# Configuration
SEEDS="42 123 456 789 999"  # Default seeds
OUTPUT_DIR="results/experiments"
PYTHON_CMD="python"

# Override seeds if provided as argument
if [ $# -gt 0 ]; then
    # Generate seeds from 1 to N
    N=$1
    SEEDS=$(seq 1 $N | xargs)
    echo "Using $N seeds: $SEEDS"
else
    echo "Using default seeds: $SEEDS"
fi

echo "Starting experiments for all datasets..."
echo "Output directory: $OUTPUT_DIR"
echo "Seeds: $SEEDS"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create configs directory if it doesn't exist
mkdir -p configs

# Check if config files exist, if not create them
if [ ! -f "configs/mnist.yaml" ]; then
    echo "Creating dataset config files..."
    
    # Create MNIST config
    cat > configs/mnist.yaml << EOF
dataset: mnist
depth: 18
model_save_path: best_model.pkl
random_seed: 42

batch_size: 64
effective_batch_size: 256

learning_rate: 0.001
num_epochs: 50
noise_std: 1.0
delta: 1e-5
max_epsilon: 10.0

use_augmentation: true
augment_mult: 2

use_ema: true
ema:
  mu: 0.999
  use_warmup: true
  start_step: 0
  eval_with_ema: true

augmentation:
  random_crop: true
  crop_size: [28, 28]
  crop_padding: 2
  random_flip: true
  flip_prob: 0.5
  random_brightness: true
  brightness_delta: 0.1
  random_contrast: true
  contrast_range: [0.9, 1.1]
  random_saturation: false
  random_hue: false
EOF

    # Create Fashion-MNIST config
    cat > configs/fashion_mnist.yaml << EOF
dataset: fashion_mnist
depth: 18
model_save_path: best_model.pkl
random_seed: 42

batch_size: 64
effective_batch_size: 256

learning_rate: 0.001
num_epochs: 50
noise_std: 1.0
delta: 1e-5
max_epsilon: 10.0

use_augmentation: true
augment_mult: 3

use_ema: true
ema:
  mu: 0.999
  use_warmup: true
  start_step: 0
  eval_with_ema: true

augmentation:
  random_crop: true
  crop_size: [28, 28]
  crop_padding: 2
  random_flip: true
  flip_prob: 0.5
  random_brightness: true
  brightness_delta: 0.1
  random_contrast: true
  contrast_range: [0.9, 1.1]
  random_saturation: false
  random_hue: false
EOF

    # Create CIFAR-10 config
    cat > configs/cifar10.yaml << EOF
dataset: cifar10
depth: 18
model_save_path: best_model.pkl
random_seed: 42

batch_size: 32
effective_batch_size: 512

learning_rate: 0.001
num_epochs: 120
noise_std: 1.0
delta: 1e-5
max_epsilon: 10.0

use_augmentation: true
augment_mult: 4

use_ema: true
ema:
  mu: 0.999
  use_warmup: true
  start_step: 0
  eval_with_ema: true

augmentation:
  random_crop: true
  crop_size: [32, 32]
  crop_padding: 4
  random_flip: true
  flip_prob: 0.5
  random_brightness: true
  brightness_delta: 0.2
  random_contrast: true
  contrast_range: [0.8, 1.2]
  random_saturation: true
  saturation_range: [0.8, 1.2]
  random_hue: true
  hue_delta: 0.1
EOF

    echo "Config files created in configs/ directory"
fi

# Function to run experiments for a single dataset
run_dataset_experiments() {
    local config_file="$1"
    local dataset_name=$(basename "$config_file" .yaml)
    
    echo ""
    echo "="*60
    echo "Running experiments for $dataset_name"
    echo "="*60
    
    if [ ! -f "$config_file" ]; then
        echo "Error: Config file $config_file not found!"
        return 1
    fi
    
    $PYTHON_CMD scripts/run_multiple_seeds.py \
        --config "$config_file" \
        --seeds $SEEDS \
        --output_dir "$OUTPUT_DIR" \
        --python "$PYTHON_CMD"
    
    if [ $? -eq 0 ]; then
        echo "✓ $dataset_name experiments completed successfully"
    else
        echo "✗ $dataset_name experiments failed"
        return 1
    fi
}

# Check if Python scripts exist
if [ ! -f "scripts/run_multiple_seeds.py" ]; then
    echo "Error: scripts/run_multiple_seeds.py not found!"
    echo "Please make sure you have created the script files."
    exit 1
fi

# Record start time
start_time=$(date +%s)
echo "Experiment started at: $(date)"

# Run experiments for each dataset
failed_datasets=()

for config in configs/*.yaml; do
    if [ -f "$config" ]; then
        if ! run_dataset_experiments "$config"; then
            dataset_name=$(basename "$config" .yaml)
            failed_datasets+=("$dataset_name")
        fi
    fi
done

# Record end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "="*60
echo "EXPERIMENT SUMMARY"
echo "="*60
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
echo "Seeds used: $SEEDS"

if [ ${#failed_datasets[@]} -eq 0 ]; then
    echo "✓ All experiments completed successfully!"
    
    echo ""
    echo "Generating plots and summary..."
    
    # Generate plots if the plotting script exists
    if [ -f "scripts/plot_results.py" ]; then
        $PYTHON_CMD scripts/plot_results.py \
            --results_dir "$OUTPUT_DIR" \
            --output_dir "results/plots"
        
        if [ $? -eq 0 ]; then
            echo "✓ Plots and summary generated successfully"
            echo "Results available in:"
            echo "  - Experiment data: $OUTPUT_DIR"
            echo "  - Plots and summary: results/plots"
        else
            echo "⚠ Warning: Plot generation failed, but experiments completed"
        fi
    else
        echo "⚠ Warning: Plot script not found, skipping plot generation"
        echo "Results available in: $OUTPUT_DIR"
    fi
    
    exit 0
else
    echo "✗ Some experiments failed:"
    for dataset in "${failed_datasets[@]}"; do
        echo "  - $dataset"
    done
    echo ""
    echo "Check the logs in $OUTPUT_DIR for details"
    exit 1
fi
