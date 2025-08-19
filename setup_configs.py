#!/usr/bin/env python3
"""
Script to create dataset-specific configuration files.
Run this once to set up the configs directory.
"""

import os
import yaml

def create_configs():
    """Create dataset-specific configuration files"""
    
    # Create configs directory
    os.makedirs('configs', exist_ok=True)
    
    # MNIST configuration
    mnist_config = {
        'dataset': 'mnist',
        'depth': 18,
        'model_save_path': 'best_model.pkl',
        'random_seed': 42,
        
        'batch_size': 64,
        'effective_batch_size': 256,
        
        'learning_rate': 0.001,
        'num_epochs': 50,
        'noise_std': 1.0,
        'delta': 1e-5,
        'max_epsilon': 10.0,
        
        'use_augmentation': True,
        'augment_mult': 2,
        
        'use_ema': True,
        'ema': {
            'mu': 0.999,
            'use_warmup': True,
            'start_step': 0,
            'eval_with_ema': True
        },
        
        'augmentation': {
            'random_crop': True,
            'crop_size': [28, 28],
            'crop_padding': 2,
            'random_flip': True,
            'flip_prob': 0.5,
            'random_brightness': True,
            'brightness_delta': 0.1,
            'random_contrast': True,
            'contrast_range': [0.9, 1.1],
            'random_saturation': False,
            'random_hue': False
        }
    }
    
    # Fashion-MNIST configuration
    fashion_mnist_config = mnist_config.copy()
    fashion_mnist_config['dataset'] = 'fashion_mnist'
    fashion_mnist_config['augment_mult'] = 3
    
    # CIFAR-10 configuration
    cifar10_config = {
        'dataset': 'cifar10',
        'depth': 18,
        'model_save_path': 'best_model.pkl',
        'random_seed': 42,
        
        'batch_size': 32,
        'effective_batch_size': 512,
        
        'learning_rate': 0.001,
        'num_epochs': 120,
        'noise_std': 1.0,
        'delta': 1e-5,
        'max_epsilon': 10.0,
        
        'use_augmentation': True,
        'augment_mult': 4,
        
        'use_ema': True,
        'ema': {
            'mu': 0.999,
            'use_warmup': True,
            'start_step': 0,
            'eval_with_ema': True
        },
        
        'augmentation': {
            'random_crop': True,
            'crop_size': [32, 32],
            'crop_padding': 4,
            'random_flip': True,
            'flip_prob': 0.5,
            'random_brightness': True,
            'brightness_delta': 0.2,
            'random_contrast': True,
            'contrast_range': [0.8, 1.2],
            'random_saturation': True,
            'saturation_range': [0.8, 1.2],
            'random_hue': True,
            'hue_delta': 0.1
        }
    }
    
    # Save configurations
    configs = {
        'mnist': mnist_config,
        'fashion_mnist': fashion_mnist_config,
        'cifar10': cifar10_config
    }
    
    for name, config in configs.items():
        config_path = f'configs/{name}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Created {config_path}")
    
    print(f"\nDataset configurations created in configs/ directory")
    print("You can now run experiments with:")
    print("  python scripts/run_multiple_seeds.py --config configs/mnist.yaml")
    print("  bash scripts/run_all_experiments.sh")

if __name__ == "__main__":
    create_configs()
