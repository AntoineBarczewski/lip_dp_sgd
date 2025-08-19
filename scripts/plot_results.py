#!/usr/bin/env python3
"""
Script to collect results from multiple seed experiments and generate plots with standard deviations.
"""

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import re
from typing import Dict, List, Any
import logging

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def collect_results(results_dir: str) -> pd.DataFrame:
    """Collect results from multiple seed runs"""
    logger = setup_logging()
    results = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory {results_dir} not found!")
        return pd.DataFrame()
    
    logger.info(f"Collecting results from {results_dir}")
    
    # Look for dataset directories
    for dataset_dir in results_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Look for seed directories
        seed_count = 0
        for seed_dir in dataset_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                continue
                
            seed = seed_dir.name.split('_')[1]
            seed_count += 1
            
            # Look for metrics file first (preferred)
            metrics_file = seed_dir / 'final_metrics.yaml'
            config_file = seed_dir / 'config.yaml'
            
            if metrics_file.exists():
                # Load metrics directly
                with open(metrics_file, 'r') as f:
                    metrics = yaml.safe_load(f)
                
                # Load config for additional info
                config = {}
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                
                result = {
                    'dataset': dataset_name,
                    'seed': int(seed),
                    'final_accuracy': metrics.get('final_accuracy', None),
                    'final_epsilon': metrics.get('final_epsilon', None),
                    'best_accuracy': metrics.get('best_accuracy', None),
                    'final_loss': metrics.get('final_loss', None),
                    'total_runtime': metrics.get('total_runtime', None),
                    'num_epochs': metrics.get('num_epochs_completed', None),
                    'noise_std': metrics.get('noise_std', 0.0),
                    'learning_rate': metrics.get('learning_rate', 0.001),
                    'effective_batch_size': metrics.get('effective_batch_size', 32),
                    'use_augmentation': metrics.get('use_augmentation', False),
                    'use_ema': metrics.get('use_ema', False),
                    'source': 'metrics_file'
                }
                
                # Add EMA metrics if available
                if metrics.get('use_ema', False):
                    result.update({
                        'final_ema_accuracy': metrics.get('final_ema_accuracy', None),
                        'best_ema_accuracy': metrics.get('best_ema_accuracy', None),
                        'final_ema_loss': metrics.get('final_ema_loss', None)
                    })
                
            elif config_file.exists():
                # Fallback to parsing log file
                log_file = seed_dir / 'train_training.log'
                if log_file.exists():
                    metrics = parse_log_file(log_file)
                    
                    # Load config
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    result = {
                        'dataset': dataset_name,
                        'seed': int(seed),
                        'final_accuracy': metrics.get('final_accuracy', None),
                        'final_epsilon': metrics.get('final_epsilon', None),
                        'best_accuracy': metrics.get('best_accuracy', None),
                        'final_loss': metrics.get('final_loss', None),
                        'noise_std': config.get('noise_std', 0.0),
                        'learning_rate': config.get('learning_rate', 0.001),
                        'effective_batch_size': config.get('effective_batch_size', config.get('batch_size', 32)),
                        'use_augmentation': config.get('use_augmentation', False),
                        'use_ema': config.get('use_ema', False),
                        'source': 'log_file'
                    }
                    
                    # Add EMA metrics if available
                    if config.get('use_ema', False):
                        result.update({
                            'final_ema_accuracy': metrics.get('final_ema_accuracy', None),
                            'best_ema_accuracy': metrics.get('best_ema_accuracy', None)
                        })
                else:
                    logger.warning(f"No log file found for {dataset_name}/seed_{seed}")
                    continue
            else:
                logger.warning(f"No config file found for {dataset_name}/seed_{seed}")
                continue
            
            results.append(result)
        
        logger.info(f"Found {seed_count} seed runs for {dataset_name}")
    
    df = pd.DataFrame(results)
    logger.info(f"Collected {len(df)} total experiment runs across {df['dataset'].nunique() if not df.empty else 0} datasets")
    return df

def parse_log_file(log_file: Path) -> Dict[str, Any]:
    """Extract metrics from log file"""
    metrics = {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find final metrics and best accuracy
    for line in reversed(lines):
        line = line.strip()
        if 'Final test accuracy:' in line:
            try:
                metrics['final_accuracy'] = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'Final epsilon:' in line:
            try:
                metrics['final_epsilon'] = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'Final test loss:' in line:
            try:
                metrics['final_loss'] = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'Best accuracy achieved:' in line:
            try:
                metrics['best_accuracy'] = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'Final EMA test accuracy:' in line:
            try:
                metrics['final_ema_accuracy'] = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'Best EMA accuracy achieved:' in line:
            try:
                metrics['best_ema_accuracy'] = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
    
    return metrics

def plot_accuracy_comparison(df: pd.DataFrame, save_path: str = None, use_ema: bool = False):
    """Plot accuracy comparison across datasets with error bars"""
    accuracy_col = 'best_ema_accuracy' if use_ema else 'best_accuracy'
    
    # Filter out None values
    df_clean = df.dropna(subset=[accuracy_col])
    
    if df_clean.empty:
        print(f"No data available for {accuracy_col}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Calculate mean and std for each dataset
    summary = df_clean.groupby('dataset')[accuracy_col].agg(['mean', 'std', 'count']).reset_index()
    summary = summary.sort_values('mean', ascending=False)
    
    # Create bar plot with error bars
    bars = plt.bar(range(len(summary)), summary['mean'], 
                   yerr=summary['std'], capsize=5, alpha=0.7, 
                   color=sns.color_palette("husl", len(summary)))
    
    # Add individual points as scatter plot
    for i, dataset in enumerate(summary['dataset']):
        dataset_data = df_clean[df_clean['dataset'] == dataset][accuracy_col]
        x_pos = [i] * len(dataset_data)
        plt.scatter(x_pos, dataset_data, alpha=0.6, color='red', s=40, zorder=3)
    
    plt.ylabel('Best Test Accuracy', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    title = f'Model Performance Across Datasets {"(EMA)" if use_ema else ""}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(summary)), summary['dataset'], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add text annotations with mean ± std
    for i, row in summary.iterrows():
        plt.text(i, row['mean'] + row['std'] + 0.01, 
                f"{row['mean']:.3f} ± {row['std']:.3f}\n(n={row['count']})",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        suffix = "_ema" if use_ema else ""
        full_path = save_path.replace('.png', f'{suffix}.png')
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {full_path}")
    plt.show()

def plot_privacy_accuracy_tradeoff(df: pd.DataFrame, save_path: str = None):
    """Plot privacy-accuracy tradeoff"""
    # Filter out None values
    df_clean = df.dropna(subset=['best_accuracy', 'final_epsilon'])
    
    if df_clean.empty:
        print("No data available for privacy-accuracy tradeoff")
        return
    
    datasets = df_clean['dataset'].unique()
    
    if len(datasets) == 1:
        # Single subplot for one dataset
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    else:
        # Multiple subplots
        ncols = min(3, len(datasets))
        nrows = (len(datasets) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        if len(datasets) == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
    
    for i, dataset in enumerate(datasets):
        dataset_df = df_clean[df_clean['dataset'] == dataset]
        
        if len(dataset_df) == 0:
            continue
            
        ax = axes[i] if i < len(axes) else axes[-1]
        
        # Group by noise_std and calculate mean/std
        if 'noise_std' in dataset_df.columns:
            privacy_summary = dataset_df.groupby('noise_std').agg({
                'best_accuracy': ['mean', 'std', 'count'],
                'final_epsilon': ['mean', 'std']
            }).reset_index()
            
            privacy_summary.columns = ['noise_std', 'acc_mean', 'acc_std', 'acc_count', 'eps_mean', 'eps_std']
            
            # Plot with error bars
            ax.errorbar(privacy_summary['eps_mean'], privacy_summary['acc_mean'],
                       yerr=privacy_summary['acc_std'], xerr=privacy_summary['eps_std'],
                       marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
            
            # Add point labels with noise std
            for _, row in privacy_summary.iterrows():
                ax.annotate(f'σ={row["noise_std"]:.1f}\n(n={row["acc_count"]})', 
                           (row['eps_mean'], row['acc_mean']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            # Simple scatter plot if no noise_std grouping
            ax.scatter(dataset_df['final_epsilon'], dataset_df['best_accuracy'], 
                      alpha=0.7, s=50)
        
        ax.set_xlabel('Privacy Loss (ε)', fontsize=12)
        ax.set_ylabel('Best Test Accuracy', fontsize=12)
        ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Hide extra subplots
    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Privacy-Accuracy Tradeoff Across Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()

def plot_training_comparison(df: pd.DataFrame, save_path: str = None):
    """Plot comparison of different training configurations"""
    if df.empty:
        print("No data available for training comparison")
        return
    
    # Create a configuration label
    df = df.copy()
    df['config'] = df.apply(lambda row: f"Aug: {row.get('use_augmentation', False)}, EMA: {row.get('use_ema', False)}", axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy by dataset and configuration
    df_clean = df.dropna(subset=['best_accuracy'])
    if not df_clean.empty:
        summary = df_clean.groupby(['dataset', 'config'])['best_accuracy'].agg(['mean', 'std', 'count']).reset_index()
        
        datasets = summary['dataset'].unique()
        configs = summary['config'].unique()
        
        x = np.arange(len(datasets))
        width = 0.35
        
        for i, config in enumerate(configs):
            config_data = summary[summary['config'] == config]
            means = [config_data[config_data['dataset'] == d]['mean'].iloc[0] if len(config_data[config_data['dataset'] == d]) > 0 else 0 for d in datasets]
            stds = [config_data[config_data['dataset'] == d]['std'].iloc[0] if len(config_data[config_data['dataset'] == d]) > 0 else 0 for d in datasets]
            
            axes[0].bar(x + i*width, means, width, yerr=stds, label=config, alpha=0.8, capsize=5)
        
        axes[0].set_ylabel('Best Test Accuracy')
        axes[0].set_xlabel('Dataset')
        axes[0].set_title('Accuracy by Configuration')
        axes[0].set_xticks(x + width/2)
        axes[0].set_xticklabels(datasets, rotation=45)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Runtime analysis
    df_runtime = df.dropna(subset=['total_runtime'])
    if not df_runtime.empty:
        runtime_summary = df_runtime.groupby('dataset')['total_runtime'].agg(['mean', 'std']).reset_index()
        
        bars = axes[1].bar(runtime_summary['dataset'], runtime_summary['mean'], 
                          yerr=runtime_summary['std'], capsize=5, alpha=0.7)
        axes[1].set_ylabel('Training Time (seconds)')
        axes[1].set_xlabel('Dataset')
        axes[1].set_title('Training Runtime by Dataset')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add text annotations
        for i, (bar, row) in enumerate(zip(bars, runtime_summary.itertuples())):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + row.std,
                        f'{row.mean/60:.1f}±{row.std/60:.1f}min',
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()

def generate_summary_table(df: pd.DataFrame, save_path: str = None):
    """Generate a summary table of results"""
    if df.empty:
        print("No data available for summary table")
        return
    
    # Calculate summary statistics
    summary_stats = []
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        for metric in ['best_accuracy', 'final_epsilon']:
            if metric in dataset_df.columns:
                clean_data = dataset_df.dropna(subset=[metric])
                if len(clean_data) > 0:
                    stats = {
                        'Dataset': dataset.upper(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Mean': clean_data[metric].mean(),
                        'Std': clean_data[metric].std(),
                        'Min': clean_data[metric].min(),
                        'Max': clean_data[metric].max(),
                        'Count': len(clean_data)
                    }
                    summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    if not summary_df.empty:
        # Round numeric columns
        numeric_cols = ['Mean', 'Std', 'Min', 'Max']
        summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        if save_path:
            summary_df.to_csv(save_path, index=False)
            print(f"\nSummary table saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot results from multiple seed experiments')
    parser.add_argument('--results_dir', default='results/experiments',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', default='results/plots',
                       help='Directory to save plots')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect and analyze results
    df = collect_results(args.results_dir)
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"\nFound {len(df)} experiment runs across {df['dataset'].nunique()} datasets")
    print("Datasets:", list(df['dataset'].unique()))
    print("Seeds per dataset:")
    print(df.groupby('dataset')['seed'].count().to_string())
    
    # Generate plots
    file_ext = f".{args.format}"
    
    # Accuracy comparison plots
    plot_accuracy_comparison(df, os.path.join(args.output_dir, f'accuracy_comparison{file_ext}'))
    
    # EMA accuracy comparison if EMA data is available
    if 'best_ema_accuracy' in df.columns and df['best_ema_accuracy'].notna().any():
        plot_accuracy_comparison(df, os.path.join(args.output_dir, f'accuracy_comparison_ema{file_ext}'), use_ema=True)
    
    # Privacy-accuracy tradeoff
    plot_privacy_accuracy_tradeoff(df, os.path.join(args.output_dir, f'privacy_accuracy_tradeoff{file_ext}'))
    
    # Training comparison
    plot_training_comparison(df, os.path.join(args.output_dir, f'training_comparison{file_ext}'))
    
    # Generate summary table
    generate_summary_table(df, os.path.join(args.output_dir, 'results_summary.csv'))
    
    # Save raw data
    df.to_csv(os.path.join(args.output_dir, 'all_results.csv'), index=False)
    print(f"\nAll results saved to {os.path.join(args.output_dir, 'all_results.csv')}")
    
    print(f"\nAll plots and summaries saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
