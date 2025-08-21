#!/usr/bin/env python3
"""
Script to plot training progression: accuracy vs epsilon over epochs with confidence intervals.
Parses train_training.log files to extract epoch-by-epoch data.
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
import json
from typing import Dict, List, Any, Tuple
import logging
from scipy import stats

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_sota_data(dataset_name: str, sota_dir: str = "data/sota") -> List[Dict]:
    """Load state-of-the-art data points for comparison"""
    sota_file = os.path.join(sota_dir, f"{dataset_name}.txt")
    
    logger = setup_logging()
    logger.info(f"Looking for SOTA file: {sota_file}")
    
    if not os.path.exists(sota_file):
        logger.warning(f"SOTA file not found: {sota_file}")
        # List available files in the directory for debugging
        if os.path.exists(sota_dir):
            available_files = os.listdir(sota_dir)
            logger.info(f"Available files in {sota_dir}: {available_files}")
        else:
            logger.warning(f"SOTA directory does not exist: {sota_dir}")
        return []
    
    sota_data = []
    try:
        with open(sota_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        sota_data.append(data)
                        logger.debug(f"Loaded SOTA data line {line_num}: {data['legend']}")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error on line {line_num} in {sota_file}: {e}")
                        logger.error(f"Problematic line: {line}")
        
        logger.info(f"Successfully loaded {len(sota_data)} SOTA entries from {sota_file}")
        
    except Exception as e:
        logger.error(f"Error reading SOTA file {sota_file}: {e}")
    
    return sota_data

def parse_training_log(log_file: Path) -> pd.DataFrame:
    """Parse training log to extract epoch-by-epoch metrics"""
    
    if not log_file.exists():
        return pd.DataFrame()
    
    epochs_data = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        # Look for epoch training lines like:
        # "Epoch 0, Train Loss: 2.3456, Test Loss: 2.1234, Test Accuracy: 0.1234, Epsilon: 1.2345, Runtime: 12.34s"
        # or with EMA:
        # "Epoch 0, Train Loss: 2.3456, Test Loss: 2.1234, Test Accuracy: 0.1234, EMA Test Loss: 2.0000, EMA Test Accuracy: 0.1300, Epsilon: 1.2345, Runtime: 12.34s"
        
        epoch_match = re.search(r'Epoch (\d+),', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            
            # Extract metrics using regex
            train_loss_match = re.search(r'Train Loss: ([\d.]+)', line)
            test_loss_match = re.search(r'Test Loss: ([\d.]+)', line)
            test_acc_match = re.search(r'Test Accuracy: ([\d.]+)', line)
            epsilon_match = re.search(r'Epsilon: ([\d.]+)', line)
            runtime_match = re.search(r'Runtime: ([\d.]+)s', line)
            
            # EMA metrics (optional)
            ema_test_loss_match = re.search(r'EMA Test Loss: ([\d.]+)', line)
            ema_test_acc_match = re.search(r'EMA Test Accuracy: ([\d.]+)', line)
            
            if all([train_loss_match, test_loss_match, test_acc_match, epsilon_match]):
                epoch_data = {
                    'epoch': epoch,
                    'train_loss': float(train_loss_match.group(1)),
                    'test_loss': float(test_loss_match.group(1)),
                    'test_accuracy': float(test_acc_match.group(1)),
                    'epsilon': float(epsilon_match.group(1)),
                    'runtime': float(runtime_match.group(1)) if runtime_match else None
                }
                
                # Add EMA metrics if available
                if ema_test_loss_match and ema_test_acc_match:
                    epoch_data.update({
                        'ema_test_loss': float(ema_test_loss_match.group(1)),
                        'ema_test_accuracy': float(ema_test_acc_match.group(1))
                    })
                
                epochs_data.append(epoch_data)
    
    return pd.DataFrame(epochs_data)

def collect_training_data(results_dir: str) -> Dict[str, List[pd.DataFrame]]:
    """Collect training progression data from all seed runs"""
    logger = setup_logging()
    training_data = {}
    
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory {results_dir} not found!")
        return {}
    
    logger.info(f"Collecting training data from {results_dir}")
    
    # Look for dataset directories
    for dataset_dir in results_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        logger.info(f"Processing dataset: {dataset_name}")
        
        dataset_runs = []
        
        # Look for seed directories
        for seed_dir in dataset_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                continue
                
            seed = seed_dir.name.split('_')[1]
            log_file = seed_dir / 'train_training.log'
            
            if log_file.exists():
                training_df = parse_training_log(log_file)
                if not training_df.empty:
                    training_df['seed'] = int(seed)
                    training_df['dataset'] = dataset_name
                    dataset_runs.append(training_df)
                    logger.debug(f"Parsed {len(training_df)} epochs for {dataset_name}/seed_{seed}")
                else:
                    logger.warning(f"No training data found in {log_file}")
            else:
                logger.warning(f"No log file found: {log_file}")
        
        if dataset_runs:
            training_data[dataset_name] = dataset_runs
            logger.info(f"Found {len(dataset_runs)} training runs for {dataset_name}")
        else:
            logger.warning(f"No training data found for {dataset_name}")
    
    return training_data

def calculate_confidence_band(grouped_data: pd.DataFrame, epsilon_col: str, accuracy_col: str, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate mean and confidence bands for accuracy vs epsilon"""
    
    # Group by epsilon values (or create bins if continuous)
    epsilon_values = sorted(grouped_data[epsilon_col].unique())
    
    means = []
    ci_lowers = []
    ci_uppers = []
    valid_epsilons = []
    
    for eps in epsilon_values:
        eps_data = grouped_data[grouped_data[epsilon_col] == eps][accuracy_col]
        
        if len(eps_data) > 0:
            mean_acc = eps_data.mean()
            
            if len(eps_data) > 1:
                # Calculate confidence interval
                sem = stats.sem(eps_data)
                t_val = stats.t.ppf((1 + confidence) / 2, len(eps_data) - 1)
                margin_error = t_val * sem
                ci_lower = mean_acc - margin_error
                ci_upper = mean_acc + margin_error
            else:
                # Only one data point
                ci_lower = mean_acc
                ci_upper = mean_acc
            
            means.append(mean_acc)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
            valid_epsilons.append(eps)
    
    return np.array(valid_epsilons), np.array(means), np.array(ci_lowers), np.array(ci_uppers)

def plot_training_progression(training_data: Dict[str, List[pd.DataFrame]], output_dir: str, confidence: float = 0.95, use_ema: bool = True, sota_dir: str = "data/sota"):
    """Plot training progression for each dataset: accuracy vs epsilon with confidence intervals and SOTA comparisons"""
    logger = setup_logging()
    
    accuracy_col = 'ema_test_accuracy' if use_ema else 'test_accuracy'
    
    # Define colors for SOTA papers
    sota_colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive']
    sota_markers = ['s', '^', 'D', 'v', '<', '>', 'p', '*']
    
    for dataset_name, dataset_runs in training_data.items():
        if not dataset_runs:
            continue
        
        logger.info(f"Creating training progression plot for {dataset_name}")
        
        # Combine all runs for this dataset
        all_data = pd.concat(dataset_runs, ignore_index=True)
        
        # Check if we have the required columns
        if accuracy_col not in all_data.columns:
            if use_ema:
                logger.warning(f"No EMA data found for {dataset_name}, falling back to regular accuracy")
                accuracy_col = 'test_accuracy'
            if accuracy_col not in all_data.columns:
                logger.error(f"No accuracy data found for {dataset_name}")
                continue
        
        plt.figure(figsize=(12, 8))
        
        # Calculate and plot mean with confidence interval (our method)
        epsilon_vals, mean_accs, ci_lowers, ci_uppers = calculate_confidence_band(
            all_data, 'epsilon', accuracy_col, confidence
        )
        
        if len(epsilon_vals) > 0:
            # Convert to percentages for better comparison
            mean_accs_pct = mean_accs * 100
            ci_lowers_pct = ci_lowers * 100
            ci_uppers_pct = ci_uppers * 100
            
            # Plot mean line (our method) - NO individual seed lines
            plt.plot(epsilon_vals, mean_accs_pct, 'o-', linewidth=3, markersize=8, 
                    color='darkblue', label=f'Our Method (n={len(dataset_runs)})', zorder=10)
            
            # Plot confidence interval as shaded area
            plt.fill_between(epsilon_vals, ci_lowers_pct, ci_uppers_pct, 
                           alpha=0.8, color='lightgray', 
                           label=f'{int(confidence*100)}% Confidence Interval', zorder=5)
        
        # Load and plot SOTA data
        sota_data = load_sota_data(dataset_name, sota_dir)
        
        if sota_data:
            logger.info(f"Found SOTA data for {dataset_name}: {len(sota_data)} papers")
            
            for i, paper_data in enumerate(sota_data):
                epsilons = paper_data['coordinate'][0]
                accuracies = paper_data['coordinate'][1]
                legend = paper_data['legend']
                
                color = sota_colors[i % len(sota_colors)]
                marker = sota_markers[i % len(sota_markers)]
                
                plt.scatter(epsilons, accuracies, 
                          color=color, marker=marker, s=100, alpha=0.8,
                          label=legend, zorder=15, edgecolors='black', linewidth=1)
                
                # Connect points if multiple points for same paper
                if len(epsilons) > 1:
                    plt.plot(epsilons, accuracies, 
                           color=color, linestyle='--', alpha=0.6, linewidth=2)
        else:
            logger.warning(f"No SOTA data found for {dataset_name} in {sota_dir}")
            logger.info(f"Looking for file: {os.path.join(sota_dir, f'{dataset_name}.txt')}")
        
        # Add statistics text box
        # if len(epsilon_vals) > 0:
        #     final_mean_acc = mean_accs_pct[-1] if len(mean_accs_pct) > 0 else 0
        #     final_epsilon = epsilon_vals[-1] if len(epsilon_vals) > 0 else 0
        #     max_acc = np.max(mean_accs_pct) if len(mean_accs_pct) > 0 else 0
            
        #     stats_text = f'Our Results:\nFinal: {final_mean_acc:.1f}% @ ε={final_epsilon:.1f}\nMax: {max_acc:.1f}%\nRuns: {len(dataset_runs)}'
        #     plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
        #             fontsize=11, verticalalignment='top',
        #             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Formatting
        plt.xlabel('Privacy Loss (ε)', fontsize=18, fontweight='bold')
        plt.ylabel(f'Test Accuracy (%)', fontsize=18, fontweight='bold')
        # plt.title(f'{dataset_name.upper()}: Privacy-Accuracy Tradeoff\n{"EMA " if use_ema else ""}Accuracy vs Privacy Loss with SOTA Comparison', 
        #          fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # Legend with better positioning
        plt.legend(fontsize=17, loc='best', framealpha=0.9)
        
        # Set reasonable axis limits
        if len(all_data) > 0:
            y_min = max(0, (all_data[accuracy_col].min() * 100) - 2)
            y_max = min(100, (all_data[accuracy_col].max() * 100) + 2)
            
            # Adjust y_min and y_max to include SOTA data
            if sota_data:
                all_sota_accs = []
                for paper_data in sota_data:
                    all_sota_accs.extend(paper_data['coordinate'][1])
                if all_sota_accs:
                    y_min = min(y_min, min(all_sota_accs) - 2)
                    y_max = max(y_max, max(all_sota_accs) + 2)
            
            plt.ylim(y_min, y_max)
            
            x_min = max(0, all_data['epsilon'].min() - 0.2)
            x_max = all_data['epsilon'].max() + 0.5
            
            # Adjust x limits to include SOTA data
            if sota_data:
                all_sota_eps = []
                for paper_data in sota_data:
                    all_sota_eps.extend(paper_data['coordinate'][0])
                if all_sota_eps:
                    x_min = min(x_min, min(all_sota_eps) - 0.2)
                    x_max = max(x_max, max(all_sota_eps) + 0.5)
            
            plt.xlim(x_min, x_max)
        
        # Add a subtle background
        plt.gca().patch.set_facecolor('white')
        plt.gca().patch.set_alpha(0.9)
        
        # Save plot
        suffix = "_ema" if use_ema else ""
        filename = f'{dataset_name}_training_progression{suffix}.png'
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved plot: {filepath}")
        plt.show()
        plt.close()

def generate_training_summary(training_data: Dict[str, List[pd.DataFrame]], save_path: str = None):
    """Generate summary statistics for training progression"""
    summary_stats = []
    
    for dataset_name, dataset_runs in training_data.items():
        if not dataset_runs:
            continue
        
        all_data = pd.concat(dataset_runs, ignore_index=True)
        
        # Overall statistics
        stats = {
            'Dataset': dataset_name.upper(),
            'Number of Runs': len(dataset_runs),
            'Total Epochs': len(all_data),
            'Avg Epochs per Run': len(all_data) / len(dataset_runs),
            'Final Epsilon (Mean)': all_data.groupby('seed')['epsilon'].last().mean(),
            'Final Epsilon (Std)': all_data.groupby('seed')['epsilon'].last().std(),
            'Final Accuracy % (Mean)': all_data.groupby('seed')['test_accuracy'].last().mean() * 100,
            'Final Accuracy % (Std)': all_data.groupby('seed')['test_accuracy'].last().std() * 100,
            'Max Accuracy % (Mean)': all_data.groupby('seed')['test_accuracy'].max().mean() * 100,
            'Max Accuracy % (Std)': all_data.groupby('seed')['test_accuracy'].max().std() * 100,
        }
        
        # Add EMA stats if available
        if 'ema_test_accuracy' in all_data.columns:
            stats.update({
                'Final EMA Accuracy % (Mean)': all_data.groupby('seed')['ema_test_accuracy'].last().mean() * 100,
                'Final EMA Accuracy % (Std)': all_data.groupby('seed')['ema_test_accuracy'].last().std() * 100,
                'Max EMA Accuracy % (Mean)': all_data.groupby('seed')['ema_test_accuracy'].max().mean() * 100,
                'Max EMA Accuracy % (Std)': all_data.groupby('seed')['ema_test_accuracy'].max().std() * 100,
            })
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    if not summary_df.empty:
        # Round numeric columns
        numeric_cols = [col for col in summary_df.columns if '(' in col and col != 'Dataset']
        summary_df[numeric_cols] = summary_df[numeric_cols].round(2)
        
        print("\n" + "="*120)
        print("TRAINING PROGRESSION SUMMARY")
        print("="*120)
        print(summary_df.to_string(index=False))
        
        if save_path:
            summary_df.to_csv(save_path, index=False)
            print(f"\nTraining summary saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot training progression: accuracy vs epsilon with confidence intervals and SOTA comparison')
    parser.add_argument('--results_dir', default='results/experiments',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', default='results/plots',
                       help='Directory to save plots')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level for intervals (default: 0.95)')
    parser.add_argument('--use_ema', action='store_true',
                       help='Use EMA accuracy instead of regular accuracy')
    parser.add_argument('--sota_dir', default='data/sota',
                       help='Directory containing SOTA data files (default: data/sota)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect training data
    training_data = collect_training_data(args.results_dir)
    
    if not training_data:
        print("No training data found!")
        return
    
    logger = setup_logging()
    total_runs = sum(len(runs) for runs in training_data.values())
    logger.info(f"Found training data from {total_runs} runs across {len(training_data)} datasets")
    
    for dataset, runs in training_data.items():
        total_epochs = sum(len(run) for run in runs)
        logger.info(f"  {dataset}: {len(runs)} runs, {total_epochs} total epochs")
    
    # Generate plots
    plot_training_progression(training_data, args.output_dir, args.confidence, args.use_ema, args.sota_dir)
    
    # Generate summary
    generate_training_summary(training_data, os.path.join(args.output_dir, 'training_progression_summary.csv'))
    
    print(f"\nTraining progression plots with SOTA comparison and {int(args.confidence*100)}% confidence intervals saved to {args.output_dir}/")

if __name__ == "__main__":
    main()