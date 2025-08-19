#!/usr/bin/env python3
"""
Script to run training experiments with multiple random seeds for reproducibility.
"""

import os
import subprocess
import yaml
import argparse
from pathlib import Path
import logging
from datetime import datetime
import sys

def setup_logging():
    """Setup logging for the experiment runner"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment_runner.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_experiment_with_seeds(config_path, seeds, base_output_dir, python_cmd='python'):
    """Run training with multiple random seeds"""
    logger = setup_logging()
    config_name = Path(config_path).stem
    
    logger.info(f"Starting experiments for {config_name} with seeds: {seeds}")
    logger.info(f"Base output directory: {base_output_dir}")
    
    results = []
    
    for i, seed in enumerate(seeds):
        logger.info(f"Running {config_name} with seed {seed} ({i+1}/{len(seeds)})")
        
        # Create seed-specific output directory
        output_dir = f"{base_output_dir}/{config_name}/seed_{seed}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up log file for this specific run
        log_file = os.path.join(output_dir, 'train_training.log')
        
        # Run training
        cmd = [
            python_cmd, 'train.py', 
            '--config', config_path,
            '--seed', str(seed),
            '--output_dir', output_dir,
            '--log_file', log_file
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run the training process
            start_time = datetime.now()
            result = subprocess.run(
                cmd, 
                check=True, 
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Seed {seed} completed successfully in {duration:.1f}s")
            
            # Try to load the final metrics
            metrics_file = os.path.join(output_dir, 'final_metrics.yaml')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = yaml.safe_load(f)
                logger.info(f"Seed {seed} - Best accuracy: {metrics.get('best_accuracy', 'N/A'):.4f}, "
                           f"Final epsilon: {metrics.get('final_epsilon', 'N/A'):.4f}")
                results.append({
                    'seed': seed,
                    'status': 'success',
                    'duration': duration,
                    'metrics': metrics
                })
            else:
                logger.warning(f"Metrics file not found for seed {seed}")
                results.append({
                    'seed': seed,
                    'status': 'success_no_metrics',
                    'duration': duration
                })
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running seed {seed}: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            results.append({
                'seed': seed,
                'status': 'failed',
                'error': str(e)
            })
            continue
        except Exception as e:
            logger.error(f"Unexpected error running seed {seed}: {e}")
            results.append({
                'seed': seed,
                'status': 'failed',
                'error': str(e)
            })
            continue
    
    # Save summary of all runs
    summary_file = os.path.join(base_output_dir, config_name, 'experiment_summary.yaml')
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    summary = {
        'config_name': config_name,
        'config_path': str(config_path),
        'seeds': seeds,
        'total_runs': len(seeds),
        'successful_runs': len([r for r in results if r['status'] == 'success']),
        'failed_runs': len([r for r in results if r['status'] == 'failed']),
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    # Print summary
    successful_runs = [r for r in results if r['status'] == 'success']
    if successful_runs:
        accuracies = [r['metrics']['best_accuracy'] for r in successful_runs if 'metrics' in r]
        if accuracies:
            import numpy as np
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            logger.info(f"\n{'='*50}")
            logger.info(f"SUMMARY FOR {config_name.upper()}")
            logger.info(f"{'='*50}")
            logger.info(f"Successful runs: {len(successful_runs)}/{len(seeds)}")
            logger.info(f"Best accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            logger.info(f"Min accuracy: {min(accuracies):.4f}")
            logger.info(f"Max accuracy: {max(accuracies):.4f}")
            logger.info(f"Summary saved to: {summary_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run experiments with multiple random seeds')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--seeds', nargs='+', type=int, 
                       default=[42, 123, 456, 789, 999], 
                       help='Random seeds to use (default: 42 123 456 789 999)')
    parser.add_argument('--output_dir', default='results/experiments',
                       help='Base output directory (default: results/experiments)')
    parser.add_argument('--python', default='python',
                       help='Python command to use (default: python)')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    results = run_experiment_with_seeds(
        args.config, 
        args.seeds, 
        args.output_dir,
        args.python
    )
    
    # Exit with error code if any runs failed
    failed_runs = [r for r in results if r['status'] == 'failed']
    if failed_runs:
        print(f"\n{len(failed_runs)} out of {len(results)} runs failed!")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} runs completed successfully!")

if __name__ == "__main__":
    main()
