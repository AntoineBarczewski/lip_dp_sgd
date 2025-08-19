import jax
import jax.numpy as jnp
import optax
import yaml
import tensorflow_datasets as tfds
from jax import random, device_put, pmap
from flax import linen as nn
from flax.training import train_state
from utils import create_train_state, train_epoch, eval_model, save_model, load_model, train_step, train_epoch_with_augmentation
from augmult import get_default_augment_params
import time
import logging
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from models import ResNet18, ResNet34, CNN
from privacy import NoisySGD_mech
import os
import argparse
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def setup_logging(log_file='train_training.log'):
    """Setup logging with configurable log file location"""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filename=log_file, 
        filemode='a'
    )
    logger = logging.getLogger()
    return logger

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train neural networks with differential privacy')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for models and logs')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config file)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config file)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Override config with command line arguments
    if args.seed is not None:
        config['random_seed'] = args.seed
    if args.dataset is not None:
        config['dataset'] = args.dataset
    
    # Extract configuration
    dataset_name = config['dataset']
    model_save_path = config.get('model_save_path', 'best_model.pkl')
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.1)
    num_epochs = config.get('num_epochs', 1)
    noise_std = config.get('noise_std', 0.0)
    delta = float(config.get('delta', 1e-5))
    max_epsilon = config.get('max_epsilon', 10.0)
    random_seed = config.get('random_seed', 0)
    
    # New configuration parameters for gradient accumulation
    effective_batch_size = config.get('effective_batch_size', batch_size)
    accumulation_steps = max(1, effective_batch_size // batch_size)

    # Data augmentation configuration
    use_augmentation = config.get('use_augmentation', True)
    k = config.get('augment_mult', 1)
    custom_augment_params = config.get('augmentation', {})
    
    # EMA configuration
    use_ema = config.get('use_ema', False)
    ema_config = config.get('ema', {}) if use_ema else None
    eval_with_ema = ema_config.get('eval_with_ema', False) if ema_config else False
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'train_training.log')
        if args.log_file:
            log_file = args.log_file
    else:
        # Use original timestamp-based naming
        run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f'train_{dataset_name}_{run_timestamp}'
        if use_augmentation:
            folder_name += '_aug'
        if use_ema:
            folder_name += '_ema'
        output_dir = f'models_saved/{folder_name}'
        os.makedirs(output_dir, exist_ok=True)
        log_file = args.log_file or 'train_training.log'
    
    # Setup logging
    logger = setup_logging(log_file)
    
    logger.info(f'Starting train experiment with configuration: {config}')
    logger.info(f'Random seed: {random_seed}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Physical batch size: {batch_size}, Effective batch size: {effective_batch_size}, Accumulation steps: {accumulation_steps}')
    logger.info(f'Data augmentation enabled: {use_augmentation}, augmentation multiplicity: {k}')
    logger.info(f'EMA enabled: {use_ema}')
    if use_ema:
        logger.info(f'EMA configuration: {ema_config}')
    
    # Use configurable random seed
    rng = random.PRNGKey(random_seed)

    devices = jax.devices()
    num_devices = len(devices)
    batch_size_per_device = batch_size // num_devices

    ds_builder = tfds.builder(dataset_name)
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=batch_size, shuffle_files=True))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=batch_size))

    # Use effective batch size for privacy accounting
    batch_rate = effective_batch_size / len(ds_builder.as_dataset(split='train'))

    # Set up data augmentation parameters
    if use_augmentation:
        augment_params = get_default_augment_params(dataset_name)
        # Override with custom parameters if provided
        augment_params.update(custom_augment_params)
        logger.info(f'Augmentation parameters: {augment_params}')
    else:
        augment_params = None

    rng, init_rng = random.split(rng)
    if dataset_name in ['mnist', 'fashion_mnist']:
        model = CNN(num_classes=10)
        input_shape = (batch_size_per_device, 28, 28, 1)
    elif dataset_name in ['cifar10']:
        model = ResNet18(num_classes=10)
        input_shape = (batch_size_per_device, 32, 32, 3)
    
    # Scale noise by effective batch size instead of physical batch size
    noise_scale = noise_std / effective_batch_size
    state = create_train_state(init_rng, input_shape, model, learning_rate, noise_scale, use_ema=use_ema)
    
    # Load previous state if available (but only if using original path)
    if not args.output_dir:
        state = load_model(state, model_save_path)

    # Save modified config with accumulation and EMA info
    config_to_save = config.copy()
    config_to_save['effective_batch_size'] = effective_batch_size
    config_to_save['accumulation_steps'] = accumulation_steps
    config_to_save['physical_batch_size'] = batch_size
    config_to_save['output_dir'] = output_dir
    if use_augmentation:
        config_to_save['final_augmentation_params'] = augment_params
        config_to_save['final_augmentation_multiplicity'] = k
    if use_ema:
        config_to_save['final_ema_config'] = ema_config

    # Save config to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config_to_save, file)
    
    # Setup tensorboard logging
    tensorboard_dir = output_dir.replace('models_saved', 'logs') if 'models_saved' in output_dir else f'logs/{os.path.basename(output_dir)}'
    os.makedirs(os.path.dirname(tensorboard_dir), exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    best_accuracy = 0.0
    best_ema_accuracy = 0.0
    best_epsilon = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        rng, epoch_rng = random.split(rng)
        
        # Use the augmented training function with EMA support
        state, train_loss = train_epoch_with_augmentation(
            state, train_ds, train_step, batch_size_per_device, 
            epoch_rng, accumulation_steps, augment_params, k, ema_config
        )
        
        epoch_runtime = time.time() - epoch_start_time
        
        # Evaluate with regular parameters
        test_loss, test_accuracy = eval_model(state, test_ds, use_ema_params=False)
        
        # Evaluate with EMA parameters if available
        if use_ema and state.ema_params is not None:
            ema_test_loss, ema_test_accuracy = eval_model(state, test_ds, use_ema_params=True)
        else:
            ema_test_loss, ema_test_accuracy = test_loss, test_accuracy
        
        # Privacy accounting uses effective batch size
        noisysgd = NoisySGD_mech(prob=batch_rate,sigma=noise_std,niter=int(np.ceil((epoch+1)/batch_rate)))
        epsilon = noisysgd.get_approxDP(delta=delta)
        
        log_msg = f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}'
        if use_ema and state.ema_params is not None:
            log_msg += f', EMA Test Loss: {ema_test_loss:.4f}, EMA Test Accuracy: {ema_test_accuracy:.4f}'
        log_msg += f', Epsilon: {epsilon:.4f}, Runtime: {epoch_runtime:.2f}s'
        logger.info(log_msg)
        
        # Log metrics to tensorboard
        writer.add_scalar('Loss/train', np.array(train_loss.tolist()), epoch)
        writer.add_scalar('Epsilon/train', np.array(epsilon), epoch)
        writer.add_scalar('Loss/test', np.array(test_loss.tolist()), epoch)
        writer.add_scalar('Accuracy/test', np.array(test_accuracy.tolist()), epoch)
        
        if use_ema and state.ema_params is not None:
            writer.add_scalar('Loss/test_ema', np.array(ema_test_loss.tolist()), epoch)
            writer.add_scalar('Accuracy/test_ema', np.array(ema_test_accuracy.tolist()), epoch)
            writer.add_scalar('EMA/step', np.array(state.ema_step).item(), epoch)

        # Determine which accuracy to use for model saving
        primary_accuracy = ema_test_accuracy if (eval_with_ema and state.ema_params is not None) else test_accuracy
        
        # Save the model if the test accuracy improves
        if primary_accuracy > best_accuracy:
            model_path = os.path.join(output_dir, 'best_model.pkl')
            state = save_model(state, model_path)
            best_accuracy = primary_accuracy.copy()
            best_epsilon = epsilon.copy()
            model_type = "EMA" if (eval_with_ema and state.ema_params is not None) else "regular"
            logger.info(f'New best model saved with {model_type} accuracy: {primary_accuracy:.4f} and epsilon: {epsilon:.4f}')
        else:
            logger.info(f'No improvement in accuracy. Current best: {best_accuracy:.4f}')
            
        # Also track best EMA accuracy separately if using EMA
        if use_ema and state.ema_params is not None:
            if ema_test_accuracy > best_ema_accuracy:
                best_ema_accuracy = ema_test_accuracy.copy()
        
        if epsilon > max_epsilon:
            logger.info(f'Epsilon exceeded maximum limit: {epsilon:.4f} > {max_epsilon:.4f}. Stopping training.')
            break

    # Log final results
    logger.info(f'Final test accuracy: {test_accuracy:.4f}')
    logger.info(f'Final test loss: {test_loss:.4f}')
    if use_ema and state.ema_params is not None:
        logger.info(f'Final EMA test accuracy: {ema_test_accuracy:.4f}')
        logger.info(f'Final EMA test loss: {ema_test_loss:.4f}')
        logger.info(f'Best EMA accuracy achieved: {best_ema_accuracy:.4f}')
    logger.info(f'Best accuracy achieved: {best_accuracy:.4f}')
    logger.info(f'Final epsilon: {epsilon:.4f}')
    
    total_runtime = time.time() - start_time
    logger.info(f'Total training time: {total_runtime:.2f}s')

    # Rename the saved model file to include the accuracy
    best_model_path = os.path.join(output_dir, 'best_model.pkl')
    if os.path.exists(best_model_path):
        model_suffix = "ema" if (eval_with_ema and state.ema_params is not None) else "reg"
        new_model_path = os.path.join(output_dir, f'best_model_{model_suffix}_{best_accuracy:.4f}_eps={best_epsilon:.2f}.pkl')
        os.rename(best_model_path, new_model_path)
        logger.info(f'Model saved as {new_model_path}')
    else:
        logger.warning(f'Best model file not found at {best_model_path}')

    logger.info('Train experiment completed')
    writer.close()

    # Save final metrics summary for easy parsing
    final_metrics = {
        'dataset': dataset_name,
        'random_seed': random_seed,
        'final_accuracy': float(test_accuracy),
        'final_loss': float(test_loss),
        'best_accuracy': float(best_accuracy),
        'final_epsilon': float(epsilon),
        'total_runtime': total_runtime,
        'num_epochs_completed': epoch + 1,
        'use_augmentation': use_augmentation,
        'use_ema': use_ema,
        'noise_std': noise_std,
        'learning_rate': learning_rate,
        'effective_batch_size': effective_batch_size
    }
    
    if use_ema and state.ema_params is not None:
        final_metrics.update({
            'final_ema_accuracy': float(ema_test_accuracy),
            'final_ema_loss': float(ema_test_loss),
            'best_ema_accuracy': float(best_ema_accuracy)
        })
    
    # Save metrics as YAML for easy parsing
    with open(os.path.join(output_dir, 'final_metrics.yaml'), 'w') as f:
        yaml.dump(final_metrics, f)

if __name__ == '__main__':
    main()
