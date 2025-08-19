import functools
from jax import random
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import linen as nn
import pickle
from typing import Any, Optional
from privacy import tree_map_add_normal_noise
from augmult import apply_augmentations
from functools import partial

from typing import Any, Callable, Dict, Tuple

import numpy as np
from flax.struct import dataclass
from functools import partial

# Type aliases
PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]


class TrainStateWithBatchStats(train_state.TrainState):
    batch_stats: Any  # to track batch statistics
    noise_scale: Any  # noise multiplier / batch size
    ema_params: Optional[Any] = None  # EMA parameters
    ema_step: int = 0  # Step counter for EMA


def ema_update(
    tree_old: PyTree,
    tree_new: PyTree,
    t: jnp.ndarray,
    *,
    mu: float,
    use_warmup: bool = True,
    start_step: int = 0,
) -> PyTree:
    """Exponential Moving Averaging if t >= start_step, return tree_new otherwise."""
    # Do not average until t >= start_step
    t = jnp.maximum(t - start_step, 0.0)
    mu_effective = mu * (t >= 0)
    if use_warmup:
        mu_effective = jnp.minimum(mu_effective, (1.0 + t) / (10.0 + t))
    
    return jax.tree_util.tree_map(
        lambda old, new: mu_effective * old + (1.0 - mu_effective) * new,
        tree_old,
        tree_new,
    )


def create_train_state(rng, input_shape, model, learning_rate, noise_scale, use_ema=False):
    # Initialize the model to get the variables (params and batch_stats)
    variables = model.init(rng, jnp.ones(input_shape, model.dtype), True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)  # Extract batch_stats if available

    # Define the optimizer
    tx = optax.adam(learning_rate=learning_rate)

    # Create the TrainStateWithBatchStats
    state = TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        noise_scale=noise_scale,
        ema_params=params if use_ema else None,
        ema_step=0,
    )
    return state


def train_epoch_with_augmentation(state, train_ds, p_train_step, batch_size_per_device, rng, 
                                accumulation_steps=1, augment_params=None, k=1, ema_config=None):
    """Modified train_epoch that applies data augmentation and optionally EMA."""
    train_loss = 0
    num_batches = 0
    
    # JIT compile the appropriate training function
    if accumulation_steps == 1:
        if ema_config is not None:
            # Extract EMA parameters for JIT compilation
            ema_mu = ema_config['mu']
            ema_use_warmup = ema_config.get('use_warmup', True)
            ema_start_step = ema_config.get('start_step', 0)
            jit_train_step = jax.jit(functools.partial(train_step_original_with_ema, 
                                                     ema_mu=ema_mu, 
                                                     ema_use_warmup=ema_use_warmup,
                                                     ema_start_step=ema_start_step))
        else:
            jit_train_step = jax.jit(train_step_original)
    else:
        if ema_config is not None:
            # Extract EMA parameters for JIT compilation
            ema_mu = ema_config['mu']
            ema_use_warmup = ema_config.get('use_warmup', True)
            ema_start_step = ema_config.get('start_step', 0)
            jit_train_step = jax.jit(functools.partial(train_step_with_accumulation_jit_ema, 
                                                     num_accumulation_steps=accumulation_steps,
                                                     ema_mu=ema_mu,
                                                     ema_use_warmup=ema_use_warmup,
                                                     ema_start_step=ema_start_step))
        else:
            jit_train_step = jax.jit(functools.partial(train_step_with_accumulation_jit, 
                                                     num_accumulation_steps=accumulation_steps))
    
    for batch in train_ds:
        rng, step_rng, aug_rng = random.split(rng, 3)
        
        # Convert batch to JAX arrays
        jax_batch = {
            'image': jnp.array(batch['image']),
            'label': jnp.array(batch['label'])
        }
        
        # Apply data augmentation if specified
        if augment_params is not None:
            jax_batch = apply_augmentations(aug_rng, jax_batch, augment_params, k)
        
        state, loss = jit_train_step(state, jax_batch, step_rng)
        train_loss += loss
        num_batches += 1
            
    return state, train_loss / num_batches


def train_epoch(state, train_ds, p_train_step, batch_size_per_device, rng, 
               accumulation_steps=1, ema_config=None):
    """Modified to support gradient accumulation with JIT compilation and EMA"""
    train_loss = 0
    num_batches = 0
    
    # JIT compile the appropriate training function
    if accumulation_steps == 1:
        if ema_config is not None:
            # Extract EMA parameters for JIT compilation
            ema_mu = ema_config['mu']
            ema_use_warmup = ema_config.get('use_warmup', True)
            ema_start_step = ema_config.get('start_step', 0)
            jit_train_step = jax.jit(partial(train_step_original_with_ema, 
                                           ema_mu=ema_mu,
                                           ema_use_warmup=ema_use_warmup,
                                           ema_start_step=ema_start_step))
        else:
            jit_train_step = jax.jit(train_step_original)
    else:
        if ema_config is not None:
            # Extract EMA parameters for JIT compilation
            ema_mu = ema_config['mu']
            ema_use_warmup = ema_config.get('use_warmup', True)
            ema_start_step = ema_config.get('start_step', 0)
            jit_train_step = jax.jit(partial(train_step_with_accumulation_jit_ema, 
                                           num_accumulation_steps=accumulation_steps,
                                           ema_mu=ema_mu,
                                           ema_use_warmup=ema_use_warmup,
                                           ema_start_step=ema_start_step))
        else:
            jit_train_step = jax.jit(partial(train_step_with_accumulation_jit, 
                                           num_accumulation_steps=accumulation_steps))
    
    for batch in train_ds:
        rng, step_rng = jax.random.split(rng)
        
        # Convert batch to JAX arrays if needed
        jax_batch = {
            'image': jnp.array(batch['image']),
            'label': jnp.array(batch['label'])
        }
        
        state, loss = jit_train_step(state, jax_batch, step_rng)
        train_loss += loss
        num_batches += 1
            
    return state, train_loss / num_batches


@partial(jax.jit, static_argnums=(3, 4, 5, 6))  # num_accumulation_steps, ema_mu, ema_use_warmup, ema_start_step are static
def train_step_with_accumulation_jit_ema(state, batch, rng, num_accumulation_steps, ema_mu, ema_use_warmup, ema_start_step):
    """JIT-compiled training step with gradient accumulation and EMA"""
    
    def compute_loss_and_stats(params, micro_batch, batch_stats):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            x=micro_batch['image'],
            train=True,
            mutable=['batch_stats']
        )
        loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(micro_batch['label'], 10)))
        return loss, updates['batch_stats']

    grad_fn = jax.value_and_grad(compute_loss_and_stats, has_aux=True)
    
    # Calculate micro-batch size
    total_batch_size = batch['image'].shape[0]
    micro_batch_size = total_batch_size // num_accumulation_steps
    
    # Get initial structure by running one forward pass
    first_micro_batch = {
        'image': jax.lax.dynamic_slice(batch['image'], (0, 0, 0, 0), 
                                    (micro_batch_size, batch['image'].shape[1], 
                                     batch['image'].shape[2], batch['image'].shape[3])),
        'label': jax.lax.dynamic_slice(batch['label'], (0,), (micro_batch_size,))
    }
    (_, initial_batch_stats), initial_grads = grad_fn(state.params, first_micro_batch, state.batch_stats)
    
    # Initialize carry with proper structure - zero out gradients for accumulation
    initial_carry = (
        jax.tree.map(jnp.zeros_like, initial_grads),  # accumulated_grads
        0.0,  # total_loss
        initial_batch_stats  # global_max_norms
    )
    
    def accumulation_step(carry, i):
        accumulated_grads, total_loss, global_max_norms = carry
        
        # Create micro-batch using dynamic slicing
        start_idx = i * micro_batch_size
        micro_batch = {
            'image': jax.lax.dynamic_slice(batch['image'], (start_idx, 0, 0, 0), 
                                        (micro_batch_size, batch['image'].shape[1], 
                                         batch['image'].shape[2], batch['image'].shape[3])),
            'label': jax.lax.dynamic_slice(batch['label'], (start_idx,), (micro_batch_size,))
        }
        
        # Forward pass and gradient computation
        (loss, updated_batch_stats), grads = grad_fn(state.params, micro_batch, state.batch_stats)
        
        # Update global max norms
        new_global_max_norms = jax.tree.map(
            lambda g, n: jnp.maximum(g, n), 
            global_max_norms, 
            updated_batch_stats
        )
        
        # Accumulate gradients
        new_accumulated_grads = jax.tree.map(
            lambda x, y: x + y, 
            accumulated_grads, 
            grads
        )
        
        return (new_accumulated_grads, total_loss + loss, new_global_max_norms), None
    
    # Use scan for the accumulation loop
    (accumulated_grads, total_loss, final_batch_stats), _ = jax.lax.scan(
        accumulation_step, initial_carry, jnp.arange(num_accumulation_steps)
    )
    
    # Average the accumulated gradients
    accumulated_grads = jax.tree.map(lambda x: x / num_accumulation_steps, accumulated_grads)
    average_loss = total_loss / num_accumulation_steps
    
    # Apply gradient scaling using global max norms (vectorized)
    def scale_gradients(grads, batch_stats):
        def scale_layer_grads(layer_grads, layer_name):
            if 'kernel' not in layer_grads:
                return layer_grads
                
            # Find corresponding max norm
            scaling_factor = None
            for collection_key, collection_stats in batch_stats.items():
                if layer_name in collection_stats:
                    scaling_factor = collection_stats[layer_name]
                    break
            
            if scaling_factor is None:
                return layer_grads
            
            # Apply scaling based on layer type
            kernel_shape = layer_grads['kernel'].shape
            if len(kernel_shape) == 4:  # Conv layer
                scaling_factors = scaling_factor[:, None, :, None]
            elif len(kernel_shape) == 2:  # Dense layer  
                scaling_factors = scaling_factor
            else:
                return layer_grads
                
            scaled_kernel = layer_grads['kernel'] / scaling_factors
            return {**layer_grads, 'kernel': scaled_kernel}
        
        return jax.tree.map_with_path(
            lambda path, grads: scale_layer_grads(grads, path[0].key), 
            grads
        )
    
    scaled_grads = scale_gradients(accumulated_grads, final_batch_stats)
    
    # Add noise to the gradients
    rng, noise_rng = jax.random.split(rng)
    noisy_grads = tree_map_add_normal_noise(scaled_grads, state.noise_scale, noise_rng)
    
    # Update the state
    new_state = state.apply_gradients(grads=noisy_grads)
    new_state = new_state.replace(batch_stats=final_batch_stats)
    
    # Apply EMA update
    if state.ema_params is not None:
        new_ema_params = ema_update(
            state.ema_params,
            new_state.params,
            state.ema_step.astype(jnp.float32),
            mu=ema_mu,
            use_warmup=ema_use_warmup,
            start_step=ema_start_step
        )
        new_state = new_state.replace(ema_params=new_ema_params, ema_step=state.ema_step + 1)
    
    return new_state, average_loss


@partial(jax.jit, static_argnums=(3, 4, 5))  # ema_mu, ema_use_warmup, ema_start_step are static
def train_step_original_with_ema(state, batch, rng, ema_mu, ema_use_warmup, ema_start_step):
    """JIT-compiled original training step with EMA"""
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x=batch['image'],
            train=True,
            mutable=['batch_stats']
        )
        loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch['label'], 10)))
        return loss, updates['batch_stats']

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updated_batch_stats), grads = grad_fn(state.params)

    # Scale the gradients by the max input norm (vectorized)
    def scale_gradients(grads, batch_stats):
        def scale_layer_grads(layer_grads, layer_name):
            if 'kernel' not in layer_grads:
                return layer_grads
                
            scaling_factor = None
            for collection_key, collection_stats in batch_stats.items():
                if layer_name in collection_stats:
                    scaling_factor = collection_stats[layer_name]
                    break
            
            if scaling_factor is None:
                return layer_grads
            
            kernel_shape = layer_grads['kernel'].shape
            if len(kernel_shape) == 4:  # Conv layer
                scaling_factors = scaling_factor[:, None, :, None]
            elif len(kernel_shape) == 2:  # Dense layer
                scaling_factors = scaling_factor
            else:
                return layer_grads
                
            scaled_kernel = layer_grads['kernel'] / scaling_factors
            return {**layer_grads, 'kernel': scaled_kernel}
        
        return jax.tree.map_with_path(
            lambda path, grads: scale_layer_grads(grads, path[0].key), 
            grads
        )
    
    scaled_grads = scale_gradients(grads, updated_batch_stats)
    noisy_grads = tree_map_add_normal_noise(scaled_grads, state.noise_scale, rng)
    
    new_state = state.apply_gradients(grads=noisy_grads)
    new_state = new_state.replace(batch_stats=updated_batch_stats)
    
    # Apply EMA update
    if state.ema_params is not None:
        new_ema_params = ema_update(
            state.ema_params,
            new_state.params,
            state.ema_step.astype(jnp.float32),
            mu=ema_mu,
            use_warmup=ema_use_warmup,
            start_step=ema_start_step
        )
        new_state = new_state.replace(ema_params=new_ema_params, ema_step=state.ema_step + 1)
    
    return new_state, loss


@partial(jax.jit, static_argnums=(3,))  # num_accumulation_steps is static
def train_step_with_accumulation_jit(state, batch, rng, num_accumulation_steps):
    """JIT-compiled training step with gradient accumulation (original function kept unchanged)"""
    
    def compute_loss_and_stats(params, micro_batch, batch_stats):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            x=micro_batch['image'],
            train=True,
            mutable=['batch_stats']
        )
        loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(micro_batch['label'], 10)))
        return loss, updates['batch_stats']

    grad_fn = jax.value_and_grad(compute_loss_and_stats, has_aux=True)
    
    # Calculate micro-batch size
    total_batch_size = batch['image'].shape[0]
    micro_batch_size = total_batch_size // num_accumulation_steps
    
    # Get initial structure by running one forward pass
    first_micro_batch = {
        'image': jax.lax.dynamic_slice(batch['image'], (0, 0, 0, 0), 
                                    (micro_batch_size, batch['image'].shape[1], 
                                     batch['image'].shape[2], batch['image'].shape[3])),
        'label': jax.lax.dynamic_slice(batch['label'], (0,), (micro_batch_size,))
    }
    (_, initial_batch_stats), initial_grads = grad_fn(state.params, first_micro_batch, state.batch_stats)
    
    # Initialize carry with proper structure - zero out gradients for accumulation
    initial_carry = (
        jax.tree.map(jnp.zeros_like, initial_grads),  # accumulated_grads
        0.0,  # total_loss
        initial_batch_stats  # global_max_norms
    )
    
    def accumulation_step(carry, i):
        accumulated_grads, total_loss, global_max_norms = carry
        
        # Create micro-batch using dynamic slicing
        start_idx = i * micro_batch_size
        micro_batch = {
            'image': jax.lax.dynamic_slice(batch['image'], (start_idx, 0, 0, 0), 
                                        (micro_batch_size, batch['image'].shape[1], 
                                         batch['image'].shape[2], batch['image'].shape[3])),
            'label': jax.lax.dynamic_slice(batch['label'], (start_idx,), (micro_batch_size,))
        }
        
        # Forward pass and gradient computation
        (loss, updated_batch_stats), grads = grad_fn(state.params, micro_batch, state.batch_stats)
        
        # Update global max norms
        new_global_max_norms = jax.tree.map(
            lambda g, n: jnp.maximum(g, n), 
            global_max_norms, 
            updated_batch_stats
        )
        
        # Accumulate gradients
        new_accumulated_grads = jax.tree.map(
            lambda x, y: x + y, 
            accumulated_grads, 
            grads
        )
        
        return (new_accumulated_grads, total_loss + loss, new_global_max_norms), None
    
    # Use scan for the accumulation loop
    (accumulated_grads, total_loss, final_batch_stats), _ = jax.lax.scan(
        accumulation_step, initial_carry, jnp.arange(num_accumulation_steps)
    )
    
    # Average the accumulated gradients
    accumulated_grads = jax.tree.map(lambda x: x / num_accumulation_steps, accumulated_grads)
    average_loss = total_loss / num_accumulation_steps
    
    # Apply gradient scaling using global max norms (vectorized)
    def scale_gradients(grads, batch_stats):
        def scale_layer_grads(layer_grads, layer_name):
            if 'kernel' not in layer_grads:
                return layer_grads
                
            # Find corresponding max norm
            scaling_factor = None
            for collection_key, collection_stats in batch_stats.items():
                if layer_name in collection_stats:
                    scaling_factor = collection_stats[layer_name]
                    break
            
            if scaling_factor is None:
                return layer_grads
            
            # Apply scaling based on layer type
            kernel_shape = layer_grads['kernel'].shape
            if len(kernel_shape) == 4:  # Conv layer
                scaling_factors = scaling_factor[:, None, :, None]
            elif len(kernel_shape) == 2:  # Dense layer  
                scaling_factors = scaling_factor
            else:
                return layer_grads
                
            scaled_kernel = layer_grads['kernel'] / scaling_factors
            return {**layer_grads, 'kernel': scaled_kernel}
        
        return jax.tree.map_with_path(
            lambda path, grads: scale_layer_grads(grads, path[0].key), 
            grads
        )
    
    scaled_grads = scale_gradients(accumulated_grads, final_batch_stats)
    
    # Add noise to the gradients
    rng, noise_rng = jax.random.split(rng)
    noisy_grads = tree_map_add_normal_noise(scaled_grads, state.noise_scale, noise_rng)
    
    # Update the state
    new_state = state.apply_gradients(grads=noisy_grads)
    new_state = new_state.replace(batch_stats=final_batch_stats)
    
    return new_state, average_loss


@jax.jit
def train_step_original(state, batch, rng):
    """JIT-compiled original training step (unchanged)"""
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x=batch['image'],
            train=True,
            mutable=['batch_stats']
        )
        loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch['label'], 10)))
        return loss, updates['batch_stats']

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updated_batch_stats), grads = grad_fn(state.params)

    # Scale the gradients by the max input norm (vectorized)
    def scale_gradients(grads, batch_stats):
        def scale_layer_grads(layer_grads, layer_name):
            if 'kernel' not in layer_grads:
                return layer_grads
                
            scaling_factor = None
            for collection_key, collection_stats in batch_stats.items():
                if layer_name in collection_stats:
                    scaling_factor = collection_stats[layer_name]
                    break
            
            if scaling_factor is None:
                return layer_grads
            
            kernel_shape = layer_grads['kernel'].shape
            if len(kernel_shape) == 4:  # Conv layer
                scaling_factors = scaling_factor[:, None, :, None]
            elif len(kernel_shape) == 2:  # Dense layer
                scaling_factors = scaling_factor
            else:
                return layer_grads
                
            scaled_kernel = layer_grads['kernel'] / scaling_factors
            return {**layer_grads, 'kernel': scaled_kernel}
        
        return jax.tree.map_with_path(
            lambda path, grads: scale_layer_grads(grads, path[0].key), 
            grads
        )
    
    scaled_grads = scale_gradients(grads, updated_batch_stats)
    noisy_grads = tree_map_add_normal_noise(scaled_grads, state.noise_scale, rng)
    
    new_state = state.apply_gradients(grads=noisy_grads)
    new_state = new_state.replace(batch_stats=updated_batch_stats)
    return new_state, loss


# Keep original train_step for backward compatibility (non-JIT version)
def train_step(state, batch, rng):
    """Original training step - kept for backward compatibility"""
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x=batch['image'],
            train=True,
            mutable=['batch_stats']
        )
        
        loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch['label'], 10)))
        # return loss, (logits, updates['batch_stats'])
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updated_batch_stats)), grads = grad_fn(state.params)

    # Scale the gradients by the max input norm
    for key_grad in grads.keys():
        for key in updated_batch_stats.keys():
            if 'MaxInputNormLayer' in key and key_grad in updated_batch_stats[key].keys():
                if 'Conv' in key_grad:
                    scaling_factors = updated_batch_stats[key][key_grad][:, None, :, None]
                elif 'Dense' in key_grad:
                    scaling_factors = updated_batch_stats[key][key_grad]
                else:
                    raise ValueError(f"Unknown layer type in {key_grad}")
                grads[key_grad]['kernel'] = grads[key_grad]['kernel'] / scaling_factors
    
    noisy_grads = tree_map_add_normal_noise(grads, state.noise_scale, rng)
    state = state.apply_gradients(grads=noisy_grads)
    state = state.replace(batch_stats=updated_batch_stats)
    return state, loss


@partial(jax.jit, static_argnums=(2,))  # use_ema_params is static
def eval_step(state, batch, use_ema_params=False):
    """JIT-compiled evaluation step with option to use EMA parameters"""
    params_to_use = state.ema_params if use_ema_params and state.ema_params is not None else state.params
    
    logits = state.apply_fn(
        {'params': params_to_use, 'batch_stats': state.batch_stats},
        batch['image'],
        train=False
    )
    loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(batch['label'], 10)))
    return loss, logits


def eval_model(state, test_ds, use_ema_params=False):
    """Evaluation with JIT-compiled step and optional EMA parameters"""
    test_loss = 0
    correct = 0
    count = 0
    
    for batch in test_ds:
        # Convert to JAX arrays
        jax_batch = {
            'image': jnp.array(batch['image']),
            'label': jnp.array(batch['label'])
        }
        
        loss, logits = eval_step(state, jax_batch, use_ema_params)
        test_loss += loss
        correct += jnp.sum(jnp.argmax(logits, axis=-1) == jax_batch['label'])
        count += len(jax_batch['label'])
        
    test_loss /= len(test_ds)
    accuracy = correct / count
    return test_loss, accuracy


def save_model(state, filepath):
    """Save only the serializable parts of the state object to a file."""
    serializable_state = {
        'params': state.params,
        'opt_state': state.opt_state,
        'ema_params': state.ema_params,
        'ema_step': state.ema_step,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(serializable_state, f)
    return state


def load_model(state, path):
    try:
        with open(path, 'rb') as f:
            loaded_state = pickle.load(f)
        
        # Update state with loaded values, handling both old and new format
        new_state = state.replace(
            params=loaded_state['params'],
            opt_state=loaded_state['opt_state'],
            ema_params=loaded_state.get('ema_params', None),
            ema_step=loaded_state.get('ema_step', 0),
        )
        print(f'Successfully loaded model from {path}')
        return new_state
    except FileNotFoundError:
        print(f'No saved model found at {path}, starting fresh.')
        return state
