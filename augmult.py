import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Optional, Tuple
import functools


def random_crop(key: jax.Array, image: jax.Array, crop_size: Tuple[int, int], padding: int = 4) -> jax.Array:
    """Random crop with padding."""
    # Add padding
    if len(image.shape) == 3:  # Single image (H, W, C)
        padded = jnp.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    else:  # Batch of images (B, H, W, C)
        padded = jnp.pad(image, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='reflect')
    
    # Get padded dimensions
    if len(image.shape) == 3:
        padded_h, padded_w = padded.shape[:2]
        crop_h, crop_w = crop_size
        
        # Random crop coordinates
        max_y = padded_h - crop_h
        max_x = padded_w - crop_w
        y = random.randint(key, (), 0, max_y + 1)
        x = random.randint(random.split(key)[1], (), 0, max_x + 1)
        
        return jax.lax.dynamic_slice(padded, (y, x, 0), (crop_h, crop_w, padded.shape[-1]))
    else:
        batch_size, padded_h, padded_w = padded.shape[:3]
        crop_h, crop_w = crop_size
        
        # Random crop coordinates for each image in batch
        max_y = padded_h - crop_h
        max_x = padded_w - crop_w
        keys = random.split(key, batch_size * 2)
        y_coords = random.randint(keys[:batch_size], (batch_size,), 0, max_y + 1)
        x_coords = random.randint(keys[batch_size:], (batch_size,), 0, max_x + 1)
        
        # Apply crop to each image
        def crop_single(i):
            return jax.lax.dynamic_slice(
                padded[i], (y_coords[i], x_coords[i], 0), (crop_h, crop_w, padded.shape[-1])
            )
        
        return jax.vmap(crop_single)(jnp.arange(batch_size))


def random_horizontal_flip(key: jax.Array, image: jax.Array, prob: float = 0.5) -> jax.Array:
    """Random horizontal flip."""
    if len(image.shape) == 3:  # Single image
        should_flip = random.bernoulli(key, prob)
        return jax.lax.cond(should_flip, lambda x: jnp.fliplr(x), lambda x: x, image)
    else:  # Batch of images
        batch_size = image.shape[0]
        keys = random.split(key, batch_size)
        should_flip = random.bernoulli(keys, prob)
        
        def maybe_flip(img, flip):
            return jax.lax.cond(flip, lambda x: jnp.fliplr(x), lambda x: x, img)
        
        return jax.vmap(maybe_flip)(image, should_flip)


def random_brightness(key: jax.Array, image: jax.Array, max_delta: float = 0.2) -> jax.Array:
    """Random brightness adjustment."""
    delta = random.uniform(key, (), minval=-max_delta, maxval=max_delta)
    return jnp.clip(image + delta, 0.0, 1.0)


def random_contrast(key: jax.Array, image: jax.Array, lower: float = 0.8, upper: float = 1.2) -> jax.Array:
    """Random contrast adjustment."""
    contrast_factor = random.uniform(key, (), minval=lower, maxval=upper)
    # Convert to grayscale mean for contrast adjustment
    if len(image.shape) == 3:
        mean = jnp.mean(image, axis=(0, 1), keepdims=True)
    else:  # Batch
        mean = jnp.mean(image, axis=(1, 2), keepdims=True)
    
    contrasted = (image - mean) * contrast_factor + mean
    return jnp.clip(contrasted, 0.0, 1.0)


def random_saturation(key: jax.Array, image: jax.Array, lower: float = 0.8, upper: float = 1.2) -> jax.Array:
    """Random saturation adjustment."""
    if image.shape[-1] != 3:  # Only works for RGB images
        return image
    
    saturation_factor = random.uniform(key, (), minval=lower, maxval=upper)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = jnp.dot(image, jnp.array([0.299, 0.587, 0.114]))
        gray = jnp.expand_dims(gray, axis=-1)
    else:  # Batch
        gray = jnp.dot(image, jnp.array([0.299, 0.587, 0.114]))
        gray = jnp.expand_dims(gray, axis=-1)
    
    saturated = gray + (image - gray) * saturation_factor
    return jnp.clip(saturated, 0.0, 1.0)


def random_hue(key: jax.Array, image: jax.Array, max_delta: float = 0.1) -> jax.Array:
    """Random hue adjustment (simplified version)."""
    if image.shape[-1] != 3:  # Only works for RGB images
        return image
    
    # Simple hue shift by rotating RGB channels
    delta = random.uniform(key, (), minval=-max_delta, maxval=max_delta)
    
    # Create rotation matrix for hue shift (simplified)
    cos_delta = jnp.cos(delta * 2 * jnp.pi)
    sin_delta = jnp.sin(delta * 2 * jnp.pi)
    
    # Simplified hue rotation (not perfect but works reasonably well)
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    new_r = r * cos_delta - g * sin_delta
    new_g = r * sin_delta + g * cos_delta
    new_b = b  # Keep blue channel as is for simplicity
    
    hue_shifted = jnp.stack([new_r, new_g, new_b], axis=-1)
    return jnp.clip(hue_shifted, 0.0, 1.0)


def apply_augmentations_compiled(
    key: jax.Array,
    image: jax.Array,
    label: jax.Array,
    k: int,
    # Static parameters - these will be baked into the compiled function
    do_crop: bool,
    crop_height: int,
    crop_width: int,
    crop_padding: int,
    do_flip: bool,
    flip_prob: float,
    do_brightness: bool,
    brightness_delta: float,
    do_contrast: bool,
    contrast_lower: float,
    contrast_upper: float,
    do_saturation: bool,
    saturation_lower: float,
    saturation_upper: float,
    do_hue: bool,
    hue_delta: float
) -> Tuple[jax.Array, jax.Array]:
    """Compiled version of augmentations with static parameters."""
    
    original_batch_size = image.shape[0]
    
    def augment_single_datapoint(datapoint_key, single_image):
        """Apply augmentations to a single datapoint k times."""
        
        # Split key for k augmentations, each needing 6 sub-keys
        aug_keys = random.split(datapoint_key, k)
        
        def single_augmentation(aug_key):
            """Apply one set of augmentations to the image."""
            # Split keys for different augmentations
            keys = random.split(aug_key, 6)
            
            aug_image = single_image
            
            # Apply augmentations - using jax.lax.cond for conditional execution
            aug_image = jax.lax.cond(
                do_crop,
                lambda img: random_crop(keys[0], img, (crop_height, crop_width), crop_padding),
                lambda img: img,
                aug_image
            )
            
            aug_image = jax.lax.cond(
                do_flip,
                lambda img: random_horizontal_flip(keys[1], img, flip_prob),
                lambda img: img,
                aug_image
            )
            
            aug_image = jax.lax.cond(
                do_brightness,
                lambda img: random_brightness(keys[2], img, brightness_delta),
                lambda img: img,
                aug_image
            )
            
            aug_image = jax.lax.cond(
                do_contrast,
                lambda img: random_contrast(keys[3], img, contrast_lower, contrast_upper),
                lambda img: img,
                aug_image
            )
            
            aug_image = jax.lax.cond(
                do_saturation,
                lambda img: random_saturation(keys[4], img, saturation_lower, saturation_upper),
                lambda img: img,
                aug_image
            )
            
            aug_image = jax.lax.cond(
                do_hue,
                lambda img: random_hue(keys[5], img, hue_delta),
                lambda img: img,
                aug_image
            )
            
            return aug_image
        
        # Apply k augmentations to this datapoint
        return jax.vmap(single_augmentation)(aug_keys)
    
    # Split keys for each datapoint in the batch
    datapoint_keys = random.split(key, original_batch_size)
    
    # Apply k augmentations to each datapoint
    # Result shape: (batch_size, k, H, W, C)
    augmented_images = jax.vmap(augment_single_datapoint)(datapoint_keys, image)
    
    # Reshape to (batch_size * k, H, W, C)
    final_batch_size = original_batch_size * k
    augmented_images = augmented_images.reshape((final_batch_size,) + augmented_images.shape[2:])
    
    # Repeat labels k times for each original datapoint
    repeated_labels = jnp.repeat(label, k, axis=0)
    
    return augmented_images, repeated_labels


def apply_augmentations(
    key: jax.Array,
    batch: Dict[str, jax.Array],
    augment_params: Dict[str, Any],
    k: int = 1
) -> Dict[str, jax.Array]:
    """Apply k augmentations to each datapoint in a batch.
    
    Args:
        key: JAX random key
        batch: Input batch with 'image' and 'label' keys
        augment_params: Augmentation parameters
        k: Number of augmentations per datapoint
    
    Returns:
        Augmented batch with batch_size = original_batch_size * k
    """
    
    image = batch['image']
    label = batch['label']
    
    # Check original dtype
    original_dtype_uint8 = image.dtype == jnp.uint8
    
    # Normalize images to [0, 1] if they're in [0, 255]
    if original_dtype_uint8 or jnp.max(image) > 1.0:
        image = image.astype(jnp.float32) / 255.0
    
    # Extract parameters with defaults
    do_crop = augment_params.get('random_crop', False)
    crop_size = augment_params.get('crop_size', (32, 32))
    crop_height, crop_width = crop_size
    crop_padding = augment_params.get('crop_padding', 4)
    do_flip = augment_params.get('random_flip', False)
    flip_prob = augment_params.get('flip_prob', 0.5)
    do_brightness = augment_params.get('random_brightness', False)
    brightness_delta = augment_params.get('brightness_delta', 0.2)
    do_contrast = augment_params.get('random_contrast', False)
    contrast_range = augment_params.get('contrast_range', (0.8, 1.2))
    contrast_lower, contrast_upper = contrast_range
    do_saturation = augment_params.get('random_saturation', False)
    saturation_range = augment_params.get('saturation_range', (0.8, 1.2))
    saturation_lower, saturation_upper = saturation_range
    do_hue = augment_params.get('random_hue', False)
    hue_delta = augment_params.get('hue_delta', 0.1)
    
    # Create a JIT-compiled function with these specific parameters
    compiled_fn = jax.jit(
        apply_augmentations_compiled,
        static_argnames=[
            'k', 'do_crop', 'crop_height', 'crop_width', 'crop_padding', 'do_flip', 'flip_prob',
            'do_brightness', 'brightness_delta', 'do_contrast', 'contrast_lower', 'contrast_upper',
            'do_saturation', 'saturation_lower', 'saturation_upper', 'do_hue', 'hue_delta'
        ]
    )
    
    # Call the compiled function (always returns float32)
    augmented_images, repeated_labels = compiled_fn(
        key, image, label, k,
        do_crop, crop_height, crop_width, crop_padding, do_flip, flip_prob,
        do_brightness, brightness_delta, do_contrast, contrast_lower, contrast_upper,
        do_saturation, saturation_lower, saturation_upper, do_hue, hue_delta
    )
    
    # Convert back to original dtype/range if needed (outside of JIT)
    if original_dtype_uint8:
        augmented_images = (augmented_images * 255.0).astype(jnp.uint8)
    
    return {'image': augmented_images, 'label': repeated_labels}


def get_default_augment_params(dataset_name: str) -> Dict[str, Any]:
    """Get default augmentation parameters for different datasets."""
    
    if dataset_name in ['mnist', 'fashion_mnist']:
        # More conservative augmentations for grayscale datasets
        return {
            'random_crop': True,
            'crop_size': (28, 28),
            'crop_padding': 2,
            'random_flip': True,
            'flip_prob': 0.5,
            'random_brightness': True,
            'brightness_delta': 0.1,
            'random_contrast': True,
            'contrast_range': (0.9, 1.1),
            'random_saturation': False,  # Not useful for grayscale
            'random_hue': False,  # Not useful for grayscale
        }
    
    elif dataset_name in ['cifar10', 'cifar100']:
        # Standard augmentations for color datasets
        return {
            'random_crop': True,
            'crop_size': (32, 32),
            'crop_padding': 4,
            'random_flip': True,
            'flip_prob': 0.5,
            'random_brightness': True,
            'brightness_delta': 0.2,
            'random_contrast': True,
            'contrast_range': (0.8, 1.2),
            'random_saturation': True,
            'saturation_range': (0.8, 1.2),
            'random_hue': True,
            'hue_delta': 0.1,
        }
    
    else:
        # Default augmentations
        return {
            'random_crop': True,
            'crop_size': (32, 32),
            'crop_padding': 4,
            'random_flip': True,
            'flip_prob': 0.5,
            'random_brightness': False,
            'random_contrast': False,
            'random_saturation': False,
            'random_hue': False,
        }
