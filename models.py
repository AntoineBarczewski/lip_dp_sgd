# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import jax


""" common blocks"""


import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Sequence, Union, Optional, Any

class WSConv2D(nn.Module):
    """2D Convolution with Scaled Weight Standardization and affine gain+bias."""
    
    features: int
    kernel_size: Union[int, Sequence[int]] = 3
    strides: Union[int, Sequence[int]] = 1
    padding: Union[str, Sequence[int], Sequence[Sequence[int]]] = 'SAME'
    input_dilation: Union[int, Sequence[int]] = 1
    kernel_dilation: Union[int, Sequence[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Any = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
    bias_init: Any = nn.initializers.zeros
    
    def standardize_weight(self, weight, gain, eps=1e-4):
        """Apply scaled WS with affine gain."""
        # Compute mean and variance over spatial and input channel dimensions
        # For conv weights: (kernel_height, kernel_width, in_features, out_features)
        mean = jnp.mean(weight, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(weight, axis=(0, 1, 2), keepdims=True)
        fan_in = np.prod(weight.shape[:-1])
        
        # Manually fused normalization: (w - mean) * gain / sqrt(N * var)
        scale = jax.lax.rsqrt(jnp.maximum(var * fan_in, eps)) * gain
        shift = mean * scale
        return weight * scale - shift
    
    @nn.compact
    def __call__(self, inputs: jax.Array, eps: float = 1e-4) -> jax.Array:
        """Apply WSConv2D layer."""
        
        # Normalize kernel_size and strides to tuples
        def _normalize_axes(x, ndim):
            if isinstance(x, int):
                return (x,) * ndim
            else:
                return tuple(x)
        
        kernel_size = _normalize_axes(self.kernel_size, 2)
        strides = _normalize_axes(self.strides, 2)
        input_dilation = _normalize_axes(self.input_dilation, 2)
        kernel_dilation = _normalize_axes(self.kernel_dilation, 2)
        
        # Get input dimensions
        inputs = jnp.asarray(inputs, self.dtype)
        
        # Determine input features (assuming NHWC format)
        in_features = inputs.shape[-1] // self.feature_group_count
        
        # Define weight shape: (kernel_height, kernel_width, in_features, out_features)
        kernel_shape = kernel_size + (in_features, self.features)
        
        # Create weight parameter
        kernel = self.param('kernel',
                          self.kernel_init,
                          kernel_shape,
                          self.param_dtype)
        
        # Create gain parameter for weight standardization
        gain = self.param('gain',
                         nn.initializers.ones,
                         (self.features,),
                         self.param_dtype)
        
        # Apply weight standardization
        standardized_kernel = self.standardize_weight(kernel, gain, eps)
        
        # Apply convolution
        y = jax.lax.conv_general_dilated(
            inputs,
            standardized_kernel,
            window_strides=strides,
            padding=self.padding,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),  # Standard conv dimension numbers
            feature_group_count=self.feature_group_count,
            precision=self.precision
        )
        
        # Add bias if requested
        if self.use_bias:
            bias = self.param('bias',
                            self.bias_init,
                            (self.features,),
                            self.param_dtype)
            y = y + bias
            
        return y


"""Flax implementation of ResNet V1.5."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Tuple
from collections.abc import Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(
      self,
      x,
      train: bool = True
  ):
    residual = x
    y = MaxInputNormLayer(nn.SpectralNorm(self.conv(self.filters, (3, 3), self.strides)))(x, update_stats=train)
    y = self.norm(y)
    y = self.act(y)
    y = MaxInputNormLayer(nn.SpectralNorm(self.conv(self.filters, (3, 3))))(y, update_stats=train)
    y = self.norm(y)

    if residual.shape != y.shape:
      residual = MaxInputNormLayer(nn.SpectralNorm(self.conv(
          self.filters, (1, 1), self.strides, name='conv_proj'
      )))(residual, update_stats=train)
      residual = self.norm(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x, train: bool = True):
    residual = x
    y = MaxInputNormLayer(nn.SpectralNorm(self.conv(self.filters, (1, 1))))(x, update_stats=train)
    # y = self.norm(y)
    y = self.act(y)
    y = MaxInputNormLayer(nn.SpectralNorm(self.conv(self.filters, (3, 3), self.strides)))(y, update_stats=train)
    # y = self.norm(y)
    y = self.act(y)
    y = MaxInputNormLayer(nn.SpectralNorm(self.conv(self.filters * 4, (1, 1))))(y, update_stats=train)
    # y = self.norm(y)

    if residual.shape != y.shape:
      residual = MaxInputNormLayer(nn.SpectralNorm(self.conv(
          self.filters * 4, (1, 1), self.strides, name='conv_proj'
      )))(residual, update_stats=train)
      residual = self.norm(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1.5."""

  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = WSConv2D # nn.Conv

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = nn.GroupNorm(num_groups=16, use_bias=False, use_scale=False)


    x = MaxInputNormLayer(nn.SpectralNorm(conv(
        self.num_filters,
        (7, 7),
        (2, 2),
        padding=[(3, 3), (3, 3)],
        name='conv_init',
    )))(x, update_stats=train)
    x = norm(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act
        )(x, train)
    x = jnp.mean(x, axis=(1, 2))
    x = MaxInputNormLayer(nn.SpectralNorm(nn.Dense(self.num_classes, dtype=self.dtype)))(x, update_stats=train)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
)
ResNet101 = partial(
    ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock
)
ResNet152 = partial(
    ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock
)
ResNet200 = partial(
    ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock
)


ResNet18Local = partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock, conv=nn.ConvLocal
)


# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
_ResNet1Local = partial(
    ResNet, stage_sizes=[1], block_cls=ResNetBlock, conv=nn.ConvLocal
)

"""Models for mnist and fashion_mnist datasets."""

class CNN(nn.Module):
    num_classes: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train):
        norm = nn.GroupNorm(num_groups=16, use_bias=False, use_scale=False)
        x = MaxInputNormLayer(nn.SpectralNorm(nn.Conv(features=32, kernel_size=(3, 3))))(x, update_stats=train)
        x = norm(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = MaxInputNormLayer(nn.SpectralNorm(nn.Conv(features=64, kernel_size=(3, 3))))(x, update_stats=train)
        x = norm(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = MaxInputNormLayer(nn.SpectralNorm(nn.Dense(features=256)))(x, update_stats=train)
        x = nn.relu(x)
        x = MaxInputNormLayer(nn.SpectralNorm(nn.Dense(features=self.num_classes)))(x, update_stats=train)
        x = nn.log_softmax(x)
        return x
    

class MaxInputNormLayer(nn.Module):
  """Module wrapper for layers that update batch statistics with max input norm."""

  layer: nn.Module
  collection_name: str = 'batch_stats'

  @nn.compact
  def __call__(self, x, *args, **kwargs):
    # update batch statistics if the layer is in training mode
    if not kwargs['update_stats']:
      return self.layer(x, *args, **kwargs)
    
    # Compute the max input norm
    kernel_axis = tuple(range(1, len(x.shape) - 1))
    kernel_axis = kernel_axis if len(kernel_axis) > 0 else (1,)
    max_input_norm = jnp.max(jnp.linalg.norm(x, axis=kernel_axis), axis=0, keepdims=True)

    # Update the variable in the specified collection
    self.variable(
      self.collection_name,
      self.layer.layer_instance.name,
      lambda: max_input_norm
    ).value = max_input_norm

    # Apply the wrapped layer
    return self.layer(x, *args, **kwargs)
