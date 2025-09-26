"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        weights = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        weights = tensor_from_numpy(weights, backend=backend)
        self.weights = Parameter(weights)
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        x_one_hot = one_hot(x, self.num_embeddings)
        x_flat = x_one_hot.view(bs * seq_len, self.num_embeddings)
        W = self.weights.value.view(self.num_embeddings, self.embedding_dim)
        xb = x_flat.view(bs * seq_len, self.num_embeddings, 1)
        Wb = W.view(1, self.num_embeddings, self.embedding_dim)
        out_flat = (xb * Wb).sum(1).view(bs * seq_len, self.embedding_dim)
        out = out_flat.view(bs, seq_len, self.embedding_dim)
        return out
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN ASSIGN3_2
        if not self.training or self.p_dropout <= 0.0:
            return x

        keep_prob = 1.0 - self.p_dropout
        mask_np = np.random.binomial(1, keep_prob, size=tuple(x.shape)).astype(np.float32)
        mask = tensor_from_numpy(mask_np, backend=x.backend if hasattr(x, 'backend') else x._tensor.backend)
        return (x * mask) / keep_prob
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        # Adapted from hw2 but with backend argument and bias boolean
        self.in_size = in_size
        self.backend = backend

        # Initialize weights/bias from Uniform(-1/sqrt(in_size), 1/sqrt(in_size))
        bound = 1.0 / (in_size ** 0.5)
        # rand() in [0,1); scale to [-bound, bound]
        W0 = (rand((in_size, out_size), backend=backend) * (2 * bound)) - bound
        self.weights = Parameter(W0)

        if bias:
            b0 = (rand((out_size,), backend=backend) * (2 * bound)) - bound
            self.bias = Parameter(b0)
        else:
            self.bias = None
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        ### BEGIN ASSIGN3_2
        in_size = self.in_size
        out_size = self.out_size
        shape = x.shape

        flat_batch = 1
        for d in shape[:-1]:
            flat_batch *= d

        x2 = x.view(flat_batch, in_size)
        W  = self.weights.value.view(in_size, out_size)

        x_b = x2.view(flat_batch, in_size, 1)
        W_b = W.view(1, in_size, out_size)
        out2d = (x_b * W_b).sum(1).view(flat_batch, out_size)

        if len(shape) == 2:
            out = out2d.view(shape[0], out_size)
        else:
            out = out2d.view(*shape[:-1], out_size)

        if self.bias is not None:
            lead_ones = [1] * (len(shape) - 1)
            b = self.bias.value.view(*(lead_ones + [out_size]))
            out = out + b

        return out
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        self.weights = Parameter(ones((dim,), backend=backend))
        self.bias = Parameter(zeros((dim,), backend=backend))
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        mean = x.mean(dim=1).view(batch, 1)
        var = ((x - mean) ** 2).mean(dim=1).view(batch, 1)
        x_hat = (x - mean) / (var + self.eps) ** 0.5
        out = x_hat * self.weights.value + self.bias.value
        return out

        ### END ASSIGN3_2
