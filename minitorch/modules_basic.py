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
        raise NotImplementedError
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
        raise NotImplementedError
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
        raise NotImplementedError
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
        self.in_size = in_size
        self.backend = backend
        bound = 1.0 / np.sqrt(in_size)

        weights = rand((in_size, out_size), backend=backend) * 2 * bound - bound
        self.weights = Parameter(weights)

        if bias:
            bias_data = rand((out_size,), backend=backend) * 2 * bound - bound
            self.bias = Parameter(bias_data)
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
        # Workaround for CudaKernelOps matrix_multiply bug
        # Flatten input to 2D, perform 2D-by-2D multiplication, then reshape back
        x_flat = x.contiguous().view(batch * in_size)  # Shape: (batch * in_size,)
        w_flat = self.weights.value.contiguous().view(in_size, self.out_size)  # Shape: (in_size, out_size)

        # Perform matrix multiplication one row at a time to avoid batch dimension bug
        output_rows = []
        for b in range(batch):
            # Extract one row of input: shape (in_size,)
            x_row = x_flat[b * in_size : (b + 1) * in_size].view(1, in_size)  # Shape: (1, in_size)
            # Multiply with weights: (1, in_size) @ (in_size, out_size) = (1, out_size)
            row_result = x_row @ w_flat  # Shape: (1, out_size)
            output_rows.append(row_result)

        # Stack all rows back together
        # Convert to numpy arrays and stack them
        numpy_rows = [row.to_numpy() for row in output_rows]
        stacked = np.vstack(numpy_rows)  # Shape: (batch, out_size)

        # Convert back to tensor
        output = tensor_from_numpy(stacked, backend=self.backend, requires_grad=True)

        if self.bias is not None:
            output = output + self.bias.value
        return output
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
        raise NotImplementedError
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
        raise NotImplementedError
        ### END ASSIGN3_2
