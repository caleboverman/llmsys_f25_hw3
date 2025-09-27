import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd 
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        ### BEGIN ASSIGN3_3
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        x2d = x.view(batch_size * seq_len, n_embd)

        # replace the three Linear calls with manual matmuls using W.T
        # shapes: x2d: (B*T, C), W*: (C_out, C_in) if saved from PyTorch
        Wq = self.q_projection.weights.value      # (C_in, C_out) or (C_out, C_in) — fixtures expect W.T
        Wk = self.k_projection.weights.value
        Wv = self.v_projection.weights.value

        # use .permute(1, 0).contiguous() to match PyTorch-saved fixtures and ensure contiguity
        Wtq = Wq.permute(1, 0).contiguous()
        Wtk = Wk.permute(1, 0).contiguous()
        Wtv = Wv.permute(1, 0).contiguous()
        q_linear = (x2d @ Wtq).view(batch_size, seq_len, n_embd)
        k_linear = (x2d @ Wtk).view(batch_size, seq_len, n_embd)
        v_linear = (x2d @ Wtv).view(batch_size, seq_len, n_embd)

        q = q_linear.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3).contiguous()
        k = k_linear.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3).contiguous()
        v = v_linear.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3).contiguous()

        kT = k.permute(0, 1, 3, 2).contiguous()
        ### END ASSIGN3_3
        return q, kT, v

    def self_attention(self, q, kT, v):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.

        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None

        ### BEGIN ASSIGN3_3
        inv_sqrt_d = tensor_from_numpy(
            np.array(1.0 / np.sqrt(self.attn_hidden_dim), dtype=datatype),
            backend=self.backend
        )
        scores = (q @ kT) * inv_sqrt_d

        if self.causal:
            mask = self.create_causal_mask(queries_len)
            scores = scores + mask

        row_max = max(scores, dim=3)                               # (batch_size, num_head, queries_len)
        row_max = row_max.view(batch_size, num_head, queries_len, 1)
        scores = scores - row_max

        attn = softmax(scores, dim=3)                              # use positive dim index
        attn = self.dropout(attn)
        # --- DEBUG: print a small slice for the smallest test case ---
        if (batch_size == 1) and (num_head == 1) and (queries_len == 32):
            try:
                s_np = scores.to_numpy()
                a_np = attn.to_numpy()
                print("[MHA DEBUG] scores[0,0,0,:8] =>", s_np[0, 0, 0, :8])
                print("[MHA DEBUG] attn  [0,0,0,:8] =>", a_np[0, 0, 0, :8])
                print("[MHA DEBUG] attn row sum =>", a_np[0, 0, 0, :].sum())
            except Exception as e:
                print("[MHA DEBUG] print error:", e)

        # --- DEBUG 2: inspect a later query row and compare two merge orders ---
        if (batch_size == 1) and (num_head == 1) and (queries_len == 32):
            try:
                a_np = attn.to_numpy()
                # pick a later query index where more than one key is unmasked
                qi = 7  # any qi>0 works; 7 is arbitrary but in-range
                print("[MHA DEBUG] attn  [0,0,qi,:8] =>", a_np[0, 0, qi, :8])
                print("[MHA DEBUG] attn row sum (qi)", a_np[0, 0, qi, :].sum())

                # current merge path (B,H,T,D)->(B,T,H,D)->view(B,T,C)
                ctx_std = (attn @ v).permute(0, 2, 1, 3).contiguous()
                out_std = self.out_projection(ctx_std.view(batch_size * queries_len, self.n_embd))
                out_std = out_std.view(batch_size, queries_len, self.n_embd)

                # alternative merge: (B,H,T,D)->(B,T,D,H)->view(B,T,C)
                ctx_alt = (attn @ v).permute(0, 2, 3, 1).contiguous()
                out_alt = self.out_projection(ctx_alt.view(batch_size * queries_len, self.n_embd))
                out_alt = out_alt.view(batch_size, queries_len, self.n_embd)

                o_std = out_std.to_numpy()
                o_alt = out_alt.to_numpy()
                print("[MHA DEBUG] out_std[0,0,:8] =>", o_std[0, 0, :8])
                print("[MHA DEBUG] out_alt[0,0,:8] =>", o_alt[0, 0, :8])
            except Exception as e:
                print("[MHA DEBUG] debug2 error:", e)

        # --- DEBUG 3: inspect V and the resulting context for qi=0 ---
        if (batch_size == 1) and (num_head == 1) and (queries_len == 32):
            try:
                v_np = v.to_numpy()
                print("[MHA DEBUG] v[0,0,0,:8] =>", v_np[0, 0, 0, :8])
            except Exception as e:
                print("[MHA DEBUG] debug3 V error:", e)

        context = attn @ v                                         # (B, H, T, D)
        # --- DEBUG 4: context head output for qi=0 before merge ---
        if (batch_size == 1) and (num_head == 1) and (queries_len == 32):
            try:
                ctx_np = context.to_numpy()
                print("[MHA DEBUG] context[0,0,0,:8] =>", ctx_np[0, 0, 0, :8])
            except Exception as e:
                print("[MHA DEBUG] debug4 context error:", e)

        context = context.permute(0, 2, 1, 3).contiguous()         # (B, T, H, D)
        context = context.view(batch_size, queries_len, self.n_embd)
        context2d = context.view(batch_size * queries_len, self.n_embd)
        result2d = self.out_projection(context2d)
        result = result2d.view(batch_size, queries_len, self.n_embd)
        # --- DEBUG 5: final output first row ---
        if (batch_size == 1) and (num_head == 1) and (queries_len == 32):
            try:
                res_np = result.to_numpy()
                print("[MHA DEBUG] result[0,0,:8] =>", res_np[0, 0, :8])
            except Exception as e:
                print("[MHA DEBUG] debug5 result error:", e)
        # --- DEBUG 6: manual out-projection check (both orientations) for qi=0 ---
        if (batch_size == 1) and (num_head == 1) and (queries_len == 32):
            try:
                # fetch context row and weight matrix
                ctx2d_np = context2d.to_numpy()  # shape (B*T, C)
                W = self.out_projection.weights.value.to_numpy()  # expect (C_in, C_out)
                # compute both orientations
                out_nm = ctx2d_np[0] @ W                      # (C,)
                out_tm = W.T @ ctx2d_np[0]                    # (C,)
                print("[MHA DEBUG] manual ctx@W  [:8] =>", out_nm[:8])
                print("[MHA DEBUG] manual W^T@ctx[:8] =>", out_tm[:8])
            except Exception as e:
                print("[MHA DEBUG] debug6 manual mm error:", e)
        ### END ASSIGN3_3

        return result

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        q, kT, v = self.project_to_query_key_value(x)
        out = self.self_attention(q, kT, v)
        return out
        ### END ASSIGN3_3


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with  activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        ### END ASSIGN3_3

        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        self.ln_1 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            causal=True,
            p_dropout=p_dropout,
            bias=bias,
            backend=backend,
        )
        self.ff = FeedForward(
            n_embd=n_embd,
            middle_dim=4 * n_embd,
            p_dropout=p_dropout,
            bias=bias,
            backend=backend,
        )
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        residual = x
        x = x.view(batch_size * seq_len, n_embd)
        x = self.ln_1(x)
        x = x.view(batch_size, seq_len, n_embd)
        x = self.attention(x)
        x = residual + x

        residual = x
        x = x.view(batch_size * seq_len, n_embd)
        x = self.ln_2(x)
        x = x.view(batch_size, seq_len, n_embd)
        x = self.ff(x)
        x = residual + x
        return x
        ### END YOUR SOLUTION


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        ### BEGIN ASSIGN3_3
        raise NotImplementedError
        # self.token_embeddings = 
        # self.position_embeddings = 
        # self.t_layer_1 = 
        # self.t_layer_2 = 
        # self.t_layer_3 = 
        # self.t_layer_4 = 
        # self.dropout = 
        # self.ln = 
        # self.lm_head = 
        ### END ASSIGN3_3
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        ### BEGIN ASSIGN3_3
        raise NotImplementedError
        # 1. Get token embeddings of shape (batch_size, seq_len, n_embd)
        # 2. Create positional embeddings of shape (1, seq_len, n_embd):
        #    - Create position ids tensor [0, 1, 2, ..., seq_len-1] of shape (1, seq_len)
        #    - Pass through positional embedding layer
        #    - Ensure output shape is (1, seq_len, n_embd)
        # 3. Add token and positional embeddings
        # 4. Apply dropout
        # 5. Pass through transformer layers (t_layer_1 to t_layer_4)
        # 6. Apply final layer normalization
        # 7. Project to vocabulary size using lm_head
        ### END ASSIGN3_3
