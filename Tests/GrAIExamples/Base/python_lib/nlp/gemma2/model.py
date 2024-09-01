import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TransformerArgs:
    """
    Transformer parameters.

    Parameters
    ----------
    dim: int
        Base hidden dimension.
    n_layers: int
        Number of Transformer blocks.
    head_dim:
        Hidden dimension of each attention head.
    hidden_dim:
        Hidden dimension of the feed forward blocks.
    n_heads: int
        Number of heads for the queries.
    n_kv_heads: int
        Number of heads for keys and values.
    norm_eps: float
        Used to avoid division by 0 during normalization.
    vocab_size: int
        Vocabulary size.
    rope_theta: float
        Coefficient used to initialize rotation matrix.
    """
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    attn_logit_softcapping: float
    final_logit_softcapping: float
    rope_theta: float = 10000


class RMSNorm(torch.nn.Module):
    """
    Root mean squared norm.

    Parameters
    ----------
    dims: int
        Embedding dimension.
    eps: float
        Epsilon value to avoid 0 division.
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dims))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.

        Returns
        -------
        _: torch.Tensor
            The output tensor.
        """
        output = self._norm(x.float())
        output = output * (1 + self.weight.float())
        return output.type_as(x)


class Attention(torch.nn.Module):
    """
    Module that can handle contextual information thanks to attention.

    Parameters
    ----------
    args: TransformerArgs
        Model parameters.
    """

    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.q_proj = torch.nn.Linear(
            args.dim, args.n_heads * args.head_dim, bias=False
        )
        self.k_proj = torch.nn.Linear(
            args.dim, args.n_kv_heads * args.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            args.dim, args.n_kv_heads * args.head_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(
            args.n_heads * args.head_dim, args.dim, bias=False
        )

    @staticmethod
    def create_additive_causal_mask(
        context_len: int, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Create causal mask.

        Parameters
        ---------
        context_len: int
            Context length.
        dtype: torch.dtype
            Precision type.

        Returns
        -------
        mask: torch.Tensor
            The causal mask.
        """
        indices = torch.arange(context_len)
        mask = torch.tensor(indices[:, None] < indices[None])
        # usually inf but 1e9 is as good and softmax(full(1e9)) != nan
        # TODO: Should replace this with finfo(dtype).min
        mask = mask.type(dtype) * -1e9
        return mask

    @staticmethod
    def create_rotation_matrix(
        positions: torch.Tensor,
        embedding_dim: int,
        rope_theta: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate the rotary matrix for RoPE.

        Parameters
        ----------
        positions: torch.Tensor
            Tensor containing the different indices of the sequential axis
            to take into account for positional encoding.
        embedding_dim: int
            Embedding dimension.
        rope_theta: float
            RoPE theta.
        device: torch.device
            Device on which the matrix is to be loaded.

        Returns
        -------
        R: torch.Tensor
            The rotary matrix of dimension
            (len(positions), embedding_dim, embedding_dim).
        """
        R = torch.zeros(
            (len(positions), embedding_dim, embedding_dim),
            requires_grad=False,
            device=device,
        )

        slice_i = torch.arange(0, embedding_dim // 2, device=device)
        theta = rope_theta ** (-2.0 * (slice_i.float()) / embedding_dim)
        m_theta = positions * theta

        cos_values = torch.cos(m_theta)
        sin_values = torch.sin(m_theta)

        R[:, 2 * slice_i, 2 * slice_i] = cos_values
        R[:, 2 * slice_i, 2 * slice_i + 1] = -sin_values
        R[:, 2 * slice_i + 1, 2 * slice_i] = sin_values
        R[:, 2 * slice_i + 1, 2 * slice_i + 1] = cos_values
        return R

    def forward(
        self,
        x: torch.Tensor,
        rotation_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.
        rotation_matrix: torch.Tensor
            Rotation matrix used for positional encoding.
        mask: torch.Tensor
            Causal mask.
        cache: (key_cache, value_cache): (torch.Tensor, torch.Tensor)
            cache for keys and values
            for generating tokens with past context.

        Returns
        -------
        (output, (keys, values)): (torch.Tensor, (torch.Tensor, torch.Tensor))
            output: the output tensor
            (keys, values): cache for keys and values
        """
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation.
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(1, 2)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)

        def repeat(a):
            a = torch.concat([torch.unsqueeze(a, 2)] * self.repeats, dim=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache

            queries = torch.einsum("bhlj,lij->bhli", [queries, rotation_matrix])
            keys = torch.einsum("bhlj,lij->bhli", [keys, rotation_matrix])

            keys = torch.concat([key_cache, keys], dim=2)
            values = torch.concat([value_cache, values], dim=2)

        else:
            queries = torch.einsum("bhlj,lij->bhli", [queries, rotation_matrix])
            keys = torch.einsum("bhlj,lij->bhli", [keys, rotation_matrix])

        scores = torch.matmul(queries, keys.transpose(2, 3)) * self.scale
        """
        # Do not use for now.
        if self.args.attn_logit_softcapping is not None:
            scores = scores / self.args.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.args.attn_logit_softcapping
        """
        if mask is not None:
            scores += mask
        scores = torch.softmax(
            scores.type(torch.float32), dim=-1
        ).type_as(scores)

        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().reshape(B, L, -1)

        return self.o_proj(output), (keys, values)


class FeedForward(torch.nn.Module):
    """
    MLP module.

    Parameters
    ----------
    args: TransformerArgs
        Model parameters.
    """

    def __init__(self, args: TransformerArgs):
        super().__init__()

        self.gate_proj = torch.nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.up_proj = torch.nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.down_proj = torch.nn.Linear(args.hidden_dim, args.dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.

        Returns
        -------
        _: torch.Tensor
            The output tensor.
        """
        return self.down_proj(
            torch.nn.GELU(approximate="tanh")(self.gate_proj(x)) *
            self.up_proj(x)
        )


class TransformerBlock(torch.nn.Module):
    """
    Transformer module.

    Parameters
    ----------
    args: TransformerArgs
        Model parameters.
    """

    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.self_attn = Attention(args)
        self.mlp = FeedForward(args=args)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self,
        x: torch.Tensor,
        rotation_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[
            Tuple[torch.Tensor,
                  Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.
        rotation_matrix: torch.Tensor
            Rotation matrix used for positional encoding.
        mask: torch.Tensor
            Causal mask.
        cache: (key_cache, value_cache): (torch.Tensor, torch.Tensor)
            cache for keys and values
            for generating tokens with past context.

        Returns
        -------
        (output, (keys, values)): (torch.Tensor, (torch.Tensor, torch.Tensor))
            output: the output tensor
            (keys, values): cache for keys and values
        """
        r, cache = self.self_attn(
            self.input_layernorm(x),
            rotation_matrix=rotation_matrix,
            mask=mask,
            cache=cache,
        )
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = h + self.post_feedforward_layernorm(r)
        return out, cache


class Transformer(torch.nn.Module):
    """
    Transformer model.

    Parameters
    ----------
    args: TransformerArgs
        Model parameters.
    """

    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.embed_tokens = torch.nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(args=args) for _ in range(args.n_layers)
        ])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = torch.nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache=None,
        n_layers=None
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.
        cache: (key_cache, value_cache): (torch.Tensor, torch.Tensor)
            cache for keys and values
            for generating tokens with past context.
        n_layers: Int
            Modifier of the number of Transformer blocks.

        Returns
        -------
        (output, cache): (torch.Tensor, list)
            output: the output tensor
            cache: cache for keys and values for each layer
        """
        h = self.embed_tokens(x)
        normalizer = torch.tensor(h.shape[-1] ** 0.5, dtype=h.dtype)
        h = h * normalizer

        mask = None
        if h.shape[1] > 1:
            mask = Attention.create_additive_causal_mask(h.shape[1])
            mask = mask.type(h.dtype)
            mask = mask.to(h.device)

            positions = torch.arange(
                1, h.shape[1] + 1, device=h.device
            ).unsqueeze(1)

        else:
            key_cache = cache[0][0]
            positions = torch.tensor(
                [key_cache.shape[2] + 1], device=h.device
            ).unsqueeze(1)

        rotation_matrix = Attention.create_rotation_matrix(
            positions=positions,
            embedding_dim=self.args.head_dim,
            rope_theta=self.args.rope_theta,
            device=h.device,
        )

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            if n_layers is not None and e == n_layers:
                break

            h, cache[e] = layer(
                h, rotation_matrix=rotation_matrix, mask=mask, cache=cache[e]
            )

        h = self.norm(h)
        logits = self.output(h)
        """
        # Do not use for now.
        if self.args.final_logit_softcapping is not None:
            logits = logits / self.args.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.args.final_logit_softcapping
        """

        return logits, cache
