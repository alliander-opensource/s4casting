# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0


import torch
from pydantic import NonNegativeFloat, NonNegativeInt, confloat
from torch import nn
from torch.nn import functional as F

from s4casting.core.loss import SubsetNLLLoss
from s4casting.model._encoders import PatchDecoder, PatchEncoder
from s4casting.model._heads import GMMHead
from s4casting.model._norm import denorm, norm, norm_target
from s4casting.model._s4_kernel import Activation

LOAD_SLICE = slice(0, 1)
WEATHER_SLICE = slice(1, 5)  # TODO fix: this will break if we change the weather features


class TransformerModel(nn.Module):
    """Transformer model for time series forecasting with GMM output head."""

    def __init__(
        self,
        seq_len: NonNegativeInt = 1024,
        latent_dim: NonNegativeInt = 512,
        n_heads: NonNegativeInt = 8,
        n_layers: NonNegativeInt = 6,
        n_input_features: int = 5,
        n_out_features: NonNegativeInt = 1,
        patch_size: NonNegativeInt = 8,
        dropout: confloat(ge=0, lt=1) = 0.0,  # type: ignore
        use_cross_attention: bool = False,
        context_n_layers: int = 4,
        dtype=torch.bfloat16,
        is_causal: bool = False,
        attn_bias: bool = True,
        mlp_bias: bool = True,
        mlp_layers: NonNegativeInt = 2,
        mlp_activation: str = "gelu",
        norm_clamp: NonNegativeFloat | None = 10.0,
        norm_eps: NonNegativeFloat = 1e-5,
        loss_fn: nn.Module = SubsetNLLLoss(1, "masked"),
        output_head: nn.Module = GMMHead(256, 2, 1),
        patch_encoder: nn.Module = PatchEncoder(256, 5, 8),
        patch_decoder: nn.Module = PatchDecoder(256, 256, 8, [15], [15]),
        base_sample_interval_minutes: int = 15,
    ):
        """Initialize the TransformerModel.

        Args:
            seq_len (NonNegativeInt): Sequence length.
            latent_dim (NonNegativeInt): Embedding dimension.
            n_heads (NonNegativeInt): Number of attention heads.
            n_layers (NonNegativeInt): Number of transformer layers.
            n_input_features (NonNegativeInt): Number of input features.
            output_head: Quantile or GMM head.
            n_out_features (NonNegativeInt): Number of output features.
            patch_size (NonNegativeInt): Size of patches for patch embedding.
            dropout (confloat): Dropout probability.
            dtype: Data type for model parameters.
            is_causal (bool): Whether to use causal attention.
            attn_bias (bool): Whether to use attention bias.
            mlp_bias (bool): Whether to use MLP bias.
            mlp_layers (NonNegativeInt): Number of MLP layers.
            mlp_activation (str): Activation function for MLP.
            norm_clamp (None | NonNegativeFloat): Clamping value for normalization.
            loss_fn (nn.Module): Loss function.
            base_sample_interval_minutes (int): Base sample interval.
            context_n_layers (int): Number of layers for processing weather information.
            use_cross_attention (bool): Whether to use cross attention block.
            norm_eps (float): Epsilon for norm.
            patch_encoder (nn.Module): Patch encoder module.
            patch_decoder (nn.Module): Patch decoder module.
        """
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_input_features = n_input_features
        self.n_out_features = n_out_features
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.use_cross_attention = use_cross_attention
        self.norm_clamp = norm_clamp
        self.norm_eps = norm_eps
        self.base_sample_interval_minutes = base_sample_interval_minutes

        self.embed_dropout = nn.Dropout(dropout)
        self.loss_fn = loss_fn
        self.output_head = output_head
        self.patch_decoder = patch_decoder

        if use_cross_attention:
            self.load_patch_encoder = PatchEncoder(latent_dim, n_out_features * 2, patch_size)
            self.weather_patch_encoder = PatchEncoder(latent_dim, (n_input_features - n_out_features) * 2, patch_size)
            self.context_layers = nn.ModuleList(
                TransformerBlock(
                    n_emb=latent_dim,
                    n_heads=n_heads,
                    n_patches=self.n_patches,
                    dropout=dropout,
                    attn_bias=attn_bias,
                    is_causal=is_causal,
                    mlp_bias=mlp_bias,
                    mlp_layers=mlp_layers,
                    mlp_activation=mlp_activation,
                    norm_eps=norm_eps,
                    use_cross_attention=False,
                )
                for _ in range(context_n_layers)
            )
            self.patch_encoder = None
        else:
            self.patch_encoder = patch_encoder or PatchEncoder(latent_dim, n_input_features * 2, patch_size)
            self.context_layers = nn.ModuleList()

        self.transformer_layers = nn.ModuleList(
            TransformerBlock(
                n_emb=latent_dim,
                n_heads=n_heads,
                n_patches=self.n_patches,
                dropout=dropout,
                attn_bias=attn_bias,
                is_causal=is_causal,
                mlp_bias=mlp_bias,
                mlp_layers=mlp_layers,
                mlp_activation=mlp_activation,
                norm_eps=norm_eps,
                use_cross_attention=use_cross_attention,
            )
            for _ in range(n_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        xm: torch.Tensor,
        input_interval: int,
        output_interval: int,
        y: torch.Tensor | None = None,
        ym: torch.Tensor | None = None,
    ):
        """Run forward pass.

        Returns:
            Tuple of (predictions, loss) where loss is None if y is not provided.
        """
        x = x * xm
        x, mean_in, std_in = norm(
            x=x,
            xm=xm,
            eps=self.norm_eps,
            clamp=self.norm_clamp,
            dims=torch.arange(x.shape[-1]),
        )
        x = x * xm

        patch_interval = input_interval / self.base_sample_interval_minutes
        output_ratio = output_interval // input_interval

        if self.use_cross_attention:
            x_load = torch.cat([x[..., LOAD_SLICE], xm[..., LOAD_SLICE].float()], dim=-1)
            x_weather = torch.cat([x[..., WEATHER_SLICE], xm[..., WEATHER_SLICE].float()], dim=-1)

            x_load = self.load_patch_encoder(x_load, patch_interval, output_ratio)
            x_weather = self.weather_patch_encoder(x_weather, patch_interval, output_ratio)
            x_load = self.embed_dropout(x_load)
            x_weather = self.embed_dropout(x_weather)

            for block in self.context_layers:
                x_weather = block(x_weather)

            for block in self.transformer_layers:
                x_load = block(x_load, context=x_weather)

            x = self.patch_decoder(x_load)
        else:
            x = torch.cat([x, xm.float()], dim=-1)
            x = self.patch_encoder(x, patch_interval, output_ratio)
            x = self.embed_dropout(x)

            for block in self.transformer_layers:
                x = block(x)

            x = self.patch_decoder(x)

        x = self.output_head(x)

        loss = None
        if y is not None:
            y = norm_target(
                mean_in=mean_in[..., : self.n_out_features],
                std_in=std_in[..., : self.n_out_features],
                y=y,
                ym=ym,
            )
            loss = self.loss_fn(
                out=x, target=y, input_interval=input_interval, output_interval=output_interval, mask=ym
            )

        x = denorm(
            mean_in=mean_in[..., : self.n_out_features].unsqueeze(-1),
            std_in=std_in[..., : self.n_out_features].unsqueeze(-1),
            x=x,
            output_type=type(self.output_head),
            xm=xm[..., : self.n_out_features],
        )
        return x, loss


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block, optionally with cross-attention to weather tokens."""

    def __init__(
        self,
        *,
        n_emb: int,
        n_heads: int,
        n_patches: int,
        dropout: float = 0.0,
        attn_bias: bool = True,
        is_causal: bool = False,
        mlp_bias: bool = True,
        mlp_layers: int = 2,
        mlp_activation: str = "gelu",
        norm_eps: float = 1e-5,
        use_cross_attention: bool = False,
    ):
        """Initialize pre-norm transformer block."""
        super().__init__()
        self.use_cross_attention = use_cross_attention

        self.norm_self = nn.LayerNorm(n_emb, eps=norm_eps)
        self.self_attention = AttentionBlock(
            n_emb=n_emb,
            n_heads=n_heads,
            n_patches=n_patches,
            bias=attn_bias,
            is_causal=is_causal,
            attn_dropout=dropout,
            res_dropout=dropout,
        )

        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(n_emb, eps=norm_eps)
            self.cross_attention = AttentionBlock(
                n_emb=n_emb,
                n_heads=n_heads,
                n_patches=n_patches,
                bias=attn_bias,
                is_causal=is_causal,
                attn_dropout=dropout,
                res_dropout=dropout,
            )
            self.cross_gate = CrossAttentionGate(n_emb)

        self.norm_mlp = nn.LayerNorm(n_emb, eps=norm_eps)
        self.mlp = MLP(
            n_interface=n_emb, n_layers=mlp_layers, dropout=dropout, bias=mlp_bias, activation=mlp_activation
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None):
        """Apply self-attention, optional cross-attention, and MLP.

        Returns:
            Transformed tensor.
        """
        x = x + self.self_attention(self.norm_self(x))

        if self.use_cross_attention:
            assert context is not None
            x_norm = self.norm_cross(x)
            x = x + self.cross_gate(x_norm) * self.cross_attention(x_norm, key_value_input=context)

        return x + self.mlp(self.norm_mlp(x))


class CrossAttentionGate(nn.Module):
    """Starts cross-attention nearly off, then learns how much weather information to use."""

    def __init__(self, n_emb: int):
        """Initialize gate with near-zero output (bias=-3.0)."""
        super().__init__()
        self.gate_proj = nn.Linear(n_emb, n_emb)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -3.0)

    def forward(self, x: torch.Tensor):
        """Return sigmoid gate values in [0, 1]."""
        return torch.sigmoid(self.gate_proj(x))


class AttentionBlock(nn.Module):
    """Self-attention when key_value_input is None; cross-attention otherwise."""

    def __init__(
        self,
        *,
        n_emb: int,
        n_heads: int,
        n_patches: int,
        bias: bool = True,
        is_causal: bool = False,
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
    ):
        """Initialize multi-head attention with RoPE."""
        super().__init__()
        assert n_emb % n_heads == 0, "n_emb must be divisible by n_heads"
        self.n_emb = n_emb
        self.n_heads = n_heads
        self.head_dim = n_emb // n_heads
        self.is_causal = is_causal
        self.attn_dropout_p = attn_dropout
        self.res_dropout = nn.Dropout(res_dropout)

        self.wq = nn.Linear(n_emb, n_emb, bias=bias)
        self.wk = nn.Linear(n_emb, n_emb, bias=bias)
        self.wv = nn.Linear(n_emb, n_emb, bias=bias)
        self.wo = nn.Linear(n_emb, n_emb, bias=bias)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=n_patches)

    def forward(self, query_input: torch.Tensor, key_value_input: torch.Tensor | None = None):
        """Compute scaled dot-product attention with rotary embeddings.

        Returns:
            Attention output tensor of shape (B, q_len, n_emb).
        """
        if key_value_input is None:
            key_value_input = query_input

        b, q_len, _ = query_input.shape
        kv_len = key_value_input.shape[1]

        q = self.wq(query_input).view(b, q_len, self.n_heads, self.head_dim)
        k = self.wk(key_value_input).view(b, kv_len, self.n_heads, self.head_dim)
        v = self.wv(key_value_input).view(b, kv_len, self.n_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dx = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=self.is_causal,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        dx = dx.transpose(1, 2).contiguous().view(b, q_len, self.n_emb)
        return self.res_dropout(self.wo(dx))


class MLP(nn.Module):
    """Multi-layer perceptron with configurable depth and activation."""

    def __init__(
        self,
        *,
        n_interface: int,
        n_internal: int | None = None,
        n_layers: int = 2,
        bias: bool = True,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        """Initialize MLP layers."""
        super().__init__()
        n_internal = n_internal or n_interface * 4
        self.first_layer = nn.Linear(n_interface, n_internal, bias=bias)
        self.mid_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(n_internal, n_internal, bias=bias),
                Activation(activation),
            )
            for _ in range(n_layers - 2)
        ])
        self.final_layer = nn.Linear(n_internal, n_interface, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = Activation(activation)

    def forward(self, x: torch.Tensor):
        """Apply MLP layers with activation and dropout.

        Returns:
            Transformed tensor.
        """
        x = self.activation(self.first_layer(x))
        x = self.mid_layers(x)
        x = self.final_layer(x)
        return self.dropout_layer(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) with cached sin/cos."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        """Initialize RoPE and build sin/cos cache."""
        super().__init__()
        assert dim % 2 == 0, "RoPE head dimension must be even"
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor):
        """Apply rotary embeddings, extending cache if needed.

        Returns:
            Rotated embedding tensor.
        """
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
