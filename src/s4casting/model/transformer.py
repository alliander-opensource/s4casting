import torch
from einops import rearrange
from pydantic import NonNegativeFloat, NonNegativeInt, confloat
from torch import nn

from s4casting.core.loss import SubsetNLLLoss
from s4casting.model._blocks import MLP, RotaryPositionalEmbedding, SelfAttention
from s4casting.model._encoders import PatchDecoder, SeriesPatchEncoder, SpatialEmbedding, TemporalEmbedding
from s4casting.model._heads import GMMHead
from s4casting.model._norm import denorm, norm, norm_target

TIME_FEATURES = 3


class TransformerModel(nn.Module):
    """Grouped time-series transformer with temporal and variable attention."""

    def __init__(
        self,
        *,
        latent_dim: NonNegativeInt = 256,
        n_heads: NonNegativeInt = 8,
        n_layers: NonNegativeInt = 6,
        output_head: nn.Module = GMMHead(256, 2, 1),
        patch_size: NonNegativeInt = 8,
        dropout: confloat(ge=0, lt=1) = 0.0,  # type: ignore
        attn_bias: bool = True,
        mlp_layers: NonNegativeInt = 2,
        norm_clamp: NonNegativeFloat | None = 10.0,
        norm_eps: NonNegativeFloat = 1e-5,
        loss_fn: nn.Module = SubsetNLLLoss(1, "masked"),
        patch_encoder: nn.Module = SeriesPatchEncoder(256, 8),
        patch_decoder: nn.Module = PatchDecoder(256, 256, 8, [15], [15]),
        has_time: bool = True,
        causal: bool = False,
    ):
        """Initialize the TransformerModel.

        Args:
            latent_dim: Embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            output_head: Quantile or GMM head.
            patch_size: Size of patches for patch embedding.
            dropout: Dropout probability.
            attn_bias: Whether to use attention bias.
            mlp_layers: Number of MLP layers.
            norm_clamp: Clamping value for normalization.
            loss_fn: Loss function.
            norm_eps: Epsilon for norm.
            patch_encoder: Patch encoder module.
            patch_decoder: Patch decoder module.
            has_time: if time, lat, lon is present
            causal: Whether the time attention is causal (tokens only attend to the past). Group
                attention is never causal: feature order carries no causal structure, and a causal
                mask there would stop the target (feature 0) from attending its covariates at all.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.norm_clamp = norm_clamp
        self.norm_eps = norm_eps
        self.causal = causal

        self.loss_fn = loss_fn
        self.output_head = output_head
        self.patch_encoder = patch_encoder
        self.patch_decoder = patch_decoder
        self.has_time = has_time
        self.temporal_embedding = TemporalEmbedding(patch_size, latent_dim) if has_time else None
        self.spatial_embedding = SpatialEmbedding(latent_dim) if has_time else None

        self.transformer_layers = nn.ModuleList(
            TransformerLayer(
                latent_dim=latent_dim,
                n_heads=n_heads,
                dropout=dropout,
                attn_bias=attn_bias,
                mlp_layers=mlp_layers,
                norm_eps=norm_eps,
                causal=causal,
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
        time_features = None
        if self.has_time:
            x, xm, time_features = self._split_time_features(x, xm)

        x = x * xm
        x, mean_in, std_in = norm(
            x=x,
            xm=xm,
            eps=self.norm_eps,
            clamp=self.norm_clamp,
        )

        x = self.patch_encoder(
            x,
            xm,
        )  # B, L // P, F -> B, F, P, D

        x = self._add_time_features(x, time_features)  # B, F, P, D

        for block in self.transformer_layers:
            x = block(x, xm)  # B, F, P, D

        x = self.patch_decoder(x[:, 0, :, :])
        x = self.output_head(x)

        loss = None
        if y is not None:
            y = norm_target(
                mean_in=mean_in[..., :1],
                std_in=std_in[..., :1],
                y=y,
                ym=ym,
            )
            loss = self.loss_fn(
                out=x,
                target=y,
                input_interval=input_interval,
                output_interval=output_interval,
                mask=ym,
            )

        x = denorm(
            mean_in=mean_in[..., :1].unsqueeze(-1),
            std_in=std_in[..., :1].unsqueeze(-1),
            x=x,
            output_type=type(self.output_head),
            xm=xm[..., :1],
        )
        return x, loss

    def _split_time_features(
        self,
        x: torch.Tensor,
        xm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate load+weather features from time/loc features.

        Returns:
            Tuple of tensors.
        """
        return x[..., :-TIME_FEATURES], xm[..., :-TIME_FEATURES], x[..., -TIME_FEATURES:]

    def _add_time_features(
        self,
        x: torch.Tensor,
        time_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add time/loc features.

        Returns:
            Tensors.
        """
        if time_features is None:
            return x
        if self.temporal_embedding is None or self.spatial_embedding is None:
            return x

        temporal_embedding = self.temporal_embedding(time_features[..., 0])
        spatial_embedding = self.spatial_embedding(time_features[:, -1:, 1:])
        return x + temporal_embedding[:, None, :, :] + spatial_embedding[:, None, :, :]


class TransformerLayer(nn.Module):
    """Time attention, group attention, MLP."""

    def __init__(
        self,
        latent_dim: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        attn_bias: bool = True,
        mlp_layers: int = 2,
        norm_eps: float = 1e-5,
        group_gate_init: float = -3.0,
        causal: bool = False,
    ):
        """Initialize the TransformerLayer.

        Args:
            latent_dim: Embedding dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            attn_bias: Whether to use attention bias.
            mlp_layers: Number of MLP layers.
            norm_eps: Epsilon for norm.
            group_gate_init: Starts group attention at zero.
            causal: Whether the time attention is causal; group attention never is.
        """
        super().__init__()

        self.time_norm = nn.LayerNorm(latent_dim, eps=norm_eps)
        self.time_attention = SelfAttention(
            latent_dim=latent_dim,
            n_heads=n_heads,
            bias=attn_bias,
            dropout=dropout,
            is_causal=causal,
            rope=RotaryPositionalEmbedding(dim=latent_dim // n_heads),
        )

        self.group_norm = nn.LayerNorm(latent_dim, eps=norm_eps)
        self.group_attention = SelfAttention(
            latent_dim=latent_dim,
            n_heads=n_heads,
            bias=attn_bias,
            dropout=dropout,
            is_causal=False,  # there's never a causal structure in feature order
        )
        # start group mixing off
        self.group_gate = nn.Parameter(torch.tensor(group_gate_init))

        self.mlp_norm = nn.LayerNorm(latent_dim, eps=norm_eps)
        self.mlp = MLP(
            embed_dim=latent_dim,
            n_layers=mlp_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, xm: torch.Tensor):
        """Apply temporal-attention, group-attention, and MLP.

        Args:
            x: Tensor of shape [B, F, P, D].
            xm: Tensor of shape [B, T, F].

        Returns:
            Tensor of shape [B, F, P, D].
        """
        b, f, p, _ = x.shape

        # Temporal attention [B, F, P, D] -> [B*F, P, D]
        time_tokens = rearrange(
            x,
            "b f p e -> (b f) p e",
        )

        time_tokens = time_tokens + self.time_attention(self.time_norm(time_tokens))

        x = rearrange(
            time_tokens,
            "(b f) p e -> b f p e",
            b=b,
            f=f,
        )

        # Group attention [B, F, P, D] -> [B*P, F, D]
        group_tokens = rearrange(
            x,
            "b f p e -> (b p) f e",
        )

        key_present = xm.reshape(b, p, -1, f).bool().any(dim=2)

        key_present[..., 0] = True

        key_present = key_present.flatten(0, 1)

        group_update = self.group_attention(
            self.group_norm(group_tokens),
            key_present=key_present,
        )

        group_tokens = group_tokens + torch.sigmoid(self.group_gate) * group_update

        x = rearrange(
            group_tokens,
            "(b p) f e -> b f p e",
            b=b,
            p=p,
        )

        return x + self.mlp(self.mlp_norm(x))
