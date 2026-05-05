# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

# type: ignore
from copy import deepcopy

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from s4casting.core.config import DTYPE_MAP, IOConfiguration, ModelConfiguration
from s4casting.core.loss import SoftClipLoss, SubsetNLLLoss, SubsetPinballLoss
from s4casting.core.machine import Machine
from s4casting.core.model_container import ModelContainer
from s4casting.model._encoders import PatchDecoder, PatchEncoder, SeperateLocTime, SSEncoder
from s4casting.model._heads import GMMHead, QuantileHead
from s4casting.model.chronos import ChronosWrapper
from s4casting.model.ss import SSModel
from s4casting.model.transformer import TransformerModel


def _build_chronos_model(config: ModelConfiguration) -> nn.Module:
    """Instantiate a ChronosWrapper from config.

    Returns:
        nn.Module: ChronosWrapper on CPU (caller moves to device).
    """
    assert config.chronos is not None, "config.model='chronos' requires a [model.chronos] section"
    predict_width_days = config.predict_width if isinstance(config.predict_width, int) else 2
    prediction_length = (predict_width_days * 24 * 60) // config.base_sample_interval_minutes
    return ChronosWrapper(
        model_id=config.chronos.model_id,
        n_out_features=config.n_out_features,
        prediction_length=prediction_length,
        freeze_backbone=config.chronos.freeze_backbone,
        use_lora=config.chronos.use_lora,
        lora_rank=config.chronos.lora_rank,
        lora_alpha=config.chronos.lora_alpha,
        lora_target_modules=config.chronos.lora_target_modules,
    )


def _build_loss_fn(config: ModelConfiguration) -> nn.Module:
    """Build the loss function from config.

    Returns:
        Configured loss function, optionally wrapped with soft clipping.
    """
    if config.loss.loss == "nll":
        loss_fn = SubsetNLLLoss(
            config.loss.sigma_regularisation_factor,
            config.loss.mask_mode,
        )
    elif config.loss.loss == "mse":
        loss_fn = nn.MSELoss()
    elif config.loss.loss == "pinball":
        loss_fn = SubsetPinballLoss(
            config.output_head.quantile_values,
            config.loss.mask_mode,
        )
    else:
        msg = f"Loss function {config.loss.loss} not implemented"
        raise ValueError(msg)

    if config.loss.alpha_clip != 0:
        loss_clip = SoftClipLoss(alpha=config.loss.alpha_clip)
        loss_core = deepcopy(loss_fn)

        def loss_fn(*args, **kwargs):
            return loss_clip(loss_core(*args, **kwargs))

    return loss_fn


def provide_model_container(config: ModelConfiguration, io_config: IOConfiguration, machine: Machine) -> ModelContainer:
    """Provide a ModelContainer instance.

    Args:
        config (ModelConfiguration): Model configuration.
        io_config (IOConfiguration): IO configuration.
        machine (Machine): Machine information.

    Returns:
        ModelContainer: An instance of ModelContainer.
    """
    n_features = sum(
        len(x.subset_features) if x.subset_features else x.n_features
        for x in {k.split("_")[0]: v for k, v in io_config.features.items()}.values()
    )
    latent_dim = config.transformer.latent_dim if config.model == "transformer" else config.latent_dim

    if config.patch_encoder.arch == "linear":
        if config.model == "transformer":
            patch_encoder = PatchEncoder(
                latent_dim,
                (n_features - 3 * any(v.loader == "time" for v in io_config.features.values()))
                * 2,  # to accept target + mask
                config.patch_encoder.patch_size,
            )
        else:
            patch_encoder = PatchEncoder(
                latent_dim,
                n_features - 3 * any(v.loader == "time" for v in io_config.features.values()),
                config.patch_encoder.patch_size,
            )

    elif config.patch_encoder.arch == "ss":
        patch_encoder = SSEncoder(
            latent_dim,
            n_features - 3 * any(v.loader == "time" for v in io_config.features.values()),
            n_layers=config.patch_encoder.n_layers,
            patch_size=config.patch_encoder.patch_size,
        )

    if any(v.loader == "time" for v in io_config.features.values()):
        patch_encoder = SeperateLocTime(patch_encoder)

    patch_decoder = PatchDecoder(
        latent_dim,
        latent_dim,
        config.patch_decoder.patch_size,
        config.input_sample_intervals_minutes,
        config.output_sample_intervals_minutes,
        config.patch_encoder.arch,
    )

    if config.output_head.arch == "gmm":
        assert config.loss.loss == "nll", "You need a nll loss to train a gmm"
        output_head = (
            GMMHead(latent_dim, config.output_head.n_gaussians, config.n_out_features)
            if config.output_head.n_gaussians > 1
            else nn.Linear(latent_dim, config.n_out_features)
        )

    elif config.output_head.arch == "quantile":
        assert config.loss.loss == "pinball", "You need a pinball loss to train a quantile head"
        output_head = QuantileHead(latent_dim, config.n_out_features, config.output_head.quantile_values)

    loss_fn = _build_loss_fn(config)

    assert config.loss.loss in ["nll", "mse", "pinball"], f"Loss function {config.loss.loss} not implemented"

    if config.loss.alpha_clip != 0:
        loss_clip = SoftClipLoss(alpha=config.loss.alpha_clip)
        loss_core = deepcopy(loss_fn)

        def loss_fn(*args, **kwargs):
            return loss_clip(loss_core(*args, **kwargs))

    # Get model
    if config.model == "ssm":
        model = SSModel(
            latent_dim=latent_dim,
            n_layer=config.ssm.n_layers,
            kernel=config.ssm.kernel,
            backend="keops" if machine.torch_device_kind == "cuda" else "naive",
            patch_size=config.patch_encoder.patch_size,
            norm_clamp=config.norm_clamp,
            norm_eps=config.norm_eps,
            loss_fn=loss_fn,
            output_head=output_head,
            patch_encoder=patch_encoder,
            patch_decoder=patch_decoder,
            base_sample_interval_minutes=config.base_sample_interval_minutes,
        )

    elif config.model == "transformer":
        input_length = (
            (config.context_window[0] - config.predict_width) * 24 * 60
        ) // config.base_sample_interval_minutes
        predict_length = (config.predict_width * 24 * 60) // config.base_sample_interval_minutes
        model = TransformerModel(
            seq_len=input_length + predict_length,
            latent_dim=latent_dim,
            n_heads=config.transformer.n_heads,
            n_layers=config.transformer.n_layers,
            patch_size=config.patch_encoder.patch_size,
            dropout=config.transformer.dropout,
            use_cross_attention=config.transformer.use_cross_attention,
            context_n_layers=config.transformer.context_n_layers,
            dtype=DTYPE_MAP[config.internal_dtype],
            is_causal=config.transformer.is_causal,
            attn_bias=config.transformer.attn_bias,
            mlp_bias=config.transformer.mlp_bias,
            mlp_layers=config.transformer.mlp_layers,
            mlp_activation=config.transformer.mlp_activation,
            loss_fn=loss_fn,
            output_head=output_head,
            patch_encoder=patch_encoder,
            patch_decoder=patch_decoder,
            norm_clamp=config.norm_clamp,
            norm_eps=config.norm_eps,
            base_sample_interval_minutes=config.base_sample_interval_minutes,
        )
    elif config.model == "chronos":
        model = _build_chronos_model(config)

    model.to(
        machine.torch_device
    )  # todo: include Bob's clamping parameter for clamping the normalization. Include in normalizer?
    config.n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ModelContainer(model=model, ddp=(DDP(model, device_ids=[machine.ddp.local_rank]) if machine.ddp else None))
