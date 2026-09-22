# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

# type: ignore
import warnings
from copy import deepcopy

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from s4casting.core.config import IOConfiguration, ModelConfiguration
from s4casting.core.loss import CompositeLoss, SoftClipLoss, SubsetNLLLoss, SubsetPinballLoss
from s4casting.core.machine import Machine
from s4casting.core.model_container import ModelContainer
from s4casting.model._encoders import PatchDecoder, PatchEncoder, SeperateLocTime, SeriesPatchEncoder, SSEncoder
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
        return SubsetNLLLoss(
            config.loss.sigma_regularisation_factor,
            config.loss.mask_mode,
        )
    if config.loss.loss == "mse":
        return nn.MSELoss()

    if config.loss.loss == "pinball":
        return SubsetPinballLoss(
            config.output_head.quantile_values,
            config.loss.mask_mode,
        )

    raise ValueError(f"Loss function {config.loss.loss} not implemented")


def provide_model_container(config: ModelConfiguration, io_config: IOConfiguration, machine: Machine) -> ModelContainer:
    """Provide a ModelContainer instance.

    Args:
        config (ModelConfiguration): Model configuration.
        io_config (IOConfiguration): IO configuration.
        machine (Machine): Machine information.

    Returns:
        ModelContainer: An instance of ModelContainer.
    """
    if config.model == "chronos":
        model = _build_chronos_model(config)
        model.to(machine.torch_device)
        config.n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return ModelContainer(
            model=model, ddp=(DDP(model, device_ids=[machine.ddp.local_rank]) if machine.ddp else None)
        )

    n_features = sum(
        len(x.subset_features) if x.subset_features else x.n_features
        for x in {k.split("_")[0]: v for k, v in io_config.features.items()}.values()
    )

    has_time = any(io_config.features[name].loader == "time" for name in io_config.feature_order)

    n_data_features = n_features - 3 * has_time
    if config.patch_encoder.patch_size != 1 and n_data_features > config.n_out_features:
        warnings.warn(
            f"Weather auxiliary loss is disabled because patch_size="
            f"{config.patch_encoder.patch_size} > 1. Set patch_size=1 to enable it.",
            UserWarning,
            stacklevel=2,
        )
    n_weather_features = max(0, n_data_features - config.n_out_features) if config.patch_encoder.patch_size == 1 else 0

    # TODO: clean this messiness up when refactoring model container - preferably to have a model specific builder
    latent_dim = config.ssm.latent_dim if config.model == "ssm" else config.transformer.latent_dim

    if config.patch_encoder.arch == "linear":
        if config.model == "transformer":
            patch_encoder = SeriesPatchEncoder(
                latent_dim,
                config.patch_encoder.patch_size,
            )
        else:
            patch_encoder = PatchEncoder(
                latent_dim,
                n_data_features,
                config.patch_encoder.patch_size,
            )

    elif config.patch_encoder.arch == "ss":
        patch_encoder = SSEncoder(
            latent_dim,
            n_data_features,
            n_layers=config.patch_encoder.n_layers,
            patch_size=config.patch_encoder.patch_size,
        )

    if has_time and config.model != "transformer":
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
    composite_loss = CompositeLoss(config.loss.components)

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
            mixer_size=config.ssm.mixer_size,
            patch_size=config.patch_encoder.patch_size,
            norm_clamp=config.norm_clamp,
            norm_eps=config.norm_eps,
            loss_fn=loss_fn,
            composite_loss=composite_loss,
            output_head=output_head,
            patch_encoder=patch_encoder,
            patch_decoder=patch_decoder,
            base_sample_interval_minutes=config.base_sample_interval_minutes,
            n_weather_features=n_weather_features,
        )

    elif config.model == "transformer":
        model = TransformerModel(
            latent_dim=latent_dim,
            n_heads=config.transformer.n_heads,
            n_layers=config.transformer.n_layers,
            patch_size=config.patch_encoder.patch_size,
            dropout=config.transformer.dropout,
            attn_bias=config.transformer.attn_bias,
            mlp_layers=config.transformer.mlp_layers,
            loss_fn=loss_fn,
            output_head=output_head,
            patch_encoder=patch_encoder,
            patch_decoder=patch_decoder,
            norm_clamp=config.norm_clamp,
            norm_eps=config.norm_eps,
            has_time=has_time,
            causal=config.transformer.causal,
        )

    model.to(
        machine.torch_device
    )  # todo: include Bob's clamping parameter for clamping the normalization. Include in normalizer?
    config.n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ModelContainer(model=model, ddp=(DDP(model, device_ids=[machine.ddp.local_rank]) if machine.ddp else None))
