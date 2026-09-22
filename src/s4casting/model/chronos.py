from __future__ import annotations

import math

import torch
from torch import nn


class ChronosWrapper(nn.Module):
    """Multivariate Chronos-2 wrapper for fine-tuning within s4casting.

    All n_features input channels are treated as variates within the same group,
    enabling cross-variate attention via Chronos-2's group mechanism. Loss is
    computed only on the first n_out_features channels (the measurement targets);
    remaining channels (e.g. weather) are passed as context-only variates with
    NaN future targets so Chronos auto-masks them from the loss.

    Chronos-2 performs its own internal instance normalisation, so raw
    (unnormalised) values must be passed. Masked positions are represented
    as NaN, which Chronos handles natively.

    Forward signature is identical to other s4casting models:
        (x, xm, input_interval, output_interval, y, ym) → (predictions, loss)
    """

    def __init__(
        self,
        model_id: str,
        n_out_features: int,
        prediction_length: int,
        freeze_backbone: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 32.0,
        lora_target_modules: list[str] | None = None,
    ):
        """Initialise the ChronosWrapper.

        Args:
            model_id: HuggingFace model identifier, e.g. 'autogluon/chronos-2-small'.
            n_out_features: Number of target channels; remaining channels are covariate-only.
            prediction_length: Default forecast horizon in timesteps (overridden by y at runtime).
            freeze_backbone: Freeze all non-LoRA parameters when True.
            use_lora: Apply LoRA adapters for parameter-efficient fine-tuning.
            lora_rank: LoRA rank (r).
            lora_alpha: LoRA scaling factor (alpha).
            lora_target_modules: Attention projection names to attach LoRA to.

        """
        super().__init__()
        from chronos import Chronos2Pipeline  # noqa: PLC0415  # deferred — optional dependency

        self.n_out_features = n_out_features
        self.prediction_length = prediction_length

        # Load onto CPU; factory calls model.to(machine.torch_device) afterwards.
        pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")
        self.pipeline = pipeline
        self.chronos_model = pipeline.model

        if use_lora:
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415

            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules or ["q", "v"],
                bias="none",
            )
            self.chronos_model = get_peft_model(self.chronos_model, lora_cfg)

        if freeze_backbone:
            for name, param in self.chronos_model.named_parameters():
                if "lora_" not in name:
                    param.requires_grad_(False)

        # Patch size used to convert pred_len (timesteps) → num_output_patches.
        self._patch_size: int = pipeline.model.chronos_config.input_patch_size

    def _flatten(self, t: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Convert (B, T, F) to (B*F, T), replacing masked positions with NaN.

        Args:
            t: (B, T, F)
            mask: (B, T, F) — 1 valid, 0 missing. When None, no NaN substitution.

        Returns:
            (B*F, T) with NaN where mask == 0.
        """
        if mask is not None:
            t = torch.where(mask.bool(), t, torch.full_like(t, float("nan")))
        B, T, F = t.shape
        return t.permute(0, 2, 1).reshape(B * F, T)

    def forward(
        self,
        x: torch.Tensor,
        xm: torch.Tensor,
        input_interval: int,
        output_interval: int,
        y: torch.Tensor | None = None,
        ym: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        The batcher produces X and Y with the same shape (B, T_total, n_features),
        where T_total = context_window + prediction_window. Y is a copy of X with Ym
        marking only the last pred_len timesteps of the target feature (dim 0) as valid.

        We therefore:
        - Use x[:, :-pred_len, :] as Chronos context (history only).
        - Extract y[:, -pred_len:, :n_out] as the actual prediction target.
        - Tell Chronos to forecast pred_len timesteps ahead.

        Args:
            x:  (B, T_total, n_features) — full window, raw values, no pre-normalisation.
            xm: (B, T_total, n_features) — 1 valid, 0 missing (0 for target feature in pred window).
            input_interval:  sample interval in minutes (unused, kept for API compatibility).
            output_interval: sample interval in minutes (unused, kept for API compatibility).
            y:  (B, T_total, n_features) — full window copy; target in last pred_len rows of dim 0.
            ym: (B, T_total, n_features) — 1 only for target feature in the last pred_len timesteps.

        Returns:
            preds: (B, pred_len, n_out_features, n_quantiles) — original-scale predictions.
            loss:  scalar tensor (Chronos internal quantile loss) or None when y is None.
        """
        B, _T_total, n_feat = x.shape
        pred_len = self.prediction_length
        n_out = self.n_out_features

        # Slice context window: exclude the prediction horizon.
        x_ctx = x[:, :-pred_len, :]  # (B, ctx_len, n_feat)
        xm_ctx = xm[:, :-pred_len, :]  # (B, ctx_len, n_feat)

        # Flatten to (B*n_feat, ctx_len); NaN marks masked positions.
        context_flat = self._flatten(x_ctx, xm_ctx)
        context_mask_flat = self._flatten(xm_ctx).float()

        # All variates from the same sample share a group_id → cross-variate attention.
        group_ids = torch.arange(B, device=x.device).repeat_interleave(n_feat)

        num_output_patches = math.ceil(pred_len / self._patch_size)

        # Build future_covariates (B, pred_len, n_feat) → flat (B*n_feat, pred_len).
        # Per Chronos-2 API: NaN in a variate's future_covariates marks it as "to be forecasted";
        # real values mark it as a known future covariate (conditioning only). Target channels
        # must be NaN; weather channels carry the forecast values that were always present in x.
        x_future = x[:, -pred_len:, :]
        xm_future = xm[:, -pred_len:, :]
        future_cov_mask_full = xm_future.clone()
        future_cov_mask_full[:, :, :n_out] = 0  # force target channels to NaN via _flatten
        future_covariates_flat = self._flatten(x_future, future_cov_mask_full)

        # future_target is ONLY for the training quantile loss — NaN for covariate channels
        # so Chronos doesn't score them, ground truth for target channels. None at inference.
        future_target_flat: torch.Tensor | None = None
        future_target_mask_flat: torch.Tensor | None = None
        if y is not None:
            y_full = torch.full((B, pred_len, n_feat), float("nan"), device=x.device, dtype=x.dtype)
            y_full[:, :, :n_out] = y[:, -pred_len:, :n_out]
            future_target_flat = self._flatten(y_full)

            if ym is not None:
                ym_full = torch.zeros(B, pred_len, n_feat, device=x.device, dtype=x.dtype)
                ym_full[:, :, :n_out] = ym[:, -pred_len:, :n_out]
                future_target_mask_flat = self._flatten(ym_full).float()

        # Chronos-2 forward: handles instance norm, patching, and quantile loss internally.
        # quantile_preds shape: (B*n_feat, n_quantiles, num_output_patches * patch_size)
        chronos_out = self.chronos_model(
            context=context_flat,
            context_mask=context_mask_flat,
            group_ids=group_ids,
            future_covariates=future_covariates_flat,
            num_output_patches=num_output_patches,
            future_target=future_target_flat,
            future_target_mask=future_target_mask_flat,
        )

        # Trim patch-rounding overshoot, restore (B, n_feat, n_q, pred_len).
        q_flat = chronos_out.quantile_preds[:, :, :pred_len]  # (B*n_feat, n_q, pred_len)
        n_q = q_flat.shape[1]
        q_preds = q_flat.reshape(B, n_feat, n_q, pred_len)

        # Keep only target features and rearrange to (B, pred_len, n_out, n_q).
        q_preds = q_preds[:, :n_out, :, :].permute(0, 3, 1, 2)

        return q_preds, chronos_out.loss
