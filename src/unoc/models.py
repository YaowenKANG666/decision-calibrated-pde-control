"""Residual FNO, finite-section TNO, DSC-DNO, and mixture world models."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def model_inputs(
    state: torch.Tensor,
    action: torch.Tensor,
    viscosity: torch.Tensor,
    boundary: torch.Tensor,
) -> torch.Tensor:
    """Build function-valued input channels."""

    batch, grid = state.shape
    x = torch.linspace(0.0, 1.0, grid, dtype=state.dtype, device=state.device)
    actuator = torch.exp(-0.5 * ((x - 0.68) / 0.12) ** 2)
    actuator = actuator / actuator.max()
    control = action[:, None] * actuator[None, :]
    nu = viscosity[:, None].expand(-1, grid)
    boundary_extension = (
        boundary[:, :1] * (1.0 - x[None, :]) + boundary[:, 1:] * x[None, :]
    )
    coordinate = x[None, :].expand(batch, -1)
    return torch.stack((state, control, nu, boundary_extension, coordinate), dim=1)


class SpectralConv1d(nn.Module):
    """Periodic FNO or doubled-grid finite-section TNO spectral map."""

    def __init__(self, width: int, modes: int, kind: str):
        super().__init__()
        if kind not in {"fno", "tno"}:
            raise ValueError(kind)
        self.width, self.modes, self.kind = width, modes, kind
        scale = 1.0 / math.sqrt(width)
        self.weight = nn.Parameter(scale * torch.randn(width, width, modes, 2))

    def _multiply(self, spectrum: torch.Tensor) -> torch.Tensor:
        modes = min(self.modes, spectrum.shape[-1])
        output = torch.zeros(
            spectrum.shape[0],
            self.width,
            spectrum.shape[-1],
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        weight = torch.view_as_complex(self.weight[:, :, :modes].contiguous())
        output[:, :, :modes] = torch.einsum(
            "bim,iom->bom",
            spectrum[:, :, :modes],
            weight,
        )
        return output

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        grid = value.shape[-1]
        if self.kind == "fno":
            spectrum = torch.fft.rfft(value)
            return torch.fft.irfft(self._multiply(spectrum), n=grid)
        doubled = 2 * grid - 2
        embedded = F.pad(value, (0, doubled - grid))
        spectrum = torch.fft.rfft(embedded)
        return torch.fft.irfft(self._multiply(spectrum), n=doubled)[..., :grid]


def _causal_dynamic(kernel: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Batchwise causal convolution for BxR kernels and BxCxN/BxRxCxN values."""

    grid = kernel.shape[-1]
    spectrum_kernel = torch.fft.rfft(kernel, n=2 * grid)
    spectrum_value = torch.fft.rfft(value, n=2 * grid)
    if value.ndim == 3:
        product = spectrum_kernel[:, :, None] * spectrum_value[:, None]
    elif value.ndim == 4:
        product = spectrum_kernel[:, :, None] * spectrum_value
    else:
        raise ValueError(value.shape)
    return torch.fft.irfft(product, n=2 * grid)[..., :grid]


def _displacement_apply(
    g: torch.Tensor,
    h: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    r"""Apply \(\sum_r L(g_r)L(h_r)^\top\) without forming dense matrices."""

    upper = _causal_dynamic(h, torch.flip(value, dims=(-1,)))
    upper = torch.flip(upper, dims=(-1,)) / value.shape[-1]
    return _causal_dynamic(g, upper) / value.shape[-1]


class DSCDNOBlock(nn.Module):
    """Residual displacement-structured convolution conditioned on the input."""

    def __init__(self, width: int, rank: int = 6, conditioning_channels: int = 5):
        super().__init__()
        self.rank = rank
        self.generator = nn.Sequential(
            nn.Conv1d(conditioning_channels, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(width, 2 * rank, 1),
        )
        nn.init.normal_(self.generator[-1].weight, std=0.005)
        nn.init.zeros_(self.generator[-1].bias)
        self.mix = nn.Parameter(torch.randn(rank, width, width) / math.sqrt(rank * width))
        self.local = nn.Conv1d(width, width, 1)
        self.scale = nn.Parameter(torch.full((rank,), 0.1))

    def forward(self, hidden: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        generators = torch.tanh(self.generator(conditioning))
        g, h = generators[:, : self.rank], generators[:, self.rank :]
        structured = _displacement_apply(g, h, hidden)
        structured = structured * self.scale[None, :, None, None]
        structured = torch.einsum("brin,rio->bon", structured, self.mix)
        return hidden + F.gelu(structured + self.local(hidden))


class OperatorWorldModel(nn.Module):
    """One-step PDE world model with a learned pointwise uncertainty scale."""

    def __init__(
        self,
        kind: str = "fno",
        width: int = 32,
        modes: int = 12,
        layers: int = 4,
    ):
        super().__init__()
        if kind not in {"fno", "tno", "dscdno"}:
            raise ValueError(kind)
        self.kind = kind
        self.lift = nn.Conv1d(5, width, 1)
        if kind in {"fno", "tno"}:
            self.global_layers = nn.ModuleList(
                [SpectralConv1d(width, modes, kind) for _ in range(layers)]
            )
            self.local_layers = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
            self.dsc_blocks = None
        else:
            self.global_layers = nn.ModuleList()
            self.local_layers = nn.ModuleList()
            self.dsc_blocks = nn.ModuleList(
                [DSCDNOBlock(width, rank=max(4, modes // 2)) for _ in range(layers)]
            )
        self.head = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, 2, 1),
        )
        nn.init.constant_(self.head[-1].bias[1], -4.0)
        # Identifiable structured scale for latent actuator-gain variation.
        # softplus(-5) is approximately 0.0067 per unit action at the actuator peak.
        self.actuator_scale_logit = nn.Parameter(torch.tensor(-5.0))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        viscosity: torch.Tensor,
        boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioning = model_inputs(state, action, viscosity, boundary)
        hidden = self.lift(conditioning)
        if self.dsc_blocks is None:
            for global_layer, local_layer in zip(self.global_layers, self.local_layers):
                # ResNet-style skip is inside every operator layer.
                hidden = hidden + F.gelu(global_layer(hidden) + local_layer(hidden))
        else:
            for block in self.dsc_blocks:
                hidden = block(hidden, conditioning)
        output = self.head(hidden)
        mean = state + output[:, 0]
        base_scale = F.softplus(output[:, 1]) + 1e-4
        grid = state.shape[1]
        x = torch.linspace(0.0, 1.0, grid, dtype=state.dtype, device=state.device)
        actuator = torch.exp(-0.5 * ((x - 0.68) / 0.12).square())
        actuator = actuator / actuator.max()
        actuator_scale = (
            F.softplus(self.actuator_scale_logit)
            * action.abs()[:, None]
            * actuator[None, :]
        )
        scale = torch.sqrt(base_scale.square() + actuator_scale.square())
        # Exact boundary projection prevents avoidable feasibility errors.
        mean = mean.clone()
        mean[:, 0], mean[:, -1] = boundary[:, 0], boundary[:, 1]
        return mean, scale


class MixtureOperatorWorldModel(nn.Module):
    """Input-conditioned mixture of residual FNO, TNO, and DSC-DNO experts."""

    def __init__(self, width: int = 24, modes: int = 12, layers: int = 3):
        super().__init__()
        self.kind = "moe"
        self.experts = nn.ModuleList(
            [
                OperatorWorldModel(kind, width, modes, layers)
                for kind in ("fno", "tno", "dscdno")
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        viscosity: torch.Tensor,
        boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.stack(
            (
                state.mean(dim=1),
                state.std(dim=1),
                state.abs().amax(dim=1),
                action,
                viscosity,
                boundary.abs().amax(dim=1),
            ),
            dim=1,
        )
        weights = torch.softmax(self.gate(features), dim=1)
        outputs = [expert(state, action, viscosity, boundary) for expert in self.experts]
        means = torch.stack([item[0] for item in outputs], dim=1)
        scales = torch.stack([item[1] for item in outputs], dim=1)
        mean = torch.sum(weights[:, :, None] * means, dim=1)
        second_moment = torch.sum(
            weights[:, :, None] * (scales.square() + means.square()),
            dim=1,
        )
        scale = torch.sqrt(torch.clamp(second_moment - mean.square(), min=1e-8))
        return mean, scale


class PerturbationScaleWorldModel(nn.Module):
    """Base prediction with a scale from label-perturbation disagreement.

    Both operators are trained on the same proper training inputs. The second
    operator sees Gaussian-perturbed labels. Their smoothed pointwise
    disagreement is a data-efficient local uncertainty proxy; split conformal
    calibration, rather than the raw disagreement, supplies validity.
    """

    def __init__(
        self,
        base_model: nn.Module,
        perturbed_model: nn.Module,
        smoothing_window: int = 5,
        scale_floor: float = 1e-4,
    ):
        super().__init__()
        if smoothing_window < 1 or smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be a positive odd integer")
        self.kind = f"{getattr(base_model, 'kind', 'operator')}_perturbation"
        self.base_model = base_model
        self.perturbed_model = perturbed_model
        self.smoothing_window = smoothing_window
        self.register_buffer("scale_floor", torch.tensor(float(scale_floor)))

    def raw_scale(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        viscosity: torch.Tensor,
        boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_mean, _ = self.base_model(state, action, viscosity, boundary)
        perturbed_mean, _ = self.perturbed_model(state, action, viscosity, boundary)
        disagreement = (base_mean - perturbed_mean).abs()
        if self.smoothing_window > 1:
            padding = self.smoothing_window // 2
            padded = F.pad(
                disagreement[:, None, :],
                (padding, padding),
                mode="replicate",
            )
            disagreement = F.avg_pool1d(
                padded,
                kernel_size=self.smoothing_window,
                stride=1,
            )[:, 0]
        return base_mean, disagreement

    def set_scale_floor(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("scale floor must be positive")
        self.scale_floor.fill_(float(value))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        viscosity: torch.Tensor,
        boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, disagreement = self.raw_scale(state, action, viscosity, boundary)
        return mean, disagreement.clamp_min(self.scale_floor.to(disagreement))


def build_model(kind: str, width: int = 32, modes: int = 12, layers: int = 4) -> nn.Module:
    if kind == "moe":
        return MixtureOperatorWorldModel(width=max(16, width * 3 // 4), modes=modes, layers=layers)
    return OperatorWorldModel(kind, width=width, modes=modes, layers=layers)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_perturbation_world_model(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[PerturbationScaleWorldModel, dict[str, object]]:
    """Reconstruct a saved two-operator world model from its metadata.

    Checkpoints produced before architecture metadata was introduced are
    interpreted as the documented quick FNO configuration.
    """

    device = torch.device(device)
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    model_kind = str(payload.get("model_kind", "fno"))
    architecture = dict(
        payload.get("architecture", {"width": 20, "modes": 10, "layers": 3})
    )
    uncertainty = dict(payload.get("uncertainty", {}))
    smoothing_window = round(float(uncertainty.get("smoothing_window", 5)))
    scale_floor = float(uncertainty.get("scale_floor", 1e-4))
    base = build_model(model_kind, **architecture).to(device)
    perturbed = build_model(model_kind, **architecture).to(device)
    model = PerturbationScaleWorldModel(
        base,
        perturbed,
        smoothing_window=smoothing_window,
        scale_floor=scale_floor,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
