"""Publication backbones for LA010010-aligned GPR inversion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return 8 if channels >= 8 else 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetInversion(nn.Module):
    def __init__(self, output_size: int = 256, dropout: float = 0.10):
        super().__init__()
        self.output_size = output_size
        self.e1 = ConvBlock(1, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.e4 = ConvBlock(128, 256, dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 256, dropout)
        self.u3 = UpBlock(256, 256, 128, dropout)
        self.u2 = UpBlock(128, 128, 64)
        self.u1 = UpBlock(64, 64, 32)
        self.u0 = UpBlock(32, 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        x = self.u3(b, e4)
        x = self.u2(x, e3)
        x = self.u1(x, e2)
        x = self.u0(x, e1)
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x))


class UNetPPInversion(nn.Module):
    """Compact U-Net++ with dense nested skip pathways."""

    def __init__(self, output_size: int = 256, dropout: float = 0.10):
        super().__init__()
        self.output_size = output_size
        c = [32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2)
        self.x00 = ConvBlock(1, c[0])
        self.x10 = ConvBlock(c[0], c[1])
        self.x20 = ConvBlock(c[1], c[2])
        self.x30 = ConvBlock(c[2], c[3], dropout)
        self.x01 = ConvBlock(c[0] + c[1], c[0])
        self.x11 = ConvBlock(c[1] + c[2], c[1])
        self.x21 = ConvBlock(c[2] + c[3], c[2], dropout)
        self.x02 = ConvBlock(c[0] * 2 + c[1], c[0])
        self.x12 = ConvBlock(c[1] * 2 + c[2], c[1])
        self.x03 = ConvBlock(c[0] * 3 + c[1], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    @staticmethod
    def up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x01 = self.x01(torch.cat([x00, self.up(x10, x00)], 1))
        x11 = self.x11(torch.cat([x10, self.up(x20, x10)], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30, x20)], 1))
        x02 = self.x02(torch.cat([x00, x01, self.up(x11, x00)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21, x10)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12, x00)], 1))
        out = F.interpolate(x03, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(out))


class TransUNetInversion(nn.Module):
    def __init__(self, output_size: int = 256, dropout: float = 0.10):
        super().__init__()
        self.output_size = output_size
        self.stem = nn.Sequential(ConvBlock(1, 64), nn.MaxPool2d(2), ConvBlock(64, 128))
        self.project = nn.Conv2d(128, 256, 1)
        self.pos = nn.Parameter(torch.zeros(1, 16 * 16, 256))
        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.decoder = nn.Sequential(
            ConvBlock(256, 128, dropout),
            ConvBlock(128, 64),
            ConvBlock(64, 32),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(self.stem(x))
        x = F.adaptive_avg_pool2d(x, (16, 16))
        tokens = x.flatten(2).transpose(1, 2) + self.pos
        x = self.transformer(tokens).transpose(1, 2).reshape(x.shape[0], 256, 16, 16)
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        x = self.decoder(x)
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x))


class TinyInversion(nn.Module):
    def __init__(self, output_size: int = 256, dropout: float = 0.10):
        super().__init__()
        self.output_size = output_size
        self.encoder = nn.Sequential(
            ConvBlock(1, 16),
            nn.MaxPool2d(2),
            ConvBlock(16, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, dropout),
        )
        self.head = nn.Sequential(ConvBlock(64, 32, dropout), nn.Conv2d(32, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x))


class SpectralConv2d(nn.Module):
    def __init__(self, channels: int, modes_time: int, modes_space: int):
        super().__init__()
        self.modes_time = modes_time
        self.modes_space = modes_space
        scale = 1.0 / channels
        shape = (channels, channels, modes_time, modes_space)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def multiply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bctw,cotw->botw", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        transformed = torch.fft.rfft2(x, norm="ortho")
        output = torch.zeros(
            batch,
            channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        mt = min(self.modes_time, height // 2)
        ms = min(self.modes_space, width // 2 + 1)
        output[:, :, :mt, :ms] = self.multiply(
            transformed[:, :, :mt, :ms], self.weight_positive[:, :, :mt, :ms]
        )
        output[:, :, -mt:, :ms] = self.multiply(
            transformed[:, :, -mt:, :ms], self.weight_negative[:, :, :mt, :ms]
        )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class FourierBlock(nn.Module):
    def __init__(self, channels: int, modes_time: int, modes_space: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, modes_time, modes_space)
        self.local = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.local(x)))


class FourierWaveformSurrogate(nn.Module):
    """Coordinate-aware Fourier neural operator for clean Deepwave records."""

    def __init__(self, n_time: int = 179, n_traces: int = 137, reflectivity_features: bool = True):
        super().__init__()
        self.output_shape = (n_time, n_traces)
        self.reflectivity_features = bool(reflectivity_features)
        width = 32
        input_channels = 5 if self.reflectivity_features else 3
        self.lift = nn.Conv2d(input_channels, width, 1)
        self.blocks = nn.ModuleList([FourierBlock(width, 24, 18) for _ in range(4)])
        self.head = nn.Sequential(
            nn.Conv2d(width, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Tanh(),
        )

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        x = waveform_surrogate_features(
            model,
            self.output_shape,
            reflectivity_features=self.reflectivity_features,
            physics_features=False,
        )
        x = self.lift(x)
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


class WaveformSurrogate(nn.Module):
    """Best validated coordinate-aware U-Net surrogate for clean Deepwave records."""

    def __init__(self, n_time: int = 179, n_traces: int = 137, reflectivity_features: bool = True):
        super().__init__()
        self.output_shape = (n_time, n_traces)
        self.reflectivity_features = bool(reflectivity_features)
        input_channels = 5 if self.reflectivity_features else 3
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(input_channels, 48)
        self.e2 = ConvBlock(48, 96)
        self.e3 = ConvBlock(96, 192)
        self.e4 = ConvBlock(192, 384)
        self.bottleneck = nn.Sequential(
            ConvBlock(384, 384),
            nn.Conv2d(384, 384, 3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.u3 = UpBlock(384, 384, 192)
        self.u2 = UpBlock(192, 192, 96)
        self.u1 = UpBlock(96, 96, 48)
        self.u0 = UpBlock(48, 48, 48)
        self.head = nn.Sequential(ConvBlock(48, 48), nn.Conv2d(48, 1, 1), nn.Tanh())

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(model, size=self.output_shape, mode="bilinear", align_corners=False)
        batch, _, height, width = x.shape
        z = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype).view(1, 1, height, 1)
        lateral = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype).view(1, 1, 1, width)
        features = [x]
        if self.reflectivity_features:
            gy = F.pad(model[..., 1:, :] - model[..., :-1, :], (0, 0, 0, 1))
            gx = F.pad(model[..., :, 1:] - model[..., :, :-1], (0, 1, 0, 0))
            features.extend(
                [
                    F.interpolate(gy, size=self.output_shape, mode="bilinear", align_corners=False),
                    F.interpolate(gx, size=self.output_shape, mode="bilinear", align_corners=False),
                ]
            )
        features.extend([z.expand(batch, 1, height, width), lateral.expand(batch, 1, height, width)])
        x = torch.cat(features, dim=1)
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(e4)
        x = self.u3(b, e4)
        x = self.u2(x, e3)
        x = self.u1(x, e2)
        x = self.u0(x, e1)
        return self.head(x)


def waveform_surrogate_features(
    model: torch.Tensor,
    output_shape: tuple[int, int],
    reflectivity_features: bool = True,
    physics_features: bool = False,
) -> torch.Tensor:
    x = F.interpolate(model, size=output_shape, mode="bilinear", align_corners=False)
    batch, _, height, width = x.shape
    z = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype).view(1, 1, height, 1)
    lateral = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype).view(1, 1, 1, width)
    features = [x]
    if reflectivity_features:
        gy = F.pad(model[..., 1:, :] - model[..., :-1, :], (0, 0, 0, 1))
        gx = F.pad(model[..., :, 1:] - model[..., :, :-1], (0, 1, 0, 0))
        features.extend(
            [
                F.interpolate(gy, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(gx, size=output_shape, mode="bilinear", align_corners=False),
            ]
        )
    if physics_features:
        eps = 1.0 + 8.0 * model.clamp(0.0, 1.0)
        slowness = torch.sqrt(eps)
        velocity = torch.rsqrt(eps)
        impedance_reflect_z = F.pad(
            (slowness[..., 1:, :] - slowness[..., :-1, :])
            / (slowness[..., 1:, :] + slowness[..., :-1, :]).clamp_min(1.0e-6),
            (0, 0, 0, 1),
        )
        impedance_reflect_x = F.pad(
            (slowness[..., :, 1:] - slowness[..., :, :-1])
            / (slowness[..., :, 1:] + slowness[..., :, :-1]).clamp_min(1.0e-6),
            (0, 1, 0, 0),
        )
        travel_time = torch.cumsum(slowness, dim=-2)
        travel_time = travel_time / travel_time.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        features.extend(
            [
                F.interpolate(slowness / 3.0, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(velocity, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(impedance_reflect_z, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(impedance_reflect_x, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(2.0 * travel_time - 1.0, size=output_shape, mode="bilinear", align_corners=False),
            ]
        )
    features.extend([z.expand(batch, 1, height, width), lateral.expand(batch, 1, height, width)])
    return torch.cat(features, dim=1)


class PhysicsWaveformSurrogate(nn.Module):
    """Wider U-Net surrogate with wave-physics input features."""

    def __init__(self, n_time: int = 179, n_traces: int = 137, reflectivity_features: bool = True):
        super().__init__()
        self.output_shape = (n_time, n_traces)
        self.reflectivity_features = bool(reflectivity_features)
        input_channels = 10 if self.reflectivity_features else 8
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(input_channels, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 512),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.u3 = UpBlock(512, 512, 256)
        self.u2 = UpBlock(256, 256, 128)
        self.u1 = UpBlock(128, 128, 64)
        self.u0 = UpBlock(64, 64, 64)
        self.head = nn.Sequential(ConvBlock(64, 64), nn.Conv2d(64, 1, 1), nn.Tanh())

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        x = waveform_surrogate_features(
            model,
            self.output_shape,
            reflectivity_features=self.reflectivity_features,
            physics_features=True,
        )
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(e4)
        x = self.u3(b, e4)
        x = self.u2(x, e3)
        x = self.u1(x, e2)
        x = self.u0(x, e1)
        return self.head(x)


class BaseConditionedResidualRefiner(nn.Module):
    """Residual refiner that conditions on the current surrogate prediction."""

    def __init__(
        self,
        n_time: int = 179,
        n_traces: int = 137,
        reflectivity_features: bool = True,
        correction_scale: float = 0.35,
        correction_start_fraction: float = 0.0,
        correction_ramp_width_fraction: float = 0.15,
    ):
        super().__init__()
        self.output_shape = (n_time, n_traces)
        self.reflectivity_features = bool(reflectivity_features)
        self.correction_scale = float(correction_scale)
        self.correction_start_fraction = float(correction_start_fraction)
        self.correction_ramp_width_fraction = float(correction_ramp_width_fraction)
        input_channels = (5 if self.reflectivity_features else 3) + 1
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(input_channels, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.e4 = ConvBlock(128, 256)
        self.bottleneck = nn.Sequential(
            ConvBlock(256, 256),
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.u3 = UpBlock(256, 256, 128)
        self.u2 = UpBlock(128, 128, 64)
        self.u1 = UpBlock(64, 64, 32)
        self.u0 = UpBlock(32, 32, 32)
        self.head = nn.Sequential(ConvBlock(32, 32), nn.Conv2d(32, 1, 1))
        self.zero_output_head()

    def zero_output_head(self) -> None:
        for module in reversed(list(self.head.modules())):
            if isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return

    def forward(self, model: torch.Tensor, base_prediction: torch.Tensor) -> torch.Tensor:
        features = waveform_surrogate_features(
            model,
            self.output_shape,
            reflectivity_features=self.reflectivity_features,
            physics_features=False,
        )
        if base_prediction.shape[-2:] != self.output_shape:
            base_prediction = F.interpolate(
                base_prediction,
                size=self.output_shape,
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([features, base_prediction], dim=1)
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(e4)
        x = self.u3(b, e4)
        x = self.u2(x, e3)
        x = self.u1(x, e2)
        x = self.u0(x, e1)
        correction = self.correction_scale * torch.tanh(self.head(x))
        if self.correction_start_fraction > 0.0:
            t = torch.linspace(
                0.0,
                1.0,
                self.output_shape[0],
                device=correction.device,
                dtype=correction.dtype,
            ).view(1, 1, self.output_shape[0], 1)
            ramp = ((t - self.correction_start_fraction) / max(self.correction_ramp_width_fraction, 1.0e-6)).clamp(
                0.0,
                1.0,
            )
            correction = correction * ramp
        return correction


SURROGATE_REGISTRY = {
    "FourierWaveformSurrogate": FourierWaveformSurrogate,
    "WaveformSurrogate": WaveformSurrogate,
    "PhysicsWaveformSurrogate": PhysicsWaveformSurrogate,
}


def build_waveform_surrogate(
    name: str = "WaveformSurrogate",
    n_time: int = 179,
    n_traces: int = 137,
    reflectivity_features: bool = True,
) -> nn.Module:
    try:
        constructor = SURROGATE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown waveform surrogate {name!r}; choose from {sorted(SURROGATE_REGISTRY)}") from exc
    return constructor(n_time=n_time, n_traces=n_traces, reflectivity_features=reflectivity_features)


MODEL_REGISTRY = {
    "unet": UNetInversion,
    "unetpp": UNetPPInversion,
    "transunet": TransUNetInversion,
    "tinynet": TinyInversion,
}


def build_inversion_model(name: str, output_size: int = 256, dropout: float = 0.10) -> nn.Module:
    try:
        constructor = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown backbone {name!r}; choose from {sorted(MODEL_REGISTRY)}") from exc
    return constructor(output_size=output_size, dropout=dropout)


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            module.train()
