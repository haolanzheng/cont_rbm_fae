"""Compact UNO--FiLM FAE with coordinate-dependent signed kernel pooling.

Pool construction follows the MLPKernelPooling idea in the official
functional_autoencoders code (Bunker et al., JMLR 2025). This is an independent
PyTorch implementation using explicit quadrature, not an exact paper replica.
Only the active components are retained; existing state_dict keys are unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

class SharedPositionEncoding(nn.Module):
    """Shared trainable continuous position encoding on the unit square."""

    def __init__(self, num_frequencies: int = 6):
        super().__init__()
        frequencies = math.pi * 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("base_frequencies", frequencies)
        self.frequency_offsets = nn.Parameter(torch.zeros(num_frequencies))
        self.phase_x = nn.Parameter(torch.zeros(num_frequencies))
        self.phase_y = nn.Parameter(torch.zeros(num_frequencies))
        # x, y, two boundary-distance features, and sin/cos for x and y.
        self.out_channels = 4 + 4 * num_frequencies
        self.refinement = nn.Sequential(
            nn.Linear(self.out_channels, 2 * self.out_channels),
            nn.GELU(),
            nn.Linear(2 * self.out_channels, self.out_channels),
        )
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def encode_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Encode [..., 2] coordinates normalized to [-1, 1]^2."""
        xx, yy = coordinates[..., 0], coordinates[..., 1]
        base = torch.stack((xx, yy, 1.0 - xx.square(), 1.0 - yy.square()), dim=-1)
        base_frequencies = self.base_frequencies.to(device=coordinates.device, dtype=coordinates.dtype)
        offsets = self.frequency_offsets.to(dtype=coordinates.dtype)
        frequencies = base_frequencies * torch.exp(0.25 * torch.tanh(offsets))
        phase_x = xx[..., None] * frequencies + self.phase_x.to(dtype=coordinates.dtype)
        phase_y = yy[..., None] * frequencies + self.phase_y.to(dtype=coordinates.dtype)
        periodic = torch.cat(
            (torch.sin(phase_x), torch.cos(phase_x), torch.sin(phase_y), torch.cos(phase_y)), dim=-1
        )
        encoded = torch.cat((base, periodic), dim=-1)
        return encoded + 0.1 * self.refinement(encoded)

    def grid_coordinates(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1)
        return coordinates[None].expand(batch_size, -1, -1, -1)

    def forward(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        coordinates = self.grid_coordinates(
            height, width, device=device, dtype=dtype, batch_size=batch_size
        )
        return self.encode_coordinates(coordinates).permute(0, 3, 1, 2)

class FiLMResidualBlock(nn.Module):
    """Smooth residual MLP block modulated by the complete latent code."""

    def __init__(self, width: int, condition_width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear_in = nn.Linear(width, 2 * width)
        self.film = nn.Linear(condition_width, 4 * width)
        self.linear_out = nn.Linear(2 * width, width)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.linear_in(self.norm(values))
        scale, shift = self.film(condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(scale[:, None])) + shift[:, None]
        hidden = F.gelu(hidden)
        return values + self.linear_out(hidden) / math.sqrt(2.0)

class FiLMCoordinateDecoder(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        position_channels: int,
        width: int,
        condition_width: int,
        blocks: int,
    ):
        super().__init__()
        self.condition = nn.Sequential(
            nn.LayerNorm(latent_dimension),
            nn.Linear(latent_dimension, condition_width),
            nn.GELU(),
            nn.Linear(condition_width, condition_width),
            nn.GELU(),
        )
        self.coordinate_lift = nn.Sequential(
            nn.Linear(position_channels, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(width, condition_width) for _ in range(blocks)]
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, code: torch.Tensor, position_features: torch.Tensor) -> torch.Tensor:
        condition = self.condition(code)
        values = self.coordinate_lift(position_features)
        for block in self.blocks:
            values = block(values, condition)
        return self.output(values)[..., 0]


def parameter_counts(module):
    parameters = [p for p in module.parameters() if p.requires_grad]
    return {
        "tensor_elements": sum(p.numel() for p in parameters),
        "real_scalars": sum(p.numel() * (2 if p.is_complex() else 1) for p in parameters),
    }


def trapezoid_weights(height, width, *, device=None, dtype=torch.float32):
    """Normalized tensor-product trapezoidal quadrature on [0,1]^2."""
    if min(height, width) < 2:
        raise ValueError("Quadrature needs at least two nodes along each axis")
    wy = torch.ones(height, device=device, dtype=dtype)
    wx = torch.ones(width, device=device, dtype=dtype)
    wy[[0, -1]] = 0.5
    wx[[0, -1]] = 0.5
    return (wy[:, None] * wx[None, :] / ((height - 1) * (width - 1))).flatten()


def fourier_transfer(values, shape, weight_positive, weight_negative):
    """Low-mode transfer between periodic grids, preserving function amplitude.

    Excludes ambiguous Nyquist modes. Forward normalization on both transforms
    makes the stored coefficients independent of the number of grid nodes.
    """
    ih, iw = values.shape[-2:]
    oh, ow = shape
    ft = torch.fft.rfft2(values, norm="forward")
    out = ft.new_zeros(*values.shape[:-2], oh, ow // 2 + 1)
    my = min(weight_positive.shape[-2], (ih - 1) // 2, (oh - 1) // 2)
    mx = min(weight_positive.shape[-1], (iw - 1) // 2 + 1, (ow - 1) // 2 + 1)
    if my < 1 or mx < 1:
        raise ValueError("Spatial grid is too small for spectral transfer")
    out[..., :my, :mx] = ft[..., :my, :mx] * weight_positive[None, :, :my, :mx]
    out[..., -my:, :mx] = ft[..., -my:, :mx] * weight_negative[None, :, :my, :mx]
    return torch.fft.irfft2(out, s=shape, norm="forward")


class ConsistentUNOOperatorLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_norm = nn.GroupNorm(1, in_channels)
        self.modes = modes
        self.spectral_rank = min(32, in_channels, out_channels)
        self.spectral_in = nn.Conv2d(in_channels, self.spectral_rank, kernel_size=1, bias=False)
        scale = 1.0 / math.sqrt(self.spectral_rank)
        self.weight_positive = nn.Parameter(
            scale * torch.randn(self.spectral_rank, modes, modes, dtype=torch.cfloat)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.randn(self.spectral_rank, modes, modes, dtype=torch.cfloat)
        )
        self.spectral_out = nn.Conv2d(self.spectral_rank, out_channels, kernel_size=1, bias=False)
        self.local = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_norm = nn.GroupNorm(1, out_channels)
        self.mixing = nn.Sequential(
            nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=1),
        )

    def forward(self, values, output_shape):
        normalized = self.in_norm(values)
        spectral = self.spectral_out(fourier_transfer(
            self.spectral_in(normalized), output_shape,
            self.weight_positive, self.weight_negative,
        ))
        local = self.local(F.interpolate(
            normalized, size=output_shape, mode="bilinear", align_corners=True,
        ))
        output = F.gelu(spectral + local)
        return output + self.mixing(self.out_norm(output))


class CompactUNOFeatures(nn.Module):
    def __init__(self, input_channels, width=16, modes=8, pool_channels=8):
        super().__init__()
        channels = [width, round(1.5 * width), 3 * width, 6 * width]
        self.lift = nn.Conv2d(input_channels, channels[0], 1)
        self.contract = nn.ModuleList([
            ConsistentUNOOperatorLayer(a, b, modes)
            for a, b in zip(channels[:-1], channels[1:])
        ])
        self.bottleneck = ConsistentUNOOperatorLayer(channels[-1], channels[-1], modes)
        self.project = nn.Sequential(
            nn.Conv2d(channels[-1], 32, 1), nn.GELU(),
            nn.Conv2d(32, pool_channels, 1),
        )

    def forward(self, values):
        height, width = values.shape[-2:]
        py, px = max(1, round(height * 0.1)), max(1, round(width * 0.1))
        features = F.pad(self.lift(values), (0, px, 0, py))
        for layer, factor in zip(self.contract, (0.75, 2 / 3, 0.5)):
            shape = tuple(max(5, round(n * factor)) for n in features.shape[-2:])
            features = layer(features, shape)
        features = self.bottleneck(features, tuple(features.shape[-2:]))
        features = self.project(features)
        # Evaluate the hidden field on the original physical domain before
        # integrating; padded nodes are not treated as physical observations.
        yy = torch.linspace(-1, 1, height, device=values.device, dtype=values.dtype)
        xx = torch.linspace(-1, 1, width, device=values.device, dtype=values.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        gx = (gx + 1) * (width - 1) / (width + px - 1) - 1
        gy = (gy + 1) * (height - 1) / (height + py - 1) - 1
        grid = torch.stack((gx, gy), -1)[None].expand(len(values), -1, -1, -1)
        return F.grid_sample(features, grid, mode="bilinear", align_corners=True)


class CoordinateKernelPooling(nn.Module):
    """z = W integral K_theta(x) H(x) dx; kernel may have either sign.

    Number of output components is independent of the number/order of nodes.
    `position` must be sampled from the single shared continuous encoding.
    """
    def __init__(self, position_channels, feature_channels=8, latent_dimension=128, hidden=64):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.feature_channels = feature_channels
        self.kernel = nn.Sequential(
            nn.Linear(position_channels, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dimension * feature_channels),
        )
        self.readout = nn.Linear(latent_dimension, latent_dimension, bias=False)
        nn.init.eye_(self.readout.weight)

    def forward(self, features, position, weights):
        # features [B,N,C], position [N,P], weights [N]. Kernel is shared by batch.
        n = features.shape[1]
        if position.shape[0] != n or weights.shape != (n,):
            raise ValueError("Feature, coordinate and quadrature sizes must agree")
        kernel = self.kernel(position).reshape(n, self.latent_dimension, self.feature_channels)
        weighted = kernel * weights[:, None, None]
        # Equivalent to einsum('bnc,nlc->bl'); one dense contraction is faster.
        z = features.reshape(len(features), -1) @ weighted.permute(0, 2, 1).reshape(-1, self.latent_dimension)
        return self.readout(z)


class KernelPoolFAE(nn.Module):
    def __init__(self, latent_dimension=128, width=16, modes=8, pool_channels=8,
                 kernel_hidden=64, decoder_width=96, decoder_blocks=3,
                 condition_width=96, coordinate_frequencies=6):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.position_encoding = SharedPositionEncoding(coordinate_frequencies)
        p = self.position_encoding.out_channels
        self.encoder = CompactUNOFeatures(1 + p, width, modes, pool_channels)
        self.pool = CoordinateKernelPooling(p, pool_channels, latent_dimension, kernel_hidden)
        self.decoder = FiLMCoordinateDecoder(latent_dimension, p, decoder_width, condition_width, decoder_blocks)

    def coordinates(self, shape, device, dtype):
        return self.position_encoding.grid_coordinates(
            *shape, device=device, dtype=dtype, batch_size=1,
        )[0].reshape(-1, 2)

    def encode(self, field):
        if field.ndim == 3:
            field = field[:, None]
        b, _, h, w = field.shape
        coordinates = self.coordinates((h, w), field.device, field.dtype)
        position = self.position_encoding.encode_coordinates(coordinates)
        extra = position.T.reshape(1, -1, h, w).expand(b, -1, -1, -1)
        features = self.encoder(torch.cat((field, extra), 1)).permute(0, 2, 3, 1).reshape(b, h*w, -1)
        weights = trapezoid_weights(h, w, device=field.device, dtype=field.dtype)
        return self.pool(features, position, weights)

    def decode_coordinates(self, code, coordinates):
        if coordinates.ndim == 2:
            position = self.position_encoding.encode_coordinates(coordinates)[None].expand(len(code), -1, -1)
        else:
            position = self.position_encoding.encode_coordinates(coordinates)
        return self.decoder(code, position)

    def decode(self, code, shape):
        coordinates = self.coordinates(shape, code.device, code.dtype)
        return self.decode_coordinates(code, coordinates).reshape(len(code), *shape)

    def forward(self, field, output_shape=None):
        code = self.encode(field)
        return self.decode(code, output_shape or tuple(field.shape[-2:])), code
