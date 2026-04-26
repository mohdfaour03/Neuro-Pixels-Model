"""Mechanistic and neural components for latent dynamics models."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Log-spaced Fourier time features for the baseline path."""

    def __init__(
        self,
        n_freq: int = 64,
        min_freq: float = 1.0,
        max_freq: float = 1000.0,
    ) -> None:
        super().__init__()
        freqs = torch.logspace(math.log10(min_freq), math.log10(max_freq), n_freq)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_proj = 2 * np.pi * t @ self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class WilsonCowanPhysics(nn.Module):
    """Wilson-Cowan derivative for multi-region excitatory/inhibitory states."""

    def __init__(self, n_regions: int = 4, n_stim: int = 15) -> None:
        super().__init__()
        self.n_regions = n_regions
        self.n_stim = n_stim

        self.tau_E = nn.Parameter(torch.full((n_regions,), 1.0))
        self.tau_I = nn.Parameter(torch.full((n_regions,), 1.0))
        self.w_EE = nn.Parameter(torch.full((n_regions,), 5.0))
        self.w_EI = nn.Parameter(torch.full((n_regions,), 4.0))
        self.w_IE = nn.Parameter(torch.full((n_regions,), 3.0))
        self.w_II = nn.Parameter(torch.full((n_regions,), 1.0))
        self.w_uE = nn.Parameter(torch.randn(n_stim, n_regions) * 0.1)
        self.b_E = nn.Parameter(torch.zeros(n_regions))
        self.b_I = nn.Parameter(torch.zeros(n_regions))
        self.a = nn.Parameter(torch.full((n_regions,), 4.0))

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a.unsqueeze(0) + x)

    def split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return state[..., : self.n_regions], state[..., self.n_regions :]

    def combine_state(self, E: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        return torch.cat([E, I], dim=-1)

    def forward_ei(
        self,
        E: torch.Tensor,
        I: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tau_E = nn.functional.softplus(self.tau_E).unsqueeze(0) + 1e-4
        tau_I = nn.functional.softplus(self.tau_I).unsqueeze(0) + 1e-4
        stim_drive = u @ self.w_uE

        dEdt = (
            -E + self.sigmoid(self.w_EE * E - self.w_EI * I + stim_drive + self.b_E)
        ) / tau_E
        dIdt = (-I + self.sigmoid(self.w_IE * E - self.w_II * I + self.b_I)) / tau_I
        return dEdt, dIdt

    def forward(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        E, I = self.split_state(state)
        dEdt, dIdt = self.forward_ei(E, I, u)
        return self.combine_state(dEdt, dIdt)


class ResidualDynamics(nn.Module):
    """Explicit residual correction term r_phi(x, u, behavior)."""

    def __init__(
        self,
        state_dim: int,
        n_stim: int,
        n_behavioral: int,
        hidden_dim: int,
        n_layers: int,
        residual_scale: float,
        use_behavior: bool,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.use_behavior = use_behavior
        self.residual_scale = residual_scale

        input_dim = state_dim + n_stim + (n_behavioral if use_behavior else 0)
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(max(n_layers, 1)):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.Tanh())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, state_dim))
        self.network = nn.Sequential(*layers)

        final_linear = self.network[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def forward_raw(
        self,
        state: torch.Tensor,
        u: torch.Tensor,
        behavior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [state, u]
        if self.use_behavior and behavior is not None:
            parts.append(behavior)
        return torch.tanh(self.network(torch.cat(parts, dim=-1)))

    def forward(
        self,
        state: torch.Tensor,
        u: torch.Tensor,
        behavior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.residual_scale * self.forward_raw(state, u, behavior)


class ZeroResidualDynamics(nn.Module):
    """Residual ablation path with no learnable correction."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.residual_scale = 0.0

    def forward_raw(
        self,
        state: torch.Tensor,
        u: torch.Tensor,
        behavior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del u, behavior
        return torch.zeros_like(state)

    def forward(
        self,
        state: torch.Tensor,
        u: torch.Tensor,
        behavior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_raw(state, u, behavior)


class LatentStateInitializer(nn.Module):
    """Initialize latent inhibitory state while keeping E aligned to the observation."""

    def __init__(
        self,
        n_regions: int,
        n_stim: int,
        n_behavioral: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.n_regions = n_regions
        self.inhibitory_head = nn.Sequential(
            nn.Linear(n_regions + n_stim + n_behavioral, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_regions),
        )

        final_linear = self.inhibitory_head[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, -1.3862944)

    def forward(
        self,
        y0: torch.Tensor,
        u0: torch.Tensor,
        beh0: torch.Tensor,
    ) -> torch.Tensor:
        E0 = torch.clamp(y0, 1e-4, 1.0 - 1e-4)
        inhibitory_logits = self.inhibitory_head(torch.cat([y0, u0, beh0], dim=-1))
        I0 = torch.sigmoid(inhibitory_logits)
        return torch.cat([E0, I0], dim=-1)


class ObservationReadout(nn.Module):
    """Transparent mapping from latent state to observed activity."""

    def __init__(self, n_regions: int, readout_type: str = "excitatory_direct") -> None:
        super().__init__()
        self.n_regions = n_regions
        self.readout_type = readout_type
        if readout_type == "linear":
            self.linear = nn.Linear(n_regions, n_regions)
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(n_regions))
                self.linear.bias.zero_()
        elif readout_type != "excitatory_direct":
            raise ValueError(f"Unsupported readout_type: {readout_type}")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        E = state[..., : self.n_regions]
        if self.readout_type == "excitatory_direct":
            return E
        return self.linear(E)


def euler_step(state: torch.Tensor, dt: float, derivative: torch.Tensor) -> torch.Tensor:
    return state + dt * derivative


def rk4_step(
    state: torch.Tensor,
    dt: float,
    derivative_fn,
) -> torch.Tensor:
    k1 = derivative_fn(state)
    k2 = derivative_fn(state + 0.5 * dt * k1)
    k3 = derivative_fn(state + 0.5 * dt * k2)
    k4 = derivative_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
