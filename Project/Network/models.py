"""Neural network and Wilson-Cowan physics modules."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(
        self,
        n_freq: int = 64,
        min_freq: float = 1.0,
        max_freq: float = 1000.0,
    ):
        super().__init__()
        freqs = torch.logspace(np.log10(min_freq), np.log10(max_freq), n_freq)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_proj = 2 * np.pi * t @ self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class WilsonCowanPhysics(nn.Module):
    def __init__(self, n_regions: int = 4, n_stim: int = 15):
        super().__init__()
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

    def forward(
        self,
        E: torch.Tensor,
        I: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tau_E = nn.functional.softplus(self.tau_E).unsqueeze(0)
        tau_I = nn.functional.softplus(self.tau_I).unsqueeze(0)
        stim_drive = u @ self.w_uE

        dEdt = (
            -E + self.sigmoid(self.w_EE * E - self.w_EI * I + stim_drive + self.b_E)
        ) / tau_E
        dIdt = (-I + self.sigmoid(self.w_IE * E - self.w_II * I + self.b_I)) / tau_I
        return dEdt, dIdt


class TrajectoryNet(nn.Module):
    def __init__(
        self,
        n_fourier: int = 128,
        fourier_freqs: int = 64,
        min_freq: float = 1.0,
        max_freq: float = 1000.0,
        n_stim: int = 15,
        n_regions: int = 4,
        n_behavioral: int = 2,
        hidden: int = 128,
        gru_hidden: int = 64,
        gru_layers: int = 2,
    ):
        super().__init__()
        self.n_regions = n_regions
        self.fourier = FourierFeatures(
            n_freq=fourier_freqs,
            min_freq=min_freq,
            max_freq=max_freq,
        )
        self.stim_head = nn.Sequential(
            nn.Linear(n_fourier + n_stim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_regions),
        )
        self.gru = nn.GRU(
            input_size=n_behavioral,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1 if gru_layers > 1 else 0.0,
        )
        self.gru_proj = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_regions * 2),
        )

    def forward_stim(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        features = self.fourier(t)
        return self.stim_head(torch.cat([features, u], dim=-1))

    def forward_internal(self, beh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gru_out, _ = self.gru(beh.unsqueeze(0))
        int_out = self.gru_proj(gru_out.squeeze(0))
        E_int = int_out[:, : self.n_regions]
        I_int = int_out[:, self.n_regions :]
        return E_int, I_int

    def forward(
        self,
        t: torch.Tensor,
        u: torch.Tensor,
        beh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stim_out = self.forward_stim(t, u)
        E_int, I_int = self.forward_internal(beh)
        combined = torch.sigmoid(stim_out + E_int)
        return combined, stim_out, E_int, I_int
