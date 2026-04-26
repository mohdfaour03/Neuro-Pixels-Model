"""Neural network models for baseline and explicit WC-residual dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    BASELINE_VARIANT,
    RESIDUAL_ONLY_VARIANT,
    WC_ONLY_VARIANT,
    normalize_variant_name,
)
from Network.components import (
    FourierFeatures,
    LatentStateInitializer,
    ObservationReadout,
    ResidualDynamics,
    WilsonCowanPhysics,
    ZeroResidualDynamics,
    euler_step,
    rk4_step,
)


class TrajectoryNet(nn.Module):
    """Original latent predictor used by the regularized baseline."""

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
        **_: object,
    ) -> None:
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


class BaselineLatentWCRegularized(nn.Module):
    """Preserved baseline: direct latent predictor plus WC regularizer."""

    variant = BASELINE_VARIANT

    def __init__(
        self,
        n_stim: int = 15,
        n_regions: int = 4,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.n_regions = n_regions
        self.trajectory = TrajectoryNet(n_stim=n_stim, n_regions=n_regions, **kwargs)
        self.wc = WilsonCowanPhysics(n_regions=n_regions, n_stim=n_stim)

    def get_residual_alpha(self) -> float:
        return 0.0

    def get_cross_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def forward(
        self,
        t: torch.Tensor,
        u: torch.Tensor,
        beh: torch.Tensor,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        pred, stim_out, E_raw, I_raw = self.trajectory(t, u, beh)
        E = torch.sigmoid(E_raw)
        I = torch.sigmoid(I_raw)
        latent_state = torch.cat([E, I], dim=-1)
        wc_derivative = self.wc(latent_state, u)
        residual_derivative = torch.zeros_like(wc_derivative)
        return {
            "pred": pred,
            "stim_out": stim_out,
            "stim_pred": torch.sigmoid(stim_out),
            "E_raw": E_raw,
            "I_raw": I_raw,
            "E_pred": E,
            "I_pred": I,
            "latent_state": latent_state,
            "wc_derivative": wc_derivative,
            "residual_derivative": residual_derivative,
            "total_derivative": wc_derivative,
            "residual_alpha": pred.new_tensor(self.get_residual_alpha()),
        }


class ExplicitWCUDE(nn.Module):
    """Explicit mechanistic rollout with optional neural residual."""

    def __init__(
        self,
        variant: str,
        n_regions: int = 4,
        n_stim: int = 15,
        n_behavioral: int = 2,
        residual_hidden: int = 64,
        residual_layers: int = 2,
        initial_hidden: int = 32,
        residual_scale: float = 0.10,
        use_behavior_in_residual: bool = False,
        readout_type: str = "excitatory_direct",
        integrator: str = "euler",
        **_: object,
    ) -> None:
        super().__init__()
        self.variant = normalize_variant_name(variant)
        self.n_regions = n_regions
        self.state_dim = n_regions * 2
        self.integrator = integrator
        self.use_behavior_in_residual = use_behavior_in_residual

        self.wc = None
        if self.variant != RESIDUAL_ONLY_VARIANT:
            self.wc = WilsonCowanPhysics(n_regions=n_regions, n_stim=n_stim)
        self.initializer = LatentStateInitializer(
            n_regions=n_regions,
            n_stim=n_stim,
            n_behavioral=n_behavioral,
            hidden_dim=initial_hidden,
        )
        self.readout = ObservationReadout(n_regions=n_regions, readout_type=readout_type)
        if self.variant == WC_ONLY_VARIANT:
            self.residual = ZeroResidualDynamics(state_dim=self.state_dim)
        else:
            self.residual = ResidualDynamics(
                state_dim=self.state_dim,
                n_stim=n_stim,
                n_behavioral=n_behavioral,
                hidden_dim=residual_hidden,
                n_layers=residual_layers,
                residual_scale=residual_scale,
                use_behavior=use_behavior_in_residual,
            )

    def get_residual_alpha(self) -> float:
        return float(getattr(self.residual, "residual_scale", 0.0))

    def get_cross_parameters(self) -> tuple[nn.Parameter, ...]:
        # The current explicit WC backbone has no explicit cross-region coupling matrix.
        return ()

    def initialize_state(
        self,
        y0: torch.Tensor,
        u0: torch.Tensor,
        behavior0: torch.Tensor,
    ) -> torch.Tensor:
        return self.initializer(y0, u0, behavior0)

    def compute_wc_derivative(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if self.variant == RESIDUAL_ONLY_VARIANT:
            return torch.zeros_like(state)
        if self.wc is None:
            return torch.zeros_like(state)
        return self.wc(state.unsqueeze(0), u.unsqueeze(0)).squeeze(0)

    def compute_residual_derivative(
        self,
        state: torch.Tensor,
        u: torch.Tensor,
        behavior: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant == WC_ONLY_VARIANT:
            return torch.zeros_like(state)

        residual = self.residual(
            state.unsqueeze(0),
            u.unsqueeze(0),
            behavior.unsqueeze(0),
        ).squeeze(0)
        return residual

    def compute_total_derivative(
        self,
        state: torch.Tensor,
        u: torch.Tensor,
        behavior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wc_derivative = self.compute_wc_derivative(state, u)
        residual_derivative = self.compute_residual_derivative(state, u, behavior)
        total_derivative = wc_derivative + residual_derivative
        return wc_derivative, residual_derivative, total_derivative

    def readout_observation(self, state: torch.Tensor) -> torch.Tensor:
        return self.readout(state)

    def _integrate_step(
        self,
        state: torch.Tensor,
        u: torch.Tensor,
        behavior: torch.Tensor,
        dt: float,
        total_derivative: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.integrator == "euler":
            derivative = total_derivative
            if derivative is None:
                _, _, derivative = self.compute_total_derivative(state, u, behavior)
            next_state = euler_step(state, dt, derivative)
        elif self.integrator == "rk4":
            next_state = rk4_step(
                state,
                dt,
                lambda current_state: self.compute_total_derivative(current_state, u, behavior)[2],
            )
        else:
            raise ValueError(f"Unsupported integrator: {self.integrator}")

        return torch.clamp(next_state, 0.0, 1.0)

    def rollout(
        self,
        u: torch.Tensor,
        behavior: torch.Tensor,
        y0: torch.Tensor,
        dt: float,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        n_steps = u.shape[0]
        state = initial_state
        if state is None:
            state = self.initialize_state(y0, u[0], behavior[0])

        states = []
        predictions = []
        wc_terms = []
        residual_terms = []
        total_terms = []

        for step in range(n_steps):
            wc_derivative, residual_derivative, total_derivative = self.compute_total_derivative(
                state,
                u[step],
                behavior[step],
            )
            states.append(state)
            predictions.append(self.readout_observation(state))
            wc_terms.append(wc_derivative)
            residual_terms.append(residual_derivative)
            total_terms.append(total_derivative)

            if step < n_steps - 1:
                state = self._integrate_step(
                    state,
                    u[step],
                    behavior[step],
                    dt,
                    total_derivative=total_derivative,
                )

        latent_state = torch.stack(states, dim=0)
        pred = torch.stack(predictions, dim=0)
        wc_derivative = torch.stack(wc_terms, dim=0)
        residual_derivative = torch.stack(residual_terms, dim=0)
        total_derivative = torch.stack(total_terms, dim=0)
        return {
            "pred": pred,
            "latent_state": latent_state,
            "E_pred": latent_state[:, : self.n_regions],
            "I_pred": latent_state[:, self.n_regions :],
            "wc_derivative": wc_derivative,
            "residual_derivative": residual_derivative,
            "total_derivative": total_derivative,
            "residual_alpha": pred.new_tensor(self.get_residual_alpha()),
        }

    def forward(
        self,
        t: torch.Tensor,
        u: torch.Tensor,
        beh: torch.Tensor,
        y0: torch.Tensor,
        dt: float,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del t
        return self.rollout(
            u=u,
            behavior=beh,
            y0=y0,
            dt=dt,
            initial_state=initial_state,
        )


def build_model(model_config) -> nn.Module:
    variant = normalize_variant_name(model_config.variant)
    kwargs = dict(model_config.__dict__)
    kwargs["variant"] = variant

    if variant == BASELINE_VARIANT:
        kwargs.pop("variant", None)
        return BaselineLatentWCRegularized(**kwargs)

    return ExplicitWCUDE(**kwargs)
