import torch
import torch.nn as nn


def build_mlp(input_dim, hidden_dims, output_dim, activation_cls=nn.SiLU):
    layers = []
    curr_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(curr_dim, hidden_dim))
        layers.append(activation_cls())
        curr_dim = hidden_dim
    layers.append(nn.Linear(curr_dim, output_dim))
    return nn.Sequential(*layers)


class PersistenceBaseline(nn.Module):
    """Predicts x(t+1) = x(t)."""

    def forward(self, x0, u_seq, x_seq=None, teacher_forcing=False, **kwargs):
        seq_len = u_seq.size(1)
        if teacher_forcing and x_seq is not None:
            return x_seq[:, :seq_len, :]
        return x0.unsqueeze(1).repeat(1, seq_len, 1)


class LinearBaseline(nn.Module):
    """Joint linear baseline on the full multi-region observation vector."""

    def __init__(self, input_dim, output_dim, dt=0.01, *args, **kwargs):
        super().__init__()
        self.dt = dt
        self.linear = nn.Linear(input_dim, output_dim)

    def step(self, x_input, u_t):
        dx = self.linear(torch.cat([x_input, u_t], dim=-1))
        return x_input + self.dt * dx

    def forward(self, x0, u_seq, x_seq=None, teacher_forcing=False, **kwargs):
        seq_len = u_seq.size(1)
        preds = []
        x_curr = x0
        for t in range(seq_len):
            x_input = x_seq[:, t, :] if teacher_forcing and x_seq is not None else x_curr
            x_curr = self.step(x_input, u_seq[:, t, :])
            preds.append(x_curr)
        return torch.stack(preds, dim=1)


class IndependentPerRegionBaseline(nn.Module):
    """Trains one small model per region using x_r(t) and shared u(t)."""

    def __init__(self, obs_dim, hidden_dim=16, dt=0.01, *args, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.dt = dt
        self.region_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(obs_dim)
            ]
        )

    def step(self, x_input, u_t):
        outputs = []
        for region_idx, region_net in enumerate(self.region_nets):
            region_input = torch.cat([x_input[:, region_idx : region_idx + 1], u_t], dim=-1)
            dx = region_net(region_input)
            outputs.append(x_input[:, region_idx : region_idx + 1] + self.dt * dx)
        return torch.cat(outputs, dim=-1)

    def forward(self, x0, u_seq, x_seq=None, teacher_forcing=False, **kwargs):
        seq_len = u_seq.size(1)
        preds = []
        x_curr = x0
        for t in range(seq_len):
            x_input = x_seq[:, t, :] if teacher_forcing and x_seq is not None else x_curr
            x_curr = self.step(x_input, u_seq[:, t, :])
            preds.append(x_curr)
        return torch.stack(preds, dim=1)


class MLPBaseline(nn.Module):
    """Joint nonlinear baseline on the full multi-region observation vector."""

    def __init__(self, hidden_dims, input_dim, output_dim, dt=0.01, *args, **kwargs):
        super().__init__()
        self.dt = dt
        self.net = build_mlp(input_dim, hidden_dims, output_dim)

    def step(self, x_input, u_t):
        dx = self.net(torch.cat([x_input, u_t], dim=-1))
        return x_input + self.dt * dx

    def forward(self, x0, u_seq, x_seq=None, teacher_forcing=False, **kwargs):
        seq_len = u_seq.size(1)
        preds = []
        x_curr = x0
        for t in range(seq_len):
            x_input = x_seq[:, t, :] if teacher_forcing and x_seq is not None else x_curr
            x_curr = self.step(x_input, u_seq[:, t, :])
            preds.append(x_curr)
        return torch.stack(preds, dim=1)


class MechanisticModel(nn.Module):
    """
    Coupled multi-region Wilson-Cowan-style model with one latent E/I pair per region.
    Observation remains one scalar signal per region.
    """

    def __init__(
        self,
        obs_dim,
        dt=0.01,
        activation_function="sigmoid",
        observation_model="E_only",
        use_cross_region_coupling=True,
        cross_coupling_mode="full",
        local_coupling_init_scale=0.2,
        cross_coupling_init_scale=0.05,
        substeps=5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = 2 * obs_dim
        self.dt = dt
        self.substeps = substeps
        self.use_cross_region_coupling = use_cross_region_coupling
        self.cross_coupling_mode = cross_coupling_mode
        self.observation_model = observation_model
        self.activation_name = activation_function

        self.encoder = nn.Linear(obs_dim, self.latent_dim)
        with torch.no_grad():
            self.encoder.weight.zero_()
            self.encoder.bias.zero_()
            self.encoder.weight[:obs_dim, :obs_dim] = torch.eye(obs_dim)

        self.w_EE_local = nn.Parameter(local_coupling_init_scale * torch.ones(obs_dim))
        self.w_EI_local = nn.Parameter(local_coupling_init_scale * torch.ones(obs_dim))
        self.w_IE_local = nn.Parameter(local_coupling_init_scale * torch.ones(obs_dim))
        self.w_II_local = nn.Parameter(0.5 * local_coupling_init_scale * torch.ones(obs_dim))

        self.W_EE_cross = nn.Parameter(cross_coupling_init_scale * torch.randn(obs_dim, obs_dim))
        self.W_EI_cross = nn.Parameter(cross_coupling_init_scale * torch.randn(obs_dim, obs_dim))
        self.W_IE_cross = nn.Parameter(cross_coupling_init_scale * torch.randn(obs_dim, obs_dim))
        self.W_II_cross = nn.Parameter(cross_coupling_init_scale * torch.randn(obs_dim, obs_dim))

        self.w_EU = nn.Parameter(0.1 * torch.ones(obs_dim))
        self.w_IU = nn.Parameter(0.1 * torch.ones(obs_dim))
        self.b_E = nn.Parameter(torch.zeros(obs_dim))
        self.b_I = nn.Parameter(torch.zeros(obs_dim))

        if observation_model == "linear_combination":
            self.obs_E = nn.Parameter(torch.ones(obs_dim))
            self.obs_I = nn.Parameter(torch.zeros(obs_dim))

        self.last_latent_smoothness = torch.tensor(0.0)

    def activation_function(self, x):
        if self.activation_name == "tanh":
            return torch.tanh(x)
        return torch.sigmoid(x)

    def encode_observation(self, x_obs):
        return self.encoder(x_obs)

    def split_latent(self, z):
        return z[:, : self.obs_dim], z[:, self.obs_dim :]

    def cross_matrix(self, weight_matrix):
        mask = 1.0 - torch.eye(self.obs_dim, device=weight_matrix.device, dtype=weight_matrix.dtype)
        return weight_matrix * mask

    def observe(self, z):
        E, I = self.split_latent(z)
        if self.observation_model == "linear_combination":
            return self.obs_E * E + self.obs_I * I
        return E

    def derivatives(self, z_curr, u_t):
        E, I = self.split_latent(z_curr)

        local_drive_E = self.w_EE_local * E - self.w_EI_local * I
        local_drive_I = self.w_IE_local * E - self.w_II_local * I

        cross_drive_E = torch.zeros_like(E)
        cross_drive_I = torch.zeros_like(I)

        if self.use_cross_region_coupling and self.obs_dim > 1 and self.cross_coupling_mode != "local_only":
            W_EE = self.cross_matrix(self.W_EE_cross)
            cross_drive_E = E @ W_EE.T

            if self.cross_coupling_mode == "full":
                W_EI = self.cross_matrix(self.W_EI_cross)
                W_IE = self.cross_matrix(self.W_IE_cross)
                W_II = self.cross_matrix(self.W_II_cross)
                cross_drive_E = cross_drive_E - I @ W_EI.T
                cross_drive_I = E @ W_IE.T - I @ W_II.T

        drive_E = local_drive_E + cross_drive_E + self.w_EU * u_t + self.b_E
        drive_I = local_drive_I + cross_drive_I + self.w_IU * u_t + self.b_I

        dE = -E + self.activation_function(drive_E)
        dI = -I + self.activation_function(drive_I)
        return torch.cat([dE, dI], dim=-1)

    def integrate_step(self, z_input, u_t, residual=None):
        z_curr = z_input
        dt_sub = self.dt / self.substeps
        if residual is None:
            residual = torch.zeros_like(z_curr)

        for _ in range(self.substeps):
            dz_mech = self.derivatives(z_curr, u_t)
            z_curr = z_curr + dt_sub * (dz_mech + residual)

        return z_curr

    def forward(self, x0, u_seq, x_seq=None, teacher_forcing=False, **kwargs):
        seq_len = u_seq.size(1)
        preds, E_seq, I_seq = [], [], []

        z_curr = self.encode_observation(x0)
        smoothness = torch.zeros((), device=x0.device)

        for t in range(seq_len):
            if teacher_forcing and x_seq is not None:
                z_input = self.encode_observation(x_seq[:, t, :])
            else:
                z_input = z_curr

            z_next = self.integrate_step(z_input, u_seq[:, t, :])
            preds.append(self.observe(z_next))
            E_next, I_next = self.split_latent(z_next)
            E_seq.append(E_next)
            I_seq.append(I_next)
            smoothness = smoothness + torch.mean((z_next - z_input) ** 2)
            z_curr = z_next

        self.last_latent_smoothness = smoothness / max(seq_len, 1)
        return torch.stack(preds, dim=1), torch.stack(E_seq, dim=1), torch.stack(I_seq, dim=1)

    def coupling_l2_penalty(self):
        penalty = (
            torch.mean(self.W_EE_cross**2)
            + torch.mean(self.W_EI_cross**2)
            + torch.mean(self.W_IE_cross**2)
            + torch.mean(self.W_II_cross**2)
        )
        return penalty

    def coupling_l1_penalty(self):
        penalty = (
            torch.mean(torch.abs(self.W_EE_cross))
            + torch.mean(torch.abs(self.W_EI_cross))
            + torch.mean(torch.abs(self.W_IE_cross))
            + torch.mean(torch.abs(self.W_II_cross))
        )
        return penalty

    def get_coupling_matrices(self):
        return {
            "W_EE_cross": self.cross_matrix(self.W_EE_cross).detach().cpu().numpy(),
            "W_EI_cross": self.cross_matrix(self.W_EI_cross).detach().cpu().numpy(),
            "W_IE_cross": self.cross_matrix(self.W_IE_cross).detach().cpu().numpy(),
            "W_II_cross": self.cross_matrix(self.W_II_cross).detach().cpu().numpy(),
            "w_EE_local": self.w_EE_local.detach().cpu().numpy(),
            "w_EI_local": self.w_EI_local.detach().cpu().numpy(),
            "w_IE_local": self.w_IE_local.detach().cpu().numpy(),
            "w_II_local": self.w_II_local.detach().cpu().numpy(),
        }


class HybridModel(nn.Module):
    """
    Multi-region Wilson-Cowan backbone plus a small shared residual network over the full latent state.
    """

    def __init__(
        self,
        obs_dim,
        dt=0.01,
        activation_function="sigmoid",
        observation_model="E_only",
        use_cross_region_coupling=True,
        cross_coupling_mode="full",
        local_coupling_init_scale=0.2,
        cross_coupling_init_scale=0.05,
        residual_hidden_dims=None,
        residual_scale_init=-2.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        if residual_hidden_dims is None:
            residual_hidden_dims = [64, 32]

        self.mechanistic = MechanisticModel(
            obs_dim=obs_dim,
            dt=dt,
            activation_function=activation_function,
            observation_model=observation_model,
            use_cross_region_coupling=use_cross_region_coupling,
            cross_coupling_mode=cross_coupling_mode,
            local_coupling_init_scale=local_coupling_init_scale,
            cross_coupling_init_scale=cross_coupling_init_scale,
        )

        residual_input_dim = self.mechanistic.latent_dim + 1 + self.mechanistic.latent_dim
        self.residual_norm = nn.LayerNorm(residual_input_dim)
        self.residual_net = build_mlp(
            residual_input_dim,
            residual_hidden_dims,
            self.mechanistic.latent_dim,
        )
        self.logit_alpha = nn.Parameter(
            torch.full((self.mechanistic.latent_dim,), float(residual_scale_init))
        )

        self.last_residual_magnitude = torch.tensor(0.0)
        self.last_latent_smoothness = torch.tensor(0.0)

    def residual_scale(self):
        return torch.sigmoid(self.logit_alpha).unsqueeze(0)

    def forward(self, x0, u_seq, x_seq=None, teacher_forcing=False, **kwargs):
        seq_len = u_seq.size(1)
        preds, E_seq, I_seq = [], [], []

        z_curr = self.mechanistic.encode_observation(x0)
        residual_magnitude = torch.zeros((), device=x0.device)
        smoothness = torch.zeros((), device=x0.device)

        for t in range(seq_len):
            if teacher_forcing and x_seq is not None:
                z_input = self.mechanistic.encode_observation(x_seq[:, t, :])
            else:
                z_input = z_curr

            u_t = u_seq[:, t, :]
            dz_mech = self.mechanistic.derivatives(z_input, u_t)
            residual_inputs = torch.cat([z_input, u_t, dz_mech], dim=-1)
            residual = torch.tanh(self.residual_net(self.residual_norm(residual_inputs)))
            scaled_residual = self.residual_scale() * residual

            z_next = self.mechanistic.integrate_step(z_input, u_t, residual=scaled_residual)
            preds.append(self.mechanistic.observe(z_next))
            E_next, I_next = self.mechanistic.split_latent(z_next)
            E_seq.append(E_next)
            I_seq.append(I_next)

            residual_magnitude = residual_magnitude + torch.mean(scaled_residual**2)
            smoothness = smoothness + torch.mean((z_next - z_input) ** 2)
            z_curr = z_next

        self.last_residual_magnitude = residual_magnitude / max(seq_len, 1)
        self.last_latent_smoothness = smoothness / max(seq_len, 1)
        return torch.stack(preds, dim=1), torch.stack(E_seq, dim=1), torch.stack(I_seq, dim=1)

    def coupling_l2_penalty(self):
        return self.mechanistic.coupling_l2_penalty()

    def coupling_l1_penalty(self):
        return self.mechanistic.coupling_l1_penalty()

    def get_coupling_matrices(self):
        return self.mechanistic.get_coupling_matrices()
