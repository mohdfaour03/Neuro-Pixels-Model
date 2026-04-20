import torch
import torch.nn as nn

class PersistenceBaseline(nn.Module):
    """Predicts x(t+1) = x(t) autoregressively (flatline logic since it ignores input)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x0, u_seq, x_seq=None, **kwargs):
        seq_len = u_seq.size(1)
        preds = []
        x_curr = x0
        for _ in range(seq_len):
            preds.append(x_curr)
            # x_curr stays at x0 for persistence
        return torch.stack(preds, dim=1)

class LinearBaseline(nn.Module):
    """Predicts dx/dt using a simple linear mapping sequentially, operating as an Integrator."""
    def __init__(self, input_dim=3, output_dim=2, dt=0.01, *args, **kwargs):
        super().__init__()
        self.dt = dt
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x0, u_seq, x_seq=None, **kwargs):
        seq_len = u_seq.size(1)
        preds = []
        x_curr = x0
        for t in range(seq_len):
            u_t = u_seq[:, t, :]
            inputs = torch.cat([x_curr, u_t], dim=-1)
            dx = self.linear(inputs)
            x_next = x_curr + self.dt * dx
            preds.append(x_next)
            x_curr = x_next
        return torch.stack(preds, dim=1)

class MLPBaseline(nn.Module):
    """Predicts dx/dt using an MLP iteratively, acting natively as a Neural ODE."""
    def __init__(self, hidden_dims=[32, 16], input_dim=3, output_dim=2, dt=0.01, *args, **kwargs):
        super().__init__()
        self.dt = dt
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.SiLU())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x0, u_seq, x_seq=None, **kwargs):
        seq_len = u_seq.size(1)
        preds = []
        x_curr = x0
        for t in range(seq_len):
            u_t = u_seq[:, t, :]
            inputs = torch.cat([x_curr, u_t], dim=-1)
            dx = self.net(inputs)
            x_next = x_curr + self.dt * dx
            preds.append(x_next)
            x_curr = x_next
        return torch.stack(preds, dim=1)

class MechanisticModel(nn.Module):
    """
    Wilson-Cowan inspired dynamical model explicitly rolled out natively over sequence lengths via BPTT.
    """
    def __init__(self, dt=0.01, activation_function='sigmoid', *args, **kwargs):
        super().__init__()
        self.dt = dt
        self.activation_function = torch.sigmoid if activation_function == 'sigmoid' else torch.tanh
        
        # Core connectivity weights
        self.w_EE = nn.Parameter(torch.tensor(0.5))
        self.w_EI = nn.Parameter(torch.tensor(0.5))
        self.w_IE = nn.Parameter(torch.tensor(0.5))
        self.w_II = nn.Parameter(torch.tensor(0.1))
        
        # External Input weights
        self.w_EU = nn.Parameter(torch.tensor(0.1))
        self.w_IU = nn.Parameter(torch.tensor(0.1))
        
        # Firing thresholds (biases)
        self.b_E = nn.Parameter(torch.tensor(0.0))
        self.b_I = nn.Parameter(torch.tensor(0.0))

    def derivatives(self, x_curr, u_t):
        E_curr = x_curr[:, 0:1]
        I_curr = x_curr[:, 1:2]

        S_E = self.activation_function(self.w_EE * E_curr - self.w_EI * I_curr + self.w_EU * u_t + self.b_E)
        dE = -E_curr + S_E

        S_I = self.activation_function(self.w_IE * E_curr - self.w_II * I_curr + self.w_IU * u_t + self.b_I)
        dI = -I_curr + S_I
        return dE, dI

    def step(self, x_curr, u_t):
        E_curr = x_curr[:, 0:1]
        I_curr = x_curr[:, 1:2]
        
        steps = 5
        dt_sub = self.dt / steps
        for _ in range(steps):
            dE, dI = self.derivatives(torch.cat([E_curr, I_curr], dim=-1), u_t)
            
            E_curr = E_curr + dt_sub * dE
            I_curr = I_curr + dt_sub * dI
            
        x_next = torch.cat([E_curr, I_curr], dim=-1)
        return x_next, E_curr, I_curr
        
    def forward(self, x0, u_seq, x_seq=None, **kwargs):
        seq_len = u_seq.size(1)
        preds, E_seq, I_seq = [], [], []
        x_curr = x0
        for t in range(seq_len):
            u_t = u_seq[:, t, :]
            x_next, E_next, I_next = self.step(x_curr, u_t)
            preds.append(x_next)
            E_seq.append(E_next)
            I_seq.append(I_next)
            x_curr = x_next
            
        return torch.stack(preds, dim=1), torch.stack(E_seq, dim=1), torch.stack(I_seq, dim=1)

class HybridModel(nn.Module):
    """
    Wilson-Cowan mechanistic core plus a recurrent residual correction.
    When observed state history is available, the model predicts one-step-ahead
    transitions from that observed state distribution consistently at both train
    and eval time. Without history, it falls back to autoregressive rollout.
    """
    def __init__(self, dt=0.01, activation_function='sigmoid', residual_hidden_dims=[16], pretrained_mechanistic_path=None, *args, **kwargs):
        super().__init__()
        self.mechanistic = MechanisticModel(dt=dt, activation_function=activation_function)
        
        if pretrained_mechanistic_path is not None:
            self.mechanistic.load_state_dict(torch.load(pretrained_mechanistic_path))
            for param in self.mechanistic.parameters():
                param.requires_grad = False
                
        hidden_dim = residual_hidden_dims[0]
        self.residual_lstm = nn.LSTMCell(input_size=5, hidden_size=hidden_dim) # E, I, U, dE_mech, dI_mech
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.residual_proj = nn.Linear(hidden_dim, 2)
        
        # Keep residual authority positive and bounded so the hybrid stays physics-first.
        self.logit_alpha = nn.Parameter(torch.tensor(-1.5))
        
        # Initialize projection very small so residual starts as a gentle correction.
        nn.init.normal_(self.residual_proj.weight, std=0.01)
        nn.init.zeros_(self.residual_proj.bias)

    def residual_scale(self):
        return torch.sigmoid(self.logit_alpha)
        
    def step(self, x_curr, u_t, dE_res, dI_res):
        E_curr = x_curr[:, 0:1]
        I_curr = x_curr[:, 1:2]
        
        steps = 5
        dt_sub = self.mechanistic.dt / steps
        alpha = self.residual_scale()
        for _ in range(steps):
            dE_mech, dI_mech = self.mechanistic.derivatives(torch.cat([E_curr, I_curr], dim=-1), u_t)
            
            E_curr = E_curr + dt_sub * (dE_mech + alpha * dE_res)
            I_curr = I_curr + dt_sub * (dI_mech + alpha * dI_res)
        
        x_next = torch.cat([E_curr, I_curr], dim=-1)
        
        # Track correction size so training can discourage the residual from taking over.
        res_mag = torch.mean((alpha * dE_res)**2 + (alpha * dI_res)**2)
        
        return x_next, E_curr, I_curr, res_mag
        
    def forward(self, x0, u_seq, x_seq=None, **kwargs):
        seq_len = u_seq.size(1)
        batch_size = x0.size(0)
        preds, E_seq, I_seq = [], [], []
        x_curr = x0
        self.last_residual_magnitude = 0.0
        
        h_t = torch.zeros(batch_size, self.residual_lstm.hidden_size, device=x0.device)
        c_t = torch.zeros(batch_size, self.residual_lstm.hidden_size, device=x0.device)
        
        for t in range(seq_len):
            u_t = u_seq[:, t, :]
            # Align train and eval around the actual task: current observed state -> next state.
            # If an observed history window is provided, use it consistently. Otherwise roll out.
            x_input = x_seq[:, t, :] if x_seq is not None else x_curr

            dE_mech_base, dI_mech_base = self.mechanistic.derivatives(x_input, u_t)
            inputs = torch.cat([x_input, u_t, dE_mech_base, dI_mech_base], dim=-1)
            h_t, c_t = self.residual_lstm(inputs, (h_t, c_t))
            res_out = self.residual_proj(self.hidden_norm(h_t))
            
            # Bounded residuals keep the neural correction interpretable and stable.
            dE_res = torch.tanh(res_out[:, 0:1])
            dI_res = torch.tanh(res_out[:, 1:2])
            
            x_next, E_next, I_next, res_mag = self.step(x_input, u_t, dE_res, dI_res)
            preds.append(x_next)
            E_seq.append(E_next)
            I_seq.append(I_next)
            x_curr = x_next
            self.last_residual_magnitude = self.last_residual_magnitude + res_mag
            
        return torch.stack(preds, dim=1), torch.stack(E_seq, dim=1), torch.stack(I_seq, dim=1)

class LatentCTRNN(nn.Module):
    def __init__(self, hidden_dim=64, dt=0.01):
        super().__init__()
        self.dt = dt
        self.hidden_dim = hidden_dim
        
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_u = nn.Linear(1, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, 2)
        
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.1)
        self.encoder = nn.Linear(2, self.hidden_dim)
        
    def step(self, h_curr, u_t):
        steps = 5
        dt_sub = self.dt / steps
        for _ in range(steps):
            # tau * dh/dt = -h + tanh(W_h h + W_u u)
            dh = (-h_curr + torch.tanh(self.W_h(h_curr) + self.W_u(u_t))) / torch.clamp(self.tau, min=1e-3)
            h_curr = h_curr + dt_sub * dh
            
        x_next = self.W_out(h_curr)
        
        E_next = x_next[:, 0:1]
        I_next = x_next[:, 1:2]
        return x_next, E_next, I_next, h_curr

    def forward(self, x0, u_seq, x_seq=None, **kwargs):
        seq_len = u_seq.size(1)
        batch_size = x0.size(0)
        
        h_curr = self.encoder(x0)
        
        preds, E_seq, I_seq = [], [], []
        for t in range(seq_len):
            u_t = u_seq[:, t, :]
            x_next, E_next, I_next, h_curr = self.step(h_curr, u_t)
            preds.append(x_next)
            E_seq.append(E_next)
            I_seq.append(I_next)
            
        return torch.stack(preds, dim=1), torch.stack(E_seq, dim=1), torch.stack(I_seq, dim=1)


class LSTMBaseline(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # We now ingest the 2D state x AND 1D stimulus u directly!
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 2)
        
    def forward(self, x0, u_seq, x_seq=None, **kwargs):
        # Time-Series Forecasting: If historical window data is available, ingest it natively!
        if x_seq is not None:
            # Shift x_seq by 1 or just map exactly. 
            # In sliding window forecasting, we use historical x[0:t] to predict x[t+1]!
            inputs = torch.cat([x_seq, u_seq], dim=-1).contiguous()
            lstm_out, _ = self.lstm(inputs)
            preds = self.fc_out(lstm_out)
            
            E_seq = preds[:, :, 0:1]
            I_seq = preds[:, :, 1:2]
            return preds, E_seq, I_seq
            
        else:
            # If strictly autonomous forecasting required without true history
            seq_len = u_seq.size(1)
            batch_size = x0.size(0)
            
            h_t = torch.zeros(1, batch_size, self.hidden_dim, device=x0.device)
            c_t = torch.zeros(1, batch_size, self.hidden_dim, device=x0.device)
            
            x_curr = x0
            preds, E_seq, I_seq = [], [], []
            for t in range(seq_len):
                u_t = u_seq[:, t, :]
                inputs = torch.cat([x_curr, u_t], dim=-1).unsqueeze(1)
                lstm_out, (h_t, c_t) = self.lstm(inputs, (h_t, c_t))
                
                x_next = self.fc_out(lstm_out.squeeze(1))
                
                E_next = x_next[:, 0:1]
                I_next = x_next[:, 1:2]
                
                preds.append(x_next)
                E_seq.append(E_next)
                I_seq.append(I_next)
                x_curr = x_next
                
            return torch.stack(preds, dim=1), torch.stack(E_seq, dim=1), torch.stack(I_seq, dim=1)
