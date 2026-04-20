import torch
from torch.utils.data import DataLoader, Dataset


class NeuropixelsSequenceDataset(Dataset):
    """
    Generic sequence dataset for aligned multi-region observations x(t) and shared stimulus u(t).

    Returns:
        x_seq_in: [sequence_length, obs_dim]
        u_seq: [sequence_length, 1]
        target_seq: [sequence_length, obs_dim]
        time_seq: [sequence_length]
    """

    def __init__(self, x_t, u_t, time_vector, sequence_length=50):
        self.sequence_length = sequence_length
        self.valid_indices = len(x_t) - sequence_length

        self.x = torch.as_tensor(x_t, dtype=torch.float32)
        self.u = torch.as_tensor(u_t, dtype=torch.float32).unsqueeze(-1)
        self.time = time_vector

    def __len__(self):
        return max(0, self.valid_indices)

    def __getitem__(self, idx):
        x_seq_in = self.x[idx : idx + self.sequence_length]
        u_seq = self.u[idx : idx + self.sequence_length]
        target_seq = self.x[idx + 1 : idx + self.sequence_length + 1]
        time_seq = self.time[idx + 1 : idx + self.sequence_length + 1]
        return x_seq_in, u_seq, target_seq, time_seq


def create_dataloaders(
    x_t,
    u_t,
    time_vector,
    train_split=0.7,
    val_split=0.15,
    batch_size=64,
    sequence_length=50,
):
    """
    Creates temporal train / validation / test dataloaders with no cross-split leakage.
    """
    n_samples = len(x_t)
    train_end = int(train_split * n_samples)
    val_end = int((train_split + val_split) * n_samples)

    if train_end <= 0 or val_end <= train_end:
        raise ValueError("Invalid splits or too few samples.")

    train_dataset = NeuropixelsSequenceDataset(
        x_t[:train_end], u_t[:train_end], time_vector[:train_end], sequence_length
    )
    val_dataset = NeuropixelsSequenceDataset(
        x_t[train_end:val_end], u_t[train_end:val_end], time_vector[train_end:val_end], sequence_length
    )
    test_dataset = NeuropixelsSequenceDataset(
        x_t[val_end:], u_t[val_end:], time_vector[val_end:], sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, test_dataset
