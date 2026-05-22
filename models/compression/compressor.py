import torch.nn as nn


class AudioCompressor(nn.Module):
    def __init__(self, target_len=50):
        super().__init__()
        self.target_len = target_len

    def forward(self, x):
        B, T, D = x.shape
        trim = (T // self.target_len) * self.target_len
        x = x[:, :trim, :]
        group_size = trim // self.target_len
        x = x.reshape(B, self.target_len, group_size, D)
        return x.mean(dim=2)
