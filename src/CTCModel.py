import torch
from torch import nn
from .constants import Constants as C


class CTC_CRNN(nn.Module):
    def __init__(
        self,
        n_mels: int = C.N_MELS,
        n_classes: int = C.N_CLASSES + 1,
        hidden_size: int = 128,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(dropout_rate),
        )

        self.freq_out = n_mels // 4
        self.rnn = nn.LSTM(
            input_size=64 * self.freq_out,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)

        b, c_dim, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c_dim * f)

        x, _ = self.rnn(x)
        logits = self.fc(x)

        return logits
