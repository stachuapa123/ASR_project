import torch
from torch import nn

from .config import CTCConfig as C


class CTCModel(nn.Module):
    """
    CNN + BiLSTM acoustic model for CTC.

    Input:
        x: (batch, n_mels, time) log-mel spectrograms

    Output:
        logits: (batch, time', n_classes+1) - time' is reduced by TIME_REDUCTION_FACTOR
    """

    def __init__(
        self,
        n_mels: int = C.N_MELS,
        n_classes: int = C.N_CLASSES + 1,  # +1 for CTC blank
        hidden_size: int = 128,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (F, T) -> (F/2, T/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (F/2, T/2) -> (F/4, T/4)
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
        """
        Args:
            x: (B, n_mels, T)

        Returns:
            logits: (B, T', n_classes)
        """
        x = x.unsqueeze(1)  # (B, 1, F, T)
        x = self.conv(x)

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F')

        x, _ = self.rnn(x)
        logits = self.fc(x)
        return logits
