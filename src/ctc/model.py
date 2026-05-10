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
        n_classes: int = C.N_CLASSES,
        conv_channels: tuple[int, int] = (32, 64),
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        bidirectional: bool = True,
        time_reduction_factor: int = C.TIME_REDUCTION_FACTOR,
    ) -> None:
        super().__init__()

        conv_in_channels, conv_out_channels = conv_channels

        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_in_channels, conv_in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (F, T) -> (F/2, T/2)
            nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (F/2, T/2) -> (F/4, T/4)
            nn.Dropout(dropout_rate),
        )

        self.freq_out = n_mels // time_reduction_factor
        rnn_directions = 2 if bidirectional else 1

        self.rnn = nn.LSTM(
            input_size=conv_out_channels * self.freq_out,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate,
        )

        self.fc = nn.Linear(hidden_size * rnn_directions, n_classes)

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
