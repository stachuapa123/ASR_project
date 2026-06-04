import torch
from torch import nn

from .config import CTCConfig as C


class CTCModel(nn.Module):
    """
    CNN + BiLSTM acoustic model for CTC.

    Input:
        x: (batch, n_mels, time) log-mel spectrograms

    Output:
        logits: (batch, time', n_classes+1) - time' is reduced by
        C.TIME_REDUCTION_FACTOR (2x; only the first conv block strides time).
    """

    def __init__(
        self,
        n_mels: int = C.N_MELS,
        n_classes: int = C.N_CLASSES,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        # One conv block per entry in conv_channels. Each block halves the frequency axis
        # Only the first block also halves time
        # Total time reduction is fixed at 2x (matching C.TIME_REDUCTION_FACTOR) while frequency is reduced by 2 ** len(conv_channels)
        layers: list[nn.Module] = []
        in_channels = 1
        for block_idx, out_channels in enumerate(conv_channels):
            time_stride = 2 if block_idx == 0 else 1
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, time_stride)),  # F/2 (and T/2 first block)
            ]
            in_channels = out_channels
        layers.append(nn.Dropout(dropout_rate))
        self.conv = nn.Sequential(*layers)

        self.freq_out = n_mels // (2 ** len(conv_channels))
        rnn_directions = 2 if bidirectional else 1

        self.rnn = nn.LSTM(
            input_size=conv_channels[-1] * self.freq_out,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate,
        )

        self.fc = nn.Linear(hidden_size * rnn_directions, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, F, T)
        x = self.conv(x)

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F')

        x, _ = self.rnn(x)
        logits = self.fc(x)
        return logits
