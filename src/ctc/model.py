import torch
from torch import nn

from .config import CTCConfig as C


class CTCModel(nn.Module):
    """
    CNN + BiLSTM acoustic model for CTC.

    Input:  x      (batch, n_mels, time) log-mel.
    Output: logits (batch, time', n_classes+1); time' = time / TIME_REDUCTION_FACTOR.

    GroupNorm (batch-size independent), spatial Dropout2d on the CNN, and per-type
    dropout (cnn/rnn/head, each defaulting to ``dropout_rate``).
    """

    def __init__(
        self,
        n_mels: int = C.N_MELS,
        n_classes: int = C.N_CLASSES,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        cnn_dropout: float | None = None,
        rnn_dropout: float | None = None,
        head_dropout: float | None = None,
        bidirectional: bool = True,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        # each falls back to dropout_rate when unset (keeps CTCModel() unchanged)
        cnn_dropout = dropout_rate if cnn_dropout is None else cnn_dropout
        rnn_dropout = dropout_rate if rnn_dropout is None else rnn_dropout
        head_dropout = dropout_rate if head_dropout is None else head_dropout

        # one block per channel count; each halves freq, only block 0 strides time (2x total)
        layers: list[nn.Module] = []
        in_channels = 1
        for block_idx, out_channels in enumerate(cnn_channels):
            time_stride = 2 if block_idx == 0 else 1
            groups = self._groups(norm_groups, out_channels)
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, time_stride)),  # F/2 (and T/2 first block)
            ]
            in_channels = out_channels
        # spatial dropout: zero whole channels (CNN maps are spatially correlated)
        layers.append(nn.Dropout2d(cnn_dropout))
        self.cnn = nn.Sequential(*layers)

        self.freq_out = n_mels // (2 ** len(cnn_channels))
        rnn_directions = 2 if bidirectional else 1

        self.rnn = nn.LSTM(
            input_size=cnn_channels[-1] * self.freq_out,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # LSTM dropout applies only between layers; 0 when num_layers==1 (else warning)
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )
        # dropout before the projection (LSTM's own dropout skips the last layer)
        self.head_dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(hidden_size * rnn_directions, n_classes)

    @staticmethod
    def _groups(requested: int, channels: int) -> int:
        """Largest divisor of ``channels`` that is <= ``requested`` (GroupNorm needs
        ``channels % groups == 0``)."""
        g = min(requested, channels)
        while channels % g != 0:
            g -= 1
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, F, T)
        x = self.cnn(x)

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F')

        x, _ = self.rnn(x)
        x = self.head_dropout(x)
        logits = self.fc(x)
        return logits
