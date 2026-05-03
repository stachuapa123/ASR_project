import torch
from torch import nn
from .constants import Constants as C


class CRNN(nn.Module):
    def __init__(self, n_mels=C.N_MELS, n_classes=C.N_CLASSES, hidden=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                     # halve freq only
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)), 
            nn.Dropout(dropout),                    
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),                    
                                # halve freq again
        )
        self.freq_out = n_mels // 4                    # 32 for 128 mels
        self.rnn = nn.LSTM(input_size=64 * self.freq_out,
                          hidden_size=hidden,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout)
        self.fc  = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        # x: (B, n_mels, T)
        x = x.unsqueeze(1)                             # (B, 1, n_mels, T)
        x = self.conv(x)                               # (B, 32, n_mels/4, T)
        B, Cc, F, Tt = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, Tt, Cc * F)  # (B, T, C*F)
        _, (h, _) = self.rnn(x)              # discard cell state
        h = torch.cat([h[0], h[1]], dim=1)   # forward + backward hidden
        return self.fc(h)                              # (B, n_classes)
