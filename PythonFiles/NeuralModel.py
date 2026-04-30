
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from .constants import Constants as C
import torchmetrics
from torch import nn

def evaluate_tm(model, data_loader, metric, device):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()

def train_model(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, patience=2, factor=0.5, device=None, epoch_callback=None):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=patience, factor=factor)
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.0
        #metric.reset()
        model.train()
        if epoch_callback is not None:
            epoch_callback(model, epoch)
        for index, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
            train_metric = metric.compute().item()
            print(f"\rBatch {index + 1}/{len(train_loader)}", end="")
            print(f", loss={total_loss/(index+1):.4f}", end="")
            print(f", {train_metric=:.2%}", end="")
        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(train_metric)
        val_metric = evaluate_tm(model, valid_loader, metric).item()
        history["valid_metrics"].append(val_metric)
        scheduler.step(val_metric)
        print(f"\rEpoch {epoch + 1}/{n_epochs},                      "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.2%}, "
              f"valid metric: {history['valid_metrics'][-1]:.2%}")
    return history

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
        self.freq_out = n_mels // 4                    # 16
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