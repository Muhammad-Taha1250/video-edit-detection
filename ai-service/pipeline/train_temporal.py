"""
train_temporal.py – Day 3 Dilated 1D-CNN + BiLSTM Sequence Tagging Model (PyTorch).

Architecture:
  - 3 Dilated 1D-Conv residual blocks (dilations 1, 2, 4)
  - 1 BiLSTM layer (captures temporal bidirectional context)
  - 1x1 Conv head predicting 5 classes per step
  - Window size T=96, Stride=16
  - Class-weighted CrossEntropyLoss
  - CPU only; reports parameter count and training time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve().parent
AI_SERVICE = _HERE.parent
sys.path.insert(0, str(AI_SERVICE))

from pipeline.features_v2 import FEATURE_NAMES_V2

MODELS_DIR = AI_SERVICE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEMPORAL_PATH = MODELS_DIR / "temporal_v2.pt"
TEMPORAL_META_PATH = MODELS_DIR / "temporal_v2_meta.json"

WINDOW_T = 96
STRIDE_T = 16


class TemporalCNN(nn.Module):
    def __init__(self, in_channels: int = 27, num_classes: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        h = self.drop(self.relu(self.bn1(self.conv1(x))))
        h = self.drop(self.relu(self.bn2(self.conv2(h)))) + h
        h = self.drop(self.relu(self.bn3(self.conv3(h)))) + h
        # (B, hidden, T) -> (B, T, hidden)
        h = h.transpose(1, 2)
        lstm_out, _ = self.bilstm(h)
        # (B, T, hidden) -> (B, hidden, T)
        out = self.head(lstm_out.transpose(1, 2))  # (B, num_classes, T)
        return out


class WindowDataset(Dataset):
    def __init__(self, windows: list[np.ndarray], targets: list[np.ndarray]):
        self.windows = torch.tensor(np.array(windows), dtype=torch.float32)  # (N, D, T)
        self.targets = torch.tensor(np.array(targets), dtype=torch.long)     # (N, T)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets[idx]


def _slice_windows(
    table: pd.DataFrame,
    le: LabelEncoder,
    T: int = WINDOW_T,
    stride: int = STRIDE_T,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    X_wins, y_wins = [], []
    for _, grp in table.groupby("video", sort=False):
        feats = grp[FEATURE_NAMES_V2].values.T.astype(np.float32)  # (D, N)
        labels = le.transform(grp["label"].values)                  # (N,)
        n = feats.shape[1]

        if n < T:
            # Pad to T
            pad_w = T - n
            feats_pad = np.pad(feats, ((0, 0), (0, pad_w)), mode="edge")
            labels_pad = np.pad(labels, (0, pad_w), mode="edge")
            X_wins.append(feats_pad)
            y_wins.append(labels_pad)
        else:
            for start in range(0, n - T + 1, stride):
                end = start + T
                X_wins.append(feats[:, start:end])
                y_wins.append(labels[start:end])
            # Ensure last window is covered
            if (n - T) % stride != 0:
                X_wins.append(feats[:, n - T : n])
                y_wins.append(labels[n - T : n])

    return X_wins, y_wins


def train_temporal(force: bool = False) -> tuple[nn.Module, LabelEncoder, dict]:
    table_csv = MODELS_DIR / "feature_table_v2.csv"
    table_parquet = MODELS_DIR / "feature_table_v2.parquet"
    if table_parquet.exists():
        table = pd.read_parquet(table_parquet)
    else:
        table = pd.read_csv(table_csv)

    y_raw = table["label"].values
    groups = table["source_file"].values
    le = LabelEncoder()
    le.fit(y_raw)

    # Identical GroupShuffleSplit 80/20 by source_file
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(table, groups=groups))

    df_train = table.iloc[train_idx]
    df_test = table.iloc[test_idx]

    # Further split 1 source from df_train for early stopping validation
    train_groups = df_train["source_file"].values
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    sub_train_idx, val_idx = next(gss_val.split(df_train, groups=train_groups))

    df_subtrain = df_train.iloc[sub_train_idx]
    df_val = df_train.iloc[val_idx]

    print(f"\nTemporal Model Data Split:")
    print(f"  Sub-Train : {len(df_subtrain):,} rows ({df_subtrain['source_file'].nunique()} sources)")
    print(f"  Validation: {len(df_val):,} rows ({df_val['source_file'].nunique()} sources)")
    print(f"  Held-Out  : {len(df_test):,} rows ({df_test['source_file'].nunique()} sources)")

    # Compute class weights for loss
    class_counts = df_subtrain["label"].value_counts()
    total_samples = len(df_subtrain)
    weights = []
    for cls in le.classes_:
        cnt = class_counts.get(cls, 1)
        w = total_samples / (len(le.classes_) * cnt)
        weights.append(min(float(w), 25.0))  # clip max weight to 25.0
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    print("Class weights for CrossEntropyLoss:", {cls: round(weights[i], 2) for i, cls in enumerate(le.classes_)})

    X_train_wins, y_train_wins = _slice_windows(df_subtrain, le, WINDOW_T, STRIDE_T)
    X_val_wins, y_val_wins = _slice_windows(df_val, le, WINDOW_T, STRIDE_T)

    train_loader = DataLoader(WindowDataset(X_train_wins, y_train_wins), batch_size=32, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val_wins, y_val_wins), batch_size=32, shuffle=False)

    model = TemporalCNN(in_channels=len(FEATURE_NAMES_V2), num_classes=len(le.classes_))
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Temporal Model Parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Train loop
    epochs = 15
    best_val_loss = float("inf")
    best_weights = None
    t0_train = time.time()

    print(f"\nTraining Temporal Model on CPU ({epochs} epochs max) …")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)  # (B, 5, T)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    train_time = time.time() - t0_train
    print(f"✓ Training complete in {train_time:.2f}s | Best Val Loss: {best_val_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    torch.save(model.state_dict(), TEMPORAL_PATH)
    meta = {
        "param_count": param_count,
        "training_time_sec": round(train_time, 2),
        "best_val_loss": round(best_val_loss, 4),
        "classes": le.classes_.tolist(),
        "in_channels": len(FEATURE_NAMES_V2),
        "window_t": WINDOW_T,
        "stride_t": STRIDE_T,
    }
    TEMPORAL_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"✓ Temporal model saved → {TEMPORAL_PATH}")

    return model, le, meta


if __name__ == "__main__":
    train_temporal(force="--force" in sys.argv)

