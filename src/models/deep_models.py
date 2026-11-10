"""
QuantEdge Deep Learning Module for Financial Time Series
Implements PyTorch CPU-optimized neural architectures:
1. LSTM (Long Short-Term Memory)
2. GRU (Gated Recurrent Unit)
3. TCN (Temporal Convolutional Network with Causal Dilated Convolutions)
4. TimeSeriesTransformer (Multi-Head Self-Attention with Positional Encoding)
Enforces sliding window generation, strict temporal splits, early stopping, and gradient clipping.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.baseline import FinancialBaselines


class SlidingWindowDataset(Dataset):
    """
    Constructs causal sliding windows [X_{t-seq_len+1}, ..., X_t] -> y_{t+1}.
    Strictly preserves causality with no lookahead.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 15):
        self.seq_len = seq_len
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_window = self.X[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]
        return x_window, y_target


# --- 1. LSTM Model ---
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


# --- 2. GRU Model ---
class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


# --- 3. Temporal Convolutional Network (TCN) ---
class ChausalConv1d(nn.Module):
    """Causal convolution with dilation to prevent future leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        out = self.conv(x)
        return out[:, :, : -self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        return torch.relu(out + res)


class TCNForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.block1 = TCNBlock(input_dim, hidden_dim, kernel_size=kernel_size, dilation=1, dropout=dropout)
        self.block2 = TCNBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=2, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len, input_dim) -> transpose to (batch, input_dim, seq_len)
        x_trans = x.transpose(1, 2)
        out = self.block1(x_trans)
        out = self.block2(out)
        # Take the final temporal state
        last_state = out[:, :, -1]
        return self.fc(last_state).squeeze(-1)


# --- 4. Lightweight Time-Series Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        x_pos = self.pos_encoder(x_proj)
        out = self.transformer(x_pos)
        last_token = out[:, -1, :]
        return self.fc(last_token).squeeze(-1)


# --- Training and Evaluation Pipeline ---
@dataclass
class DeepLearningResult:
    architecture: str
    train_losses: List[float]
    val_losses: List[float]
    test_metrics: Dict[str, float]
    predictions_test: pd.Series
    actuals_test: pd.Series
    total_epochs: int


class DeepLearningLab:
    """
    Manages lightweight CPU training for PyTorch time-series architectures.
    """

    def __init__(self, seq_len: int = 15, batch_size: int = 32, epochs: int = 25, lr: float = 0.003, seed: int = 42):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train_model(
        self,
        architecture: str,  # 'LSTM', 'GRU', 'TCN', 'Transformer'
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        val_ratio: float = 0.20,
    ) -> DeepLearningResult:
        # Scale features using training set parameters
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        # Chronological split of training set into train and validation
        n_tr = len(X_tr_sc)
        split_idx = int(n_tr * (1.0 - val_ratio))
        X_t, y_t = X_tr_sc[:split_idx], y_train.values[:split_idx]
        X_v, y_v = X_tr_sc[split_idx:], y_train.values[split_idx:]

        # Create Sliding Window Datasets
        train_ds = SlidingWindowDataset(X_t, y_t, seq_len=self.seq_len)
        val_ds = SlidingWindowDataset(X_v, y_v, seq_len=self.seq_len)

        # Combine tail of training set with test set to have contiguous windows for test set
        X_test_extended = np.vstack([X_tr_sc[-self.seq_len :], X_te_sc])
        y_test_extended = np.concatenate([y_train.values[-self.seq_len :], y_test.values])
        test_ds = SlidingWindowDataset(X_test_extended, y_test_extended, seq_len=self.seq_len)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        input_dim = X_train.shape[1]

        # Model Selection
        if architecture == "LSTM":
            model = LSTMForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1)
        elif architecture == "GRU":
            model = GRUForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1)
        elif architecture == "TCN":
            model = TCNForecaster(input_dim=input_dim, hidden_dim=32, kernel_size=3)
        elif architecture == "Transformer":
            model = TimeSeriesTransformer(input_dim=input_dim, d_model=32, nhead=4, num_layers=1)
        else:
            raise ValueError(f"Unknown deep architecture: {architecture}")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")
        best_state = None
        patience = 8
        patience_counter = 0

        # Training Loop
        model.train()
        for epoch in range(self.epochs):
            batch_losses = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            tr_loss = float(np.mean(batch_losses))
            train_losses.append(tr_loss)

            # Validation Loop
            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    out = model(batch_x)
                    val_loss = criterion(out, batch_y)
                    val_batch_losses.append(val_loss.item())

            v_loss = float(np.mean(val_batch_losses)) if val_batch_losses else tr_loss
            val_losses.append(v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            model.train()

        # Load best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)

        # Out-of-sample Test Evaluation
        model.eval()
        preds_list = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                out = model(batch_x)
                preds_list.extend(out.numpy().tolist())

        preds_arr = np.array(preds_list)[: len(y_test)]
        actuals_arr = y_test.values[: len(preds_arr)]

        eval_res = FinancialBaselines.evaluate_forecast(actuals_arr, preds_arr, f"{architecture}_Test")
        metrics = {
            "RMSE": eval_res.rmse,
            "MAE": eval_res.mae,
            "Directional_Accuracy": eval_res.directional_accuracy,
            "IC": eval_res.information_coefficient,
            "R2": eval_res.r_squared,
        }

        preds_series = pd.Series(preds_arr, index=y_test.index[: len(preds_arr)], name=f"{architecture}_Pred")

        return DeepLearningResult(
            architecture=architecture,
            train_losses=train_losses,
            val_losses=val_losses,
            test_metrics=metrics,
            predictions_test=preds_series,
            actuals_test=y_test.iloc[: len(preds_arr)],
            total_epochs=len(train_losses),
        )
