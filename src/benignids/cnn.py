from __future__ import annotations

import numpy as np


def train_cnn(X_train, y_train, X_validation, y_validation, epochs=12, batch_size=256, seed=42):
    """Train a compact 1D-CNN benchmark. PyTorch is intentionally optional."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Install the optional dependency with: pip install -e '.[deep]'") from exc

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = torch.as_tensor(np.asarray(X_train, dtype=np.float32)).unsqueeze(1)
    train_y = torch.as_tensor(np.asarray(y_train, dtype=np.float32)).unsqueeze(1)
    validation_x = torch.as_tensor(np.asarray(X_validation, dtype=np.float32)).unsqueeze(1)

    class Network(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
                nn.Linear(32 * 8, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )

        def forward(self, values):
            return self.layers(values)

    model = Network(train_x.shape[-1])
    positives = max(float(train_y.sum()), 1.0)
    negatives = max(float(len(train_y) - train_y.sum()), 1.0)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(validation_x)).squeeze(1).numpy()
    return model, scores

