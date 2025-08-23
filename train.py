import pickle
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

with open(ARTIFACTS / "preprocess.pkl", "rb") as f:
    prep = pickle.load(f)
meta = prep["meta"]

X_edge = np.load(ARTIFACTS / "X_edge.npy")
y = np.load(ARTIFACTS / "y.npy")
src = np.load(ARTIFACTS / "src.npy")
dst = np.load(ARTIFACTS / "dst.npy")

splits = meta["splits"]

class GNNModel(nn.Module):
    def __init__(self, edge_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(edge_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)

# --- Training ---
def train():
    X = torch.tensor(X_edge, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)

    model = GNNModel(edge_dim=meta["edge_feat_dim"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(1, 6):  # 5 epochs demo
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_idx = splits["val"]
            val_pred = model(X[val_idx]).numpy()
            val_y = Y[val_idx].numpy()
            auc = roc_auc_score(val_y, val_pred)

        print(f"Epoch {epoch}: loss={loss.item():.4f} | val AUC={auc:.4f}")

    # Save trained model
    torch.save(model.state_dict(), ARTIFACTS / "model.pt")
    print("Model training complete.✅  Saved to artifacts/model.pt")

if __name__ == "__main__":
    train()
