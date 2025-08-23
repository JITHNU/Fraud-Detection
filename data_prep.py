# src/data_prep.py
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
CSV = DATA_DIR / "transactions.csv"

# --- Columns we expect ---
LABEL = "isFraud"
SRC = "nameOrig"
DST = "nameDest"
TIME = "step"
CATEGORICAL = ["type"]
NUMERIC_EDGE = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

def build_id_mapping(df: pd.DataFrame):
    accs = pd.unique(pd.concat([df[SRC], df[DST]], axis=0))
    return {acc: i for i, acc in enumerate(accs)}

def preprocess_save():
    # Load CSV
    if not CSV.exists():
        raise FileNotFoundError(f"CSV not found at {CSV}. Put your file at data/transactions.csv")

    df = pd.read_csv(CSV)

    # Validate columns
    required = [TIME, "type", "amount", SRC, "oldbalanceOrg", "newbalanceOrig",
                DST, "oldbalanceDest", "newbalanceDest", LABEL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    # Basic cleaning
    df = df.dropna(subset=[SRC, DST, LABEL, "amount", "type", TIME])

    # Map accounts → integer IDs
    acc2id = build_id_mapping(df)
    with open(ARTIFACTS / "id_mapping.pkl", "wb") as f:
        pickle.dump(acc2id, f)

    # One-Hot encode transaction type (handle sklearn versions)
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # older scikit-learn
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_type = ohe.fit_transform(df[["type"]])

    # Scale numeric edge features
    num_cols = [c for c in NUMERIC_EDGE if c in df.columns]
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])

    # Scale time
    time_scaler = StandardScaler()
    X_time = time_scaler.fit_transform(df[[TIME]].astype(float))

    # Final edge features: [type OHE | numeric | time]
    X_edge = np.hstack([X_type, X_num, X_time]).astype("float32")

    # Labels and endpoints
    y = df[LABEL].astype(int).values
    src = df[SRC].map(acc2id).astype(int).values
    dst = df[DST].map(acc2id).astype(int).values

    # Time-based split (50/25/25)
    t = df[TIME].values
    t50, t75 = np.percentile(t, [50, 75])
    train_idx = np.where(t <= t50)[0]
    val_idx   = np.where((t > t50) & (t <= t75))[0]
    test_idx  = np.where(t > t75)[0]

    meta = {
        "ohe_categories": [list(c) for c in ohe.categories_],
        "num_cols": num_cols,
        "edge_feat_dim": int(X_edge.shape[1]),
        "n_accounts": int(len(acc2id)),
        "splits": {
            "train": [int(i) for i in train_idx],
            "val":   [int(i) for i in val_idx],
            "test":  [int(i) for i in test_idx],
        },
    }

    # Save preprocess artifacts
    with open(ARTIFACTS / "preprocess.pkl", "wb") as f:
        pickle.dump(
            {"ohe": ohe, "scaler": scaler, "time_scaler": time_scaler, "meta": meta},
            f
        )

    # Save arrays for training
    np.save(ARTIFACTS / "X_edge.npy", X_edge)
    np.save(ARTIFACTS / "y.npy", y)
    np.save(ARTIFACTS / "src.npy", src)
    np.save(ARTIFACTS / "dst.npy", dst)

    # Also a JSON copy of meta
    with open(ARTIFACTS / "train_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Preprocessing complete. ✅")
    print(f"Edges: {len(y)} | Nodes: {len(acc2id)} | Edge feat dim: {X_edge.shape[1]}")

if __name__ == "__main__":
    preprocess_save()
