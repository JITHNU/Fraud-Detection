# src/evaluate.py
import numpy as np
import torch
from src.model import GraphSAGE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path="artifacts/model.pt"):
    # Load model
    in_channels = 11
    model = GraphSAGE(in_channels=in_channels, hidden_channels=64)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Model loaded from ✅", model_path)

    # Load preprocessed arrays
    X_edge = np.load("artifacts/X_edge.npy")
    y = np.load("artifacts/y.npy")

    # Convert to torch tensors
    X_tensor = torch.tensor(X_edge, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)

    # Forward pass
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.sigmoid(logits).numpy()

    # Metrics
    auc_score = roc_auc_score(y_tensor.numpy(), preds)
    print(f"Validation/Test AUC: {auc_score:.4f}")
    print(classification_report(y_tensor.numpy(), preds > 0.5))

    # Confusion Matrix
    cm = confusion_matrix(y_tensor.numpy(), preds > 0.5)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_tensor.numpy(), preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    evaluate_model()

