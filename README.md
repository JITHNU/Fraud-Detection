# Fraud Detection App

A Streamlit web application for fraud detection in financial transactions using a pre-trained PyTorch model.

---

## 🚀 Features
- **Manual Prediction:** Enter transaction details and get fraud probability.
- **Bulk CSV Prediction:** Upload CSV files and process in chunks with progress bar.
- **Visualizations:** Histogram, Confusion Matrix, ROC Curve (downloadable as PNG).
- **Downloadable Results:** Save predictions as CSV.

---

## 📂 CSV Format
The uploaded CSV should contain at least these columns:
- `step`
- `type` (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

👉 If the CSV also has the column `isFraud`, the app will generate:
- Confusion Matrix
- ROC Curve
- Classification Report

---

## 🖥️ Running the App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
