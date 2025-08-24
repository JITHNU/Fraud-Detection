# Fraud Detection App

Live Demo 🚀🔗 - https://fraud-detection-k9th4vbrnxark9okngx2dx.streamlit.app/

A Streamlit web application for fraud detection in financial transactions using a pre-trained PyTorch model.

---

## 🚀 Features
- **Manual Prediction:** Enter transaction details and get fraud probability.
- **Bulk CSV Prediction:** Upload CSV files and process in chunks with progress bar.
- **Visualizations:** Histogram, Confusion Matrix, ROC Curve (downloadable as PNG).
- **Downloadable Results:** Save predictions as CSV.

---
🖥️ User Guide

1️⃣ App Modes

When you launch the app, you will see two options in the sidebar:

Single Transaction & Bulk CSV

✅ Single Transaction Mode

In this mode, you manually enter transaction details in the form.
The app will predict if the transaction is Fraud or Not Fraud.

Required Inputs:

step – Transaction time step (e.g., 100)

amount – Amount of transaction (e.g., 5000.0)

oldbalanceOrg – Balance before transaction for sender (e.g., 10000.0)

newbalanceOrig – Balance after transaction for sender (e.g., 5000.0)

oldbalanceDest – Balance before transaction for receiver (e.g., 2000.0)

newbalanceDest – Balance after transaction for receiver (e.g., 7000.0)

type – Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)

🖲️ After submitting, you’ll see:

Fraud Probability (0 to 1)

Prediction (Fraud / Not Fraud)

✅ Bulk CSV Mode

In this mode, you upload a CSV file containing multiple transactions. The app will process them in chunks for efficiency.

Expected CSV Format:

Your CSV must include the following columns:

Column Name	Description
step	Time step of the transaction
type	Type of transaction (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
amount	Amount of the transaction
oldbalanceOrg	Balance of the sender before the transaction
newbalanceOrig	Balance of the sender after the transaction
oldbalanceDest	Balance of the receiver before the transaction
newbalanceDest	Balance of the receiver after the transaction
isFraud (optional)	1 if fraud, 0 if not fraud (only required if you want performance metrics)

⚠️ Important:

If your file includes isFraud, the app will generate Confusion Matrix, ROC Curve, and Classification Report.

If isFraud is missing, the app will only predict fraud probabilities.

🖲️ Output in Bulk CSV Mode

Predictions Table: Shows fraud probability and classification.

High Probability Fraud Table: Filters only likely fraud transactions.

Visualizations:

Fraud Probability Distribution (histogram)

Confusion Matrix (if isFraud exists)

ROC Curve with AUC (if isFraud exists)

Download Options:

Fraud predictions as CSV


---

## 🖥️ Running the App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
