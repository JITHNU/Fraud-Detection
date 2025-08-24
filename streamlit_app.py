import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gdown
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from model import GraphSAGE
import os
import io
import requests

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.makedirs("artifacts", exist_ok=True)

# URLs
PREPROCESS_URL = "https://drive.google.com/uc?id=1Uds7ZTU_8NBCHzE2bMGUKovBLIxX0KRg"
MODEL_PATH = "model.pt"
PREPROCESS_PATH = "artifacts/preprocess.pkl"

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide"
)
st.config.set_option('server.maxUploadSize', 1024)  # 1GB

# Theme
theme = st.sidebar.radio("Choose Theme", ["Light 🌞", "Dark 🌙"])
if theme == "Dark 🌙":
    st.markdown(
        """
        <style>
            body, .stApp { background-color: black !important; color: white !important; }
            .stTextInput, .stButton > button { background-color: #222; color: white !important; }
            .stSelectbox label, .stSelectbox div[data-baseweb="select"] {
                background-color: #222 !important;
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body, .stApp { background-color: white !important; color: black !important; }
        .stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stButton > button {
            background-color: #f9f9f9 !important;
            color: black !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found! Upload to GitHub or external storage.")
    model = GraphSAGE(in_channels=11, hidden_channels=64)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# Load preprocessor
@st.cache_resource
def load_preprocessor():
    if not os.path.exists(PREPROCESS_PATH):
        st.info("Downloading preprocessor...")
        gdown.download(PREPROCESS_URL, PREPROCESS_PATH, quiet=False)
    with open(PREPROCESS_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()
preprocessor = load_preprocessor()

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Action", ["Manual Input", "Bulk CSV via URL"])

# ----------------- Manual Input -----------------
if app_mode == "Manual Input":
    st.header(" Manual Input Prediction ✍🏻 ")
    st.markdown("Enter transaction details below:")

    with st.form(key="manual_form"):
        payment_type = st.selectbox(
            "Payment Type",
            ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            key="manual_payment_type"
        )
        step = st.number_input("Step", min_value=0, step=1, key="manual_step")
        amount = st.number_input("Amount", min_value=0.0, step=0.01, key="manual_amount")
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, step=0.01, key="manual_oldbalanceOrg")
        newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, step=0.01, key="manual_newbalanceOrig")
        oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, step=0.01, key="manual_oldbalanceDest")
        newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, step=0.01, key="manual_newbalanceDest")
        manual_threshold = st.slider(
            "Fraud Probability Threshold",
            0.0, 1.0, 0.5,
            key="manual_threshold_slider"
        )
        submit_manual = st.form_submit_button("Predict")

    if submit_manual:
        df = pd.DataFrame([{
            "step": step,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "type": payment_type
        }])

        # Encode type
        df_type = pd.get_dummies(df['type'], prefix='type')
        expected_type_cols = ['type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER']
        for col in expected_type_cols:
            if col not in df_type.columns:
                df_type[col] = 0
        df = pd.concat([df.drop('type', axis=1), df_type[expected_type_cols]], axis=1)

        # Scale numeric
        numeric_cols = preprocessor['scaler'].feature_names_in_
        df[numeric_cols] = preprocessor['scaler'].transform(df[numeric_cols])

        # Feature order
        feature_cols = [
            'step','amount','oldbalanceOrg','newbalanceOrig',
            'oldbalanceDest','newbalanceDest',
            'type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER'
        ]
        X = df[feature_cols].values.astype(np.float32)
        X_tensor = torch.tensor(X)

        with torch.no_grad():
            logits = model(X_tensor)
            prob = torch.sigmoid(logits).item()

        prediction = "Fraud" if prob > manual_threshold else "Not Fraud"
        st.success(f"Predicted Fraud Probability: {prob:.4f}")
        st.info(f"Prediction: {prediction}")

# Bulk CSV
elif app_mode == "Bulk CSV":
    st.header("Bulk CSV Prediction 📂🧑🏻‍💻")
    uploaded_file = st.file_uploader(
        "Upload CSV file (include 'isFraud' if available)",
        type="csv",
        key="bulk_file"
    )

    if uploaded_file is not None:
        try:
            from io import BytesIO

            # Save uploaded file to memory buffer (resettable)
            file_buffer = BytesIO(uploaded_file.read())

            chunksize = 50000
            results = []

            # Count chunks (safe way)
            num_chunks = sum(1 for _ in pd.read_csv(BytesIO(file_buffer.getvalue()), chunksize=chunksize))
            file_buffer.seek(0)

            st.subheader("Processing CSV...")
            progress_bar = st.progress(0)
            current_chunk = 0

            bulk_threshold = st.slider(
                "Fraud Probability Threshold",
                0.0, 1.0, 0.5,
                key="bulk_threshold_slider"
            )

            for chunk in pd.read_csv(file_buffer, chunksize=chunksize):
                current_chunk += 1
                progress_bar.progress(current_chunk / num_chunks)

                # One-hot encode 'type'
                if 'type' in chunk.columns:
                    df_type = pd.get_dummies(chunk['type'], prefix='type')
                    expected_type_cols = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
                    for col in expected_type_cols:
                        if col not in df_type.columns:
                            df_type[col] = 0
                    chunk = pd.concat([chunk.drop('type', axis=1), df_type[expected_type_cols]], axis=1)

                # Scale numeric safely
                numeric_cols = list(preprocessor['scaler'].feature_names_in_)
                for col in numeric_cols:
                    if col not in chunk.columns:
                        chunk[col] = 0.0
                chunk[numeric_cols] = preprocessor['scaler'].transform(chunk[numeric_cols])

                # Final feature order
                feature_cols = [
                    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest',
                    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
                ]
                for col in feature_cols:
                    if col not in chunk.columns:
                        chunk[col] = 0
                X = chunk[feature_cols].values.astype(np.float32)
                X_tensor = torch.tensor(X)

                with torch.no_grad():
                    logits = model(X_tensor)
                    probs = torch.sigmoid(logits).numpy()

                chunk["fraud_probability"] = probs
                chunk["predicted_fraud"] = np.where(chunk["fraud_probability"] > bulk_threshold, "Fraud", "Not Fraud")
                results.append(chunk)

            df = pd.concat(results, ignore_index=True)
            st.success("Bulk predictions completed! ✅")

            st.subheader("Predictions (Top 20 rows)")
            st.dataframe(df[["fraud_probability", "predicted_fraud"]].head(20))

            st.subheader("High Probability Frauds ⚠️")
            st.dataframe(df[df["predicted_fraud"] == "Fraud"])

            # Histogram
            st.subheader("Fraud Probability Distribution 📊")
            fig, ax = plt.subplots()
            sns.histplot(df["fraud_probability"], bins=50, kde=True, ax=ax)
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Evaluation if labels exist
            if 'isFraud' in df.columns:
                y_true = df['isFraud']
                y_pred = np.where(df['fraud_probability'] > bulk_threshold, 1, 0)

                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # ROC
                fpr, tpr, _ = roc_curve(y_true, df['fraud_probability'])
                roc_auc = auc(fpr, tpr)
                st.subheader(f"ROC Curve (AUC = {roc_auc:.4f})")
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")
                st.pyplot(fig)

                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.json(report)

            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
