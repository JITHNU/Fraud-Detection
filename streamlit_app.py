import sys
import os
import io
import re
import requests
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

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# URLs
PREPROCESS_URL = "https://drive.google.com/uc?id=1Uds7ZTU_8NBCHzE2bMGUKovBLIxX0KRg"
MODEL_PATH = "model.pt"
PREPROCESS_PATH = "artifacts/preprocess.pkl"

# Streamlit page config
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")
st.config.set_option('server.maxUploadSize', 1024)  # 1GB

# Theme selection
theme = st.sidebar.radio("Choose Theme", ["Light 🌞", "Dark 🌙"])
if theme == "Dark 🌙":
    st.markdown("""
        <style>
            body, .stApp { background-color: black !important; color: white !important; }
            .stTextInput, .stButton > button { background-color: #222; color: white !important; }
            .stSelectbox label, .stSelectbox div[data-baseweb="select"] { background-color: #222 !important; color: white !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body, .stApp { background-color: white !important; color: black !important; }
            .stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stButton > button { background-color: #f9f9f9 !important; color: black !important; }
        </style>
    """, unsafe_allow_html=True)

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

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Action", ["Manual Input", "Bulk CSV"])

# ------------------------------
# Manual Input
# ------------------------------
if app_mode == "Manual Input":
    st.header("Manual Input Prediction ✍🏻")
    st.markdown("Enter transaction details below:")

    with st.form(key="manual_form"):
        payment_type = st.selectbox("Payment Type", ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"], key="manual_payment_type")
        step = st.number_input("Step", min_value=0, step=1, key="manual_step")
        amount = st.number_input("Amount", min_value=0.0, step=0.01, key="manual_amount")
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, step=0.01, key="manual_oldbalanceOrg")
        newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, step=0.01, key="manual_newbalanceOrig")
        oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, step=0.01, key="manual_oldbalanceDest")
        newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, step=0.01, key="manual_newbalanceDest")
        manual_threshold = st.slider("Fraud Probability Threshold", 0.0,1.0,0.5, key="manual_threshold_slider")
        submit_manual = st.form_submit_button("Predict")

    if submit_manual:
        df = pd.DataFrame([{
            "step": step, "amount": amount, "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest, "type": payment_type
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
        feature_cols = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest',
                        'type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER']
        X = df[feature_cols].values.astype(np.float32)
        X_tensor = torch.tensor(X)

        with torch.no_grad():
            logits = model(X_tensor)
            prob = torch.sigmoid(logits).item()

        prediction = "Fraud" if prob > manual_threshold else "Not Fraud"
        st.success(f"Predicted Fraud Probability: {prob:.4f}")
        st.info(f"Prediction: {prediction}")

# ------------------------------
# Bulk CSV via URL
# ------------------------------
elif app_mode == "Bulk CSV":
    st.header("Bulk CSV Prediction 📂🧑🏻‍💻 via URL")

    with st.form(key="bulk_form"):
        file_url = st.text_input("Enter CSV URL (Google Drive / Dropbox / S3) including 'isFraud' if available", key="bulk_file_url")
        bulk_threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, key="bulk_threshold_slider")
        submit_bulk = st.form_submit_button("Process CSV")

    if submit_bulk and file_url:
        try:
            st.info("Downloading CSV...")

            # Convert Google Drive share link to direct download
            drive_match = re.search(r'drive.google.com.*?/d/([a-zA-Z0-9_-]+)', file_url)
            if drive_match:
                file_id = drive_match.group(1)
                file_url = f"https://drive.google.com/uc?id={file_id}&export=download"

            response = requests.get(file_url)
            response.raise_for_status()
            csv_file = io.StringIO(response.content.decode("utf-8"))

            chunksize = 50000
            results = []

            # Count total chunks
            total_chunks = sum(1 for _ in pd.read_csv(csv_file, chunksize=chunksize))
            csv_file.seek(0)
            progress_bar = st.progress(0)
            current_chunk = 0

            for chunk in pd.read_csv(csv_file, chunksize=chunksize):
                current_chunk += 1
                progress_bar.progress(current_chunk / total_chunks)

                # Encode type
                if 'type' in chunk.columns:
                    df_type = pd.get_dummies(chunk['type'], prefix='type')
                    expected_type_cols = ['type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER']
                    for col in expected_type_cols:
                        if col not in df_type.columns:
                            df_type[col] = 0
                    chunk = pd.concat([chunk.drop('type', axis=1), df_type[expected_type_cols]], axis=1)

                # Scale numeric
                numeric_cols = preprocessor['scaler'].feature_names_in_
                chunk[numeric_cols] = preprocessor['scaler'].transform(chunk[numeric_cols])

                # Feature order
                feature_cols = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest',
                                'type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER']
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
            st.subheader("Top 20 Predictions")
            st.dataframe(df[["fraud_probability","predicted_fraud"]].head(20))

        except Exception as e:
            st.error(f"Error processing file: {e}")
