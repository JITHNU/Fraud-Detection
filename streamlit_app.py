import sys
import os
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

# Setup paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.makedirs("artifacts", exist_ok=True)

PREPROCESS_URL = "https://drive.google.com/uc?id=1Uds7ZTU_8NBCHzE2bMGUKovBLIxX0KRg"
MODEL_PATH = "artifacts/model.pt"
PREPROCESS_PATH = "artifacts/preprocess.pkl"

# Streamlit page config
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")
st.config.set_option('server.maxUploadSize', 1024)  # 1GB

# Theme
theme = st.sidebar.radio("Choose Theme", ["Light 🌞", "Dark 🌙"])
if theme == "Dark 🌙":
    st.markdown("""
        <style>
            body, .stApp { background-color: black !important; color: white !important; }
            .stTextInput, .stNumberInput, .stButton > button { background-color: #222; color: white !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body, .stApp { background-color: white !important; color: black !important; }
            .stTextInput, .stNumberInput, .stButton > button { background-color: #f9f9f9 !important; color: black !important; }
        </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found! Upload it to GitHub or artifacts folder.")
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

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Action", ["Manual Input", "Bulk CSV"])

expected_type_cols = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
feature_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest'] + expected_type_cols

# ----------------------- Manual Input -----------------------
if app_mode == "Manual Input":
    st.header("Manual Input Prediction ✍🏻")
    with st.form(key="manual_form"):
        payment_type = st.selectbox("Payment Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
        step = st.number_input("Step", min_value=0, step=1)
        amount = st.number_input("Amount", min_value=0.0, step=0.01)
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, step=0.01)
        newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, step=0.01)
        oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, step=0.01)
        newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, step=0.01)
        manual_threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5)
        submit_manual = st.form_submit_button("Predict")

    if submit_manual:
        df = pd.DataFrame([{
            "step": step, "amount": amount, "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig, "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest, "type": payment_type
        }])

        # Encode type
        df_type = pd.get_dummies(df['type'], prefix='type')
        for col in expected_type_cols:
            if col not in df_type.columns:
                df_type[col] = 0
        df = pd.concat([df.drop('type', axis=1), df_type[expected_type_cols]], axis=1)

        # Scale numeric
        numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        df[numeric_cols] = preprocessor['scaler'].transform(df[numeric_cols])

        # Prepare input tensor
        X = df[feature_cols].values.astype(np.float32)
        X_tensor = torch.tensor(X)
        with torch.no_grad():
            prob = torch.sigmoid(model(X_tensor)).item()
        prediction = "Fraud" if prob > manual_threshold else "Not Fraud"
        st.success(f"Predicted Fraud Probability: {prob:.4f}")
        st.info(f"Prediction: {prediction}")

# ----------------------- Bulk CSV -----------------------
elif app_mode == "Bulk CSV":
    st.header("Bulk CSV Prediction via URL or Upload")
    with st.form(key="bulk_form"):
        file_url = st.text_input("Enter CSV URL (Google Drive / Dropbox / S3)")
        uploaded_file = st.file_uploader("Or upload CSV file directly", type="csv")
        bulk_threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5)
        submit_bulk = st.form_submit_button("Process CSV")

    if submit_bulk:
        try:
            if uploaded_file:
                df_iter = pd.read_csv(uploaded_file, chunksize=50000)
            elif file_url:
                import requests, io
                response = requests.get(file_url)
                response.raise_for_status()
                df_iter = pd.read_csv(io.StringIO(response.content.decode('utf-8')), chunksize=50000)
            else:
                st.warning("Provide a CSV file or URL.")
                st.stop()

            results = []
            total_chunks = 0
            for _ in df_iter:
                total_chunks += 1
            if uploaded_file:
                uploaded_file.seek(0)
                df_iter = pd.read_csv(uploaded_file, chunksize=50000)
            elif file_url:
                df_iter = pd.read_csv(io.StringIO(response.content.decode('utf-8')), chunksize=50000)

            progress_bar = st.progress(0)
            current_chunk = 0

            for chunk in df_iter:
                current_chunk += 1
                progress_bar.progress(current_chunk / total_chunks)

                # Ensure numeric columns
                numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                for col in numeric_cols:
                    if col not in chunk.columns:
                        chunk[col] = 0.0

                # Encode type columns
                if 'type' in chunk.columns:
                    df_type = pd.get_dummies(chunk['type'], prefix='type')
                    for col in expected_type_cols:
                        if col not in df_type.columns:
                            df_type[col] = 0
                    chunk = pd.concat([chunk.drop('type', axis=1), df_type[expected_type_cols]], axis=1)
                else:
                    for col in expected_type_cols:
                        chunk[col] = 0

                # Reorder columns
                X = chunk[feature_cols].values.astype(np.float32)
                X_tensor = torch.tensor(X)
                with torch.no_grad():
                    probs = torch.sigmoid(model(X_tensor)).numpy()
                chunk["fraud_probability"] = probs
                chunk["predicted_fraud"] = np.where(chunk["fraud_probability"] > bulk_threshold, "Fraud", "Not Fraud")
                results.append(chunk)

            df_final = pd.concat(results, ignore_index=True)
            st.success("Bulk predictions completed ✅")
            st.dataframe(df_final[["fraud_probability", "predicted_fraud"]].head(20))

            # Optional: Histogram
            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df_final["fraud_probability"], bins=50, kde=True, ax=ax)
            st.pyplot(fig)

            # Download CSV
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="fraud_predictions.csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
