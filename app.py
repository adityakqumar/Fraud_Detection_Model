import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="💸 Fraud Detection App", layout="centered")
st.title("💸 Online Payment Fraud Detection")
st.markdown("Enter transaction details:")

# 1. Common Inputs
step = st.number_input("Step (Time)", min_value=1, max_value=744)
amount = st.number_input("Amount", min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, format="%.2f")
isFlaggedFraud = st.selectbox("Is Flagged Fraud?", [0, 1])

# 2. One-hot Transaction Type
trans_type = st.selectbox("Transaction Type", ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])

# One-hot encode the selected type
CASH_OUT = 1 if trans_type == 'CASH_OUT' else 0
DEBIT = 1 if trans_type == 'DEBIT' else 0
PAYMENT = 1 if trans_type == 'PAYMENT' else 0
TRANSFER = 1 if trans_type == 'TRANSFER' else 0

# Prediction
if st.button("🔍 Predict Fraud"):
    input_data = np.array([
        step, amount, oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest, isFlaggedFraud,
        CASH_OUT, DEBIT, PAYMENT, TRANSFER
    ]).reshape(1, -1)

    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1] * 100

        if prediction == 1:
            st.error(f"⚠️ FRAUD DETECTED!\nConfidence: {proba:.2f}%")
        else:
            st.success(f"✅ Transaction is SAFE.\nConfidence: {100 - proba:.2f}%")
    except Exception as e:
        st.warning(f"Prediction failed: {e}")
