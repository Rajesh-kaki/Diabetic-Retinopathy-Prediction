
import json, pickle, numpy as np, streamlit as st

st.set_page_config(page_title="Diabetic Retinopathy Prediction")
st.title("Diabetic Retinopathy Prediction")

# Load model, scaler, and feature order
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("feature_order.json", "r") as f:
    feature_order = json.load(f)

# UI
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age (years)", 0, 100, 50)
    systolic_bp = st.number_input("Systolic BP", 50.0, 250.0, 120.0)
with col2:
    diastolic_bp = st.number_input("Diastolic BP ", 30.0, 150.0, 80.0)
    cholesterol = st.number_input("Cholesterol ", 50.0, 500.0, 180.0)

# Prepare input
vals = {'age': age, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp, 'cholesterol': cholesterol}
x = np.array([[vals[c] for c in feature_order]], dtype=float)


if st.button("Predict"):
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0][1])

    if pred == 1:
        st.markdown("### Retinopathy: **Yes**")
    else:
        st.markdown("### Retinopathy: **No**")
