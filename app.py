import os, json, joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Retinopathy Predictor")

# ---------- figure out feature list ----------
if os.path.exists("feature_names.json"):
    FEATURES = json.load(open("feature_names.json"))
elif os.path.exists("pronostico_dataset.csv"):
    _df = pd.read_csv("pronostico_dataset.csv", sep=";")
    _df = _df.drop(columns=[c for c in ["prognosis","ID"] if c in _df.columns], errors="ignore")
    FEATURES = _df.columns.tolist()
else:
    st.error("Missing feature_names.json or pronostico_dataset.csv to infer inputs.")
    st.stop()

# ---------- pick a saved model ----------
MODEL_FILES = sorted([f for f in os.listdir(".") if f.lower().endswith((".pkl",".joblib"))])
if not MODEL_FILES:
    st.error("No .pkl/.joblib files found.")
    st.stop()

default_idx = MODEL_FILES.index("SVM_best_model.pkl") if "SVM_best_model.pkl" in MODEL_FILES else 0
model_file = st.sidebar.selectbox("Model file", MODEL_FILES, index=default_idx)

st.title("Diabetic Retinopathy Prediction")

# defaults from dataset medians
defaults = {}
if os.path.exists("pronostico_dataset.csv"):
    df_med = pd.read_csv("pronostico_dataset.csv", sep=";")
    df_med = df_med.drop(columns=[c for c in ["prognosis","ID"] if c in df_med.columns], errors="ignore")
    defaults = df_med.median(numeric_only=True).to_dict()

# inputs
vals = {}
cols = st.columns(2)
for i, f in enumerate(FEATURES):
    with cols[i % 2]:
        vals[f] = st.number_input(f, value=float(defaults.get(f, 0.0)))

# optional external scaler for legacy models
SCALER = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None

if st.button("Predict"):
    model = joblib.load(model_file)
    X = pd.DataFrame([[vals[c] for c in FEATURES]], columns=FEATURES)

    # If it's a Pipeline, preprocessing is handled; otherwise apply scaler for SVM/KNN if available.
    is_pipeline = isinstance(model, Pipeline) or hasattr(model, "named_steps")
    fname = model_file.lower()
    needs_scale = (("svm" in fname) or ("knn" in fname)) and (not is_pipeline) and (SCALER is not None)
    if needs_scale:
        X = SCALER.transform(X)

    pred = int(model.predict(X)[0])
    st.success(f"Prediction: {'Retinopathy' if pred==1 else 'No Retinopathy'}")

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0,1])
        st.write(f"Estimated probability: {proba:.3f}")
