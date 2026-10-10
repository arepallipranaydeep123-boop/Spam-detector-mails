import subprocess
import sys
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_model.joblib"
DATA_PATH = BASE_DIR / "data" / "emails.csv"


def ensure_model_ready():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        subprocess.run([sys.executable, str(BASE_DIR / "make_sample_data.py")], cwd=str(BASE_DIR), check=True)
    if not MODEL_PATH.exists():
        subprocess.run([sys.executable, str(BASE_DIR / "train.py")], cwd=str(BASE_DIR), check=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model was not created successfully.")


@st.cache_resource
def load_model():
    try:
        ensure_model_ready()
    except Exception as exc:
        st.error(f"Model not found and could not be created automatically: {exc}")
        st.stop()
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("Email Spam Detector")
st.caption("A lightweight spam classifier built with TF-IDF + scikit-learn.")

model = load_model()

message = st.text_area(
    "Paste an email or message",
    value="Congratulations! You won a free iPhone. Click http://bit.ly/x",
    height=180,
)

if st.button("Check spam status"):
    if not message.strip():
        st.warning("Please enter some text to classify.")
    else:
        proba = model.predict_proba([message])[0][1]
        label = "SPAM" if proba >= 0.5 else "HAM"
        st.metric("Prediction", label)
        st.progress(float(proba))
        st.write(f"Spam probability: {proba:.2%}")

        if label == "SPAM":
            st.warning("This message looks like spam.")
        else:
            st.success("This message looks like legitimate email.")
