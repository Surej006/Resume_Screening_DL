import streamlit as st
import numpy as np
import pickle
import re
from pypdf import PdfReader

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200

st.set_page_config(page_title="Resume Screening App", layout="centered")

st.title("📄 Resume Screening App")
st.write("Upload a resume file or paste resume text to predict the job category.")

@st.cache_resource
def load_files():
    model = load_model("models/resume_screening_model")

    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("artifacts/label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)

    return model, tokenizer, label_mapping

model, tokenizer, label_mapping = load_files()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text

uploaded_file = st.file_uploader(
    "Upload Resume (.pdf or .txt)",
    type=["pdf", "txt"]
)

resume_text = ""

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

    st.success("Resume uploaded successfully!")
    st.text_area("Extracted Resume Text", resume_text, height=250)
else:
    resume_text = st.text_area("Or paste Resume Text Here", height=300)

if st.button("Predict Category"):
    if resume_text.strip() == "":
        st.warning("Please upload or paste resume text.")
    else:
        cleaned = clean_text(resume_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        prediction = model.predict(padded)[0]

        top_indices = prediction.argsort()[-3:][::-1]

        st.subheader("Top 3 Predictions")

        for idx in top_indices:
            category = label_mapping[int(idx)]
            confidence = prediction[idx] * 100
            st.write(f"**{category}** — {confidence:.2f}%")