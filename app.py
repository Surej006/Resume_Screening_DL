import streamlit as st
import numpy as np
import pickle
import re
from pypdf import PdfReader

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="📄",
    layout="wide"
)

MAX_LEN = 150

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2f7 0%, #dbeafe 45%, #f8fafc 100%);
}

.hero {
    background: linear-gradient(90deg, #0f172a, #1e3a8a, #2563eb);
    padding: 40px;
    border-radius: 22px;
    color: white;
    text-align: center;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.18);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 44px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    color: #dbeafe;
}

.card {
    background: rgba(255,255,255,0.92);
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(15,23,42,0.12);
    border: 1px solid rgba(255,255,255,0.6);
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg, #eff6ff, #ffffff);
    padding: 20px;
    border-radius: 16px;
    border-left: 7px solid #2563eb;
    margin-bottom: 18px;
    box-shadow: 0px 4px 15px rgba(37,99,235,0.12);
}

.ai-box {
    background: #f8fafc;
    border-left: 6px solid #10b981;
    padding: 20px;
    border-radius: 16px;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model and Artifacts
# =========================
@st.cache_resource
def load_files():
    model = load_model("models/resume_screening_model.keras")

    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("artifacts/label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)

    # Convert category -> number into number -> category
    label_mapping = {v: k for k, v in label_mapping.items()}

    return model, tokenizer, label_mapping


with st.spinner("Loading AI model..."):
    model, tokenizer, label_mapping = load_files()

# =========================
# Helper Functions
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
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


def predict_resume(text):
    cleaned = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded)[0]
    top_indices = prediction.argsort()[-3:][::-1]

    results = []

    for idx in top_indices:
        results.append({
            "category": label_mapping[int(idx)],
            "confidence": float(prediction[idx] * 100)
        })

    return results, cleaned


def ai_style_analysis(cleaned_text, top_category):
    data_keywords = [
        "python", "machine", "learning", "data", "sql", "analytics",
        "deep", "model", "pandas", "numpy", "tensorflow", "keras",
        "visualization", "statistics", "classification", "regression"
    ]

    found_keywords = [
        word for word in data_keywords
        if word in cleaned_text.split()
    ]

    analysis = f"""
    The resume shows strongest similarity to **{top_category}** based on the learned resume patterns.

    **Relevant skills detected:** {", ".join(found_keywords[:10]) if found_keywords else "No strong technical keywords detected."}

    **AI Recommendation:**  
    Review the top 3 suggested categories instead of relying only on the first prediction, because resume categories can overlap.  
    For better results, the resume should clearly highlight role-specific skills, tools, projects, and experience.
    """

    return analysis

# =========================
# Header
# =========================
st.markdown("""
<div class="hero">
    <h1>📄 Resume Screening AI</h1>
    <p>Upload a resume and get AI-based job category suggestions using NLP and Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("⚙️ Project Info")

    st.write("""
    This application uses an NLP-based BiLSTM model to classify resumes into job categories.
    """)

    st.markdown("---")

    st.subheader("Tech Stack")
    st.write("""
    - Python  
    - TensorFlow / Keras  
    - NLP  
    - BiLSTM  
    - Streamlit  
    - PDF Text Extraction  
    """)

    st.markdown("---")

    st.warning(
        "Predictions are AI suggestions. For real hiring decisions, human review is still required."
    )

# =========================
# Main Layout
# =========================
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📤 Upload Resume")

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT resume",
        type=["pdf", "txt"]
    )

    resume_text = ""

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

        st.success("Resume uploaded successfully!")

        with st.expander("View Extracted Resume Text"):
            st.text_area("Extracted Text", resume_text, height=250)

    else:
        resume_text = st.text_area(
            "Or paste resume text manually",
            height=300
        )

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🎯 Prediction Result")

    if st.button("Analyze Resume", use_container_width=True):
        if resume_text.strip() == "":
            st.warning("Please upload or paste resume text.")
        else:
            with st.spinner("Analyzing resume..."):
                results, cleaned_text = predict_resume(resume_text)

            st.success("Analysis completed!")

            st.markdown("### Top 3 Suggested Categories")

            for i, result in enumerate(results, start=1):
                st.markdown(
                    f"""
                    <div class="result-card">
                        <h4>{i}. {result['category']}</h4>
                        <p><b>Confidence:</b> {result['confidence']:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(result["confidence"] / 100)

            st.markdown('<div class="ai-box">', unsafe_allow_html=True)
            st.markdown("### 🤖 AI Resume Insight")
            st.markdown(ai_style_analysis(cleaned_text, results[0]["category"]))
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Upload a resume and click **Analyze Resume** to see predictions.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("""
<div class="footer">
    Built with Python, NLP, TensorFlow, BiLSTM and Streamlit
</div>
""", unsafe_allow_html=True)