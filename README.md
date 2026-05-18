# 📄 Resume Screening AI using NLP & Deep Learning
App Link: https://resumescreeningdl-c5bhiregdkpvzxjocygyyr.streamlit.app/

## 🚀 Project Overview

This project is an AI-powered Resume Screening Web Application built using:

* Natural Language Processing (NLP)
* Deep Learning (BiLSTM)
* TensorFlow / Keras
* Streamlit

The application analyzes uploaded resumes and predicts suitable job categories using a trained deep learning model.

Users can:

* Upload resumes in PDF or TXT format
* Get AI-based job category predictions
* View Top-3 predicted categories with confidence scores
* See AI-style resume insights

---

# 🎯 Problem Statement

Recruiters often receive hundreds of resumes for different job roles.

Manually reviewing resumes:

* takes time
* requires effort
* may lead to inconsistent screening

This project automates the initial resume screening process using NLP and Deep Learning.

---

# 🧠 Technologies Used

| Technology         | Purpose                     |
| ------------------ | --------------------------- |
| Python             | Programming Language        |
| Pandas             | Data Handling               |
| NumPy              | Numerical Operations        |
| NLTK               | Text Processing             |
| TensorFlow / Keras | Deep Learning               |
| BiLSTM             | Resume Classification Model |
| Scikit-learn       | Preprocessing & Evaluation  |
| Streamlit          | Web Application             |
| Git & GitHub       | Version Control             |

---

# 📂 Project Structure

```text
Resume_Screening_DL/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
│
├── data/
│   └── Resume.csv
│
├── artifacts/
│   ├── tokenizer.pkl
│   └── label_mapping.pkl
│
├── models/
│   └── resume_screening_model.keras
│
└── notebooks/
    └── 02_clean_resume_screening.ipynb
```

---

# ⚙️ Workflow

## 1. Data Loading

The resume dataset is loaded using Pandas.

---

## 2. Text Cleaning

The resumes are cleaned by:

* converting text to lowercase
* removing URLs
* removing special characters
* removing extra spaces

---

## 3. Label Encoding

Job categories are converted into numerical labels for model training.

---

## 4. Train-Test Split

The dataset is split into:

* Training Data
* Testing Data

---

## 5. Tokenization & Padding

Text data is converted into numerical sequences using:

* Tokenizer
* Padding

This allows the neural network to process resume text.

---

## 6. Deep Learning Model

The model architecture:

```text
Embedding Layer
↓
Bidirectional LSTM (BiLSTM)
↓
Dropout Layer
↓
Dense Softmax Output Layer
```

### Why BiLSTM?

BiLSTM understands text context in both:

* forward direction
* backward direction

This improves resume classification performance.

---

# 📊 Model Performance

### Validation Accuracy

```text
~70% Accuracy
```

The project predicts:

* Top-3 job categories
* Confidence scores

instead of relying only on a single prediction.

---

# 🌐 Streamlit Web Application Features

## ✅ Features

* Professional UI
* PDF resume upload
* TXT resume upload
* Top-3 predictions
* Confidence score visualization
* AI-style resume insights
* Responsive layout

---

# 📸 Application Screenshots

## Home Page

*Add screenshot here*

---

## Prediction Results

*Add screenshot here*

---

# ▶️ How to Run the Project

## 1. Clone Repository

```bash
git clone https://github.com/Surej006/Resume_Screening_DL.git
```

---

## 2. Move into Project Folder

```bash
cd Resume_Screening_DL
```

---

## 3. Create Environment (Optional)

```bash
conda create -n resume_dl python=3.10 -y
conda activate resume_dl
```

---

## 4. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 5. Run Streamlit App

```bash
streamlit run app.py
```

---

# 🧪 Future Improvements

Possible future upgrades:

* Transformer Models (BERT / DistilBERT)
* Better Resume Parsing
* Skill Extraction
* ATS Score Prediction
* AI Career Suggestions
* Resume Improvement Recommendations

---

# 📌 Key Learnings

This project helped in understanding:

* NLP preprocessing
* Text classification
* Deep learning using BiLSTM
* Model evaluation
* Streamlit deployment
* Git & GitHub workflow
* End-to-end ML project development

---

# 👨‍💻 Author

### Surej Krishnan

