import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ----- PAGE CONFIG -----
st.set_page_config(page_title="Spam Classifier", layout="centered")

# ----- CUSTOM STYLING -----
st.markdown("""
<style>
/* Main styling */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #121212;
    color: #ffffff;
}

/* Input text areas */
.stTextArea textarea {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
    border-radius: 6px;
    border: 1px solid #444;
}

/* Buttons */
.stButton > button {
    background-color: #1e90ff;
    color: white;
    border-radius: 6px;
    border: none;
    padding: 0.5rem 1.5rem;
}

.stButton > button:hover {
    background-color: #1c7ed6;
}

/* Result box */
.result-box {
    padding: 0.8rem 1rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 1rem;
    text-align: center;
    margin-top: 1rem;
}

.success {
    background-color: #2ecc71;
    color: #ffffff;
}

.danger {
    background-color: #e74c3c;
    color: #ffffff;
}

/* File uploader text */
.css-1p05t8e {
    color: #ffffff !important;
}

/* Print-friendly style */
@media print {
    body, html {
        background: #ffffff !important;
        color: #000000 !important;
    }

    * {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
        color-adjust: exact !important;
    }

    .stTextArea textarea,
    .stButton > button,
    .result-box {
        background-color: #f8f8f8 !important;
        color: #000000 !important;
        border: 1px solid #ccc !important;
    }

    .success {
        background-color: #d4edda !important;
        color: #155724 !important;
    }

    .danger {
        background-color: #f8d7da !important;
        color: #721c24 !important;
    }

    h1, h2, h3, h4, h5 {
        color: #000 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ----- LOAD DATASET -----
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# ----- TRAIN MODEL -----
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    model = MultinomialNB()
    model.fit(X, y)
    return model, vectorizer

df = load_data()
model, vectorizer = train_model(df)

# ----- HEADER -----
st.title("📩 Spam Email Classifier")
st.caption("Predict whether a message is spam or not using machine learning.")

# ----- SINGLE PREDICTION -----
st.subheader("Single Message Prediction")
single_msg = st.text_input("Enter a message")

if st.button("Classify"):
    if single_msg.strip():
        vector = vectorizer.transform([single_msg])
        prediction = model.predict(vector)[0]
        result_class = "Spam" if prediction == 1 else "Not Spam"
        result_style = "danger" if prediction == 1 else "success"
        st.markdown(f'<div class="result-box {result_style}">{result_class}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a message.")

# ----- BATCH PREDICTION -----
st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with a column like 'message', 'text', 'email', etc.", type=['csv'])

if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        text_column = None
        for col in df_uploaded.columns:
            if col.lower() in ['message', 'text', 'email']:
                text_column = col
                break
        if text_column:
            st.success(f"Column selected: {text_column}")
            messages = df_uploaded[text_column].astype(str)
            X_batch = vectorizer.transform(messages)
            predictions = model.predict(X_batch)
            df_uploaded['Prediction'] = ['Spam' if p == 1 else 'Not Spam' for p in predictions]
            st.dataframe(df_uploaded)
        else:
            st.error("No suitable column found. Please include a column named 'message', 'text', or 'email'.")
    except Exception as e:
        st.error(f"Error: {e}")
