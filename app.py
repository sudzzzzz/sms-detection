import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color:rgb(0, 0, 0);
    }
    .stApp {
        background:rgba(0, 0, 0, 0.47);
    }
    .title {
        font-size: 44px;
        font-weight: bold;
        background: -webkit-linear-gradient(#2b4eff,rgb(83, 62, 166));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        font-weight: lighter;
        color: #555;
        text-align: center;
        margin-bottom: 40px;
        font-style: italic;
    }
    .text-box {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
    }
    .button {
        background-color: #2b4eff;
        color: #fff;
        border-radius: 5px;
        padding: 10px;
        font-size: 18px;
        cursor: pointer;
        text-align: center;
    }
    .button:hover {
        background-color: #1a35cc;
    }
    .result-spam {
        font-size: 28px;
        font-weight: bold;
        color: red;
        text-align: center;
    }
    .result-notspam {
        font-size: 28px;
        font-weight: bold;
        color: green;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>🔍 SMS Spam Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Identify Messages With Sudeshna Sarkar</div>", unsafe_allow_html=True)

input_sms = st.text_area("Enter the SMS:", height=150, max_chars=500, help="Paste or type your SMS here")

if st.button('Predict', help="Click to analyze the SMS"):
    
    transformed_sms = transform_text(input_sms)

    vector_input = tk.transform([transformed_sms])

    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.markdown("<div class='result-spam'>Spam</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-notspam'>Ham</div>", unsafe_allow_html=True)
