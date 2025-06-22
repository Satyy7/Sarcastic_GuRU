import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('sarcasm_model.h5')
    return model

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

st.title("Sarcasm Detection in News Headlines")

headline = st.text_area("Enter a news headline:")

if st.button("Check for Sarcasm"):
    if headline:
        cleaned = clean_text(headline)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=32)
        prediction = model.predict(padded)[0][0]
        result = "Sarcastic" if prediction >= 0.5 else "Not Sarcastic"
        st.write(f"Prediction: **{result}** (Confidence: {prediction:.2f})")
    else:
        st.warning("Please enter a headline.")