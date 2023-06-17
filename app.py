from flask import Flask, render_template, request
from keras.models import load_model
import nltk
import os
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

stopwords_eng = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Initialize the Tokenizer object
max_words = 1000
t = Tokenizer(num_words=max_words)

def clean_txt(txt):
    txt = re.sub(r'([^\s\w])+', ' ', txt)
    txt = " ".join([lemmatizer.lemmatize(word) for word in txt.split()
                    if not word in stopwords_eng])
    txt = txt.lower()
    return txt

app = Flask(__name__)
model = load_model('./sms_classifier_model.h5')

def preprocess_message(message):
    message = re.sub(r'([^\s\w])+', ' ', message)
    message = " ".join([lemmatizer.lemmatize(word) for word in message.split()
                        if word not in stopwords_eng])
    message = message.lower()
    return message

def predict_message(pred_text):
    sequences = t.texts_to_sequences([preprocess_message(pred_text)])
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
    p = model.predict(sequences_matrix)[0]
    return p[0], "ham" if p < 0.5 else "spam"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    message = request.form['message']
    prediction = predict_message(message)
    result = prediction[1]
    return render_template('result.html', result=result)

if __name__ == '__main__':
    nltk_data_path = 'D:/Neural Network SMS Text Classifier/data/nltk_data'
    nltk.data.path.append(nltk_data_path)
    app.run()
