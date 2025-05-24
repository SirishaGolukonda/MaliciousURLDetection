from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model = joblib.load("models/url_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def extract_features(url):
    return [url]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form["url"]
    features = extract_features(url)
    X = vectorizer.transform(features)
    prediction = model.predict(X)[0]
    result = "Malicious" if prediction == 1 else "Benign"
    return render_template("index.html", url=url, result=result)

if __name__ == "__main__":
    app.run(debug=True)
