import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Paths
MODEL_PATH = "models/url_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
LOG_PATH = "logs/user_predictions.csv"
CONFUSION_MATRIX_PATH = "static/confusion_matrix_live.png"

# Create folders if missing
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# App settings
st.set_page_config(page_title="Malicious URL Detector", layout="centered")
st.title("🔍 Malicious URL Detection App")

# Prediction function
def predict_url(url):
    X = vectorizer.transform([url])
    return model.predict(X)[0]

# Log predictions
def log_prediction(url, prediction):
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now(),
        "url": url,
        "prediction": "Malicious" if prediction == 1 else "Benign"
    }])
    if os.path.exists(LOG_PATH):
        log_entry.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(LOG_PATH, index=False)

# Generate confusion matrix and metrics
def generate_confusion_matrix():
    data = pd.read_csv("malicious_phish.csv")  # Update path if needed
    data["label"] = data["type"].apply(lambda x: 1 if x != "benign" else 0)
    X_vec = vectorizer.transform(data["url"])
    y_true = data["label"]
    y_pred = model.predict(X_vec)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Save confusion matrix plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Live)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    return acc, prec, rec, f1

# URL input
st.header("🔗 Check a URL")
url_input = st.text_input("Enter a URL")

if url_input:
    pred = predict_url(url_input)
    result_text = "🛑 Malicious" if pred == 1 else "✅ Benign"
    st.markdown(f"### Prediction: **{result_text}**")
    log_prediction(url_input, pred)

# Batch prediction
st.header("📄 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with a 'url' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "url" not in df.columns:
            st.error("The uploaded file must contain a 'url' column.")
        else:
            df["prediction"] = df["url"].apply(predict_url)
            df["prediction_label"] = df["prediction"].apply(lambda x: "Malicious" if x == 1 else "Benign")
            st.success("Batch prediction completed.")
            st.dataframe(df[["url", "prediction_label"]])
            st.download_button("Download Results", df.to_csv(index=False), file_name="batch_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Confusion matrix & metrics
st.header("📊 Confusion Matrix & Metrics")
if st.button("Regenerate Confusion Matrix & Metrics"):
    acc, prec, rec, f1 = generate_confusion_matrix()
    st.success("Confusion matrix and metrics updated!")

if os.path.exists(CONFUSION_MATRIX_PATH):
    st.image(CONFUSION_MATRIX_PATH, caption="Model Confusion Matrix", use_column_width=True)
    st.markdown(f"**Accuracy:** {acc:.2f}")
    st.markdown(f"**Precision:** {prec:.2f}")
    st.markdown(f"**Recall:** {rec:.2f}")
    st.markdown(f"**F1 Score:** {f1:.2f}")

# Log filtering
if os.path.exists(LOG_PATH):
    st.header("📜 User Prediction Logs")

    logs_df = pd.read_csv(LOG_PATH)
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])

    with st.expander("🔍 Filter Logs"):
        prediction_filter = st.multiselect("Prediction Type", ["Benign", "Malicious"], default=["Benign", "Malicious"])
        start_date = st.date_input("Start Date", logs_df["timestamp"].min().date())
        end_date = st.date_input("End Date", logs_df["timestamp"].max().date())
        filtered_logs = logs_df[
            (logs_df["prediction"].isin(prediction_filter)) &
            (logs_df["timestamp"].dt.date >= start_date) &
            (logs_df["timestamp"].dt.date <= end_date)
        ]
    st.dataframe(filtered_logs.tail(20))

# Model retraining
st.header("🔁 Retrain Model")
retrain_file = st.file_uploader("Upload CSV to retrain model (must have 'url' and 'type' columns)", type=["csv"], key="retrain")

if retrain_file:
    try:
        retrain_df = pd.read_csv(retrain_file)
        if "url" not in retrain_df.columns or "type" not in retrain_df.columns:
            st.error("CSV must contain 'url' and 'type' columns.")
        else:
            retrain_df["label"] = retrain_df["type"].apply(lambda x: 1 if x != "benign" else 0)
            new_vectorizer = TfidfVectorizer()
            X_train = new_vectorizer.fit_transform(retrain_df["url"])
            y_train = retrain_df["label"]

            new_model = LogisticRegression()
            new_model.fit(X_train, y_train)

            # Save new model and vectorizer
            joblib.dump(new_model, MODEL_PATH)
            joblib.dump(new_vectorizer, VECTORIZER_PATH)

            st.success("Model retrained and updated successfully! Please refresh the app to use the new model.")
    except Exception as e:
        st.error(f"Retraining failed: {e}")
