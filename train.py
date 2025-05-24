import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Auto-create folders if missing
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)


# Load dataset
data = pd.read_csv(r"C:\Users\siris\OneDrive\Desktop\MSD\Mlalacious URL Detection\malicious_url_detection\malicious_phish.csv")
data["label"] = data["type"].apply(lambda x: 1 if x != "benign" else 0)

# Feature and label
X = data["url"]
y = data["label"]

# Vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# ✅ Correct: split the vectorized features
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)


# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png")

# Save model and vectorizer
joblib.dump(model, "models/url_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
