# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, render_template

# -----------------------------
# Load and preprocess dataset
# -----------------------------
data = pd.read_csv(r'C:\Users\sai dhiraj kumar\.anaconda\traffic_dataset.csv')  # Make sure this file exists in same folder

# Encode target labels (Low, Moderate, High)
le = LabelEncoder()
data["target"] = le.fit_transform(data["target"])  # 0: High, 1: Low, 2: Moderate

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['hour']),
            float(request.form['day_of_week']),
            float(request.form['vehicle_count']),
            float(request.form['avg_speed']),
            float(request.form['weather_condition'])
        ]
        final_input = scaler.transform([features])
        prediction = model.predict(final_input)
        output = label_encoder.inverse_transform(prediction)[0]
        return f"<h2>Predicted Traffic Congestion Level: {output}</h2><a href='/'>Back</a>"
    except Exception as e:
        return f"<h2>Error: {e}</h2><a href='/'>Back</a>"

if __name__ == "__main__":
    app.run(debug=True)
