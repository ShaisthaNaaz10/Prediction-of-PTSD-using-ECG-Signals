from flask import Flask, render_template, request
import os
import pickle
import numpy as np
import joblib
from scipy.stats import skew, kurtosis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model, scaler, threshold
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
with open("threshold.txt", "r") as f:
    threshold = float(f.read())

# Feature extraction from ECG
def extract_features(ecg_window):
    return np.array([
        np.mean(ecg_window),
        np.std(ecg_window),
        np.min(ecg_window),
        np.max(ecg_window),
        np.median(ecg_window),
        np.percentile(ecg_window, 25),
        np.percentile(ecg_window, 75),
        np.var(ecg_window),
        np.ptp(ecg_window),
        np.sqrt(np.mean(ecg_window ** 2)),
        np.sum(np.diff(np.sign(ecg_window)) != 0),
        skew(ecg_window),
        kurtosis(ecg_window)
    ])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
          return render_template("home.html", predicted_stress="No file uploaded.")
        if file and file.filename.endswith(".pkl"):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
                ecg = data['signal']['chest']['ECG']
                labels = data['label']

                fs = 700
                window_size = fs * 5
                for i in range(0, len(ecg) - window_size, window_size):
                    label_window = labels[i:i+window_size]
                    if np.all(label_window == 1) or np.all(label_window == 2):
                        window = ecg[i:i+window_size].flatten()
                        features = extract_features(window).reshape(1, -1)
                        break
                else:
                    return render_template("home.html", predicted_stress="Invalid label window")

                scaled = scaler.transform(features)
                prob = model.predict_proba(scaled)[0][1]
                prediction = 1 if prob >= threshold else 0
                predicted_stress = "Stress" if prediction == 1 else "Not Stress"

                return render_template("home.html", predicted_stress=predicted_stress)

            except Exception as e:
                print("Error:", e)
                return render_template("home.html", predicted_stress="Error during processing.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
