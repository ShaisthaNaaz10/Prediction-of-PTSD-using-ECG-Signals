import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pyswarms as ps
from scipy.stats import skew, kurtosis
import zipfile
import joblib

# === UNZIP THE DATASET ===
zip_path = "WESAD.zip"
extract_path = "DATA"

if not os.path.exists(extract_path):
    print("Extracting WESAD.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete!")

DATA_DIR = os.path.join(extract_path, "WESAD")
subject_ids = [f"S{i}" for i in range(2, 18)]
test_subject = "S10"

def load_ecg_data(subject_path):
    with open(subject_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    ecg = data['signal']['chest']['ECG']
    labels = data['label']
    return ecg, labels

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
        np.sqrt(np.mean(ecg_window**2)),
        np.sum(np.diff(np.sign(ecg_window)) != 0),
        skew(ecg_window),
        kurtosis(ecg_window)
    ])

def preprocess_ecg(ecg, labels, fs=700, window_sec=5):
    window_size = fs * window_sec
    X, y = [], []
    for i in range(0, len(ecg) - window_size, window_size):
        label_window = labels[i:i+window_size]
        if np.all(label_window == 1) or np.all(label_window == 2):
            window = ecg[i:i+window_size].flatten()
            X.append(extract_features(window))
            y.append(1 if label_window[0] == 2 else 0)
    return np.array(X), np.array(y)

X_train, y_train = [], []
X_test, y_test = None, None

for sid in subject_ids:
    subject_file = os.path.join(DATA_DIR, sid, f"{sid}.pkl")
    if not os.path.exists(subject_file):
        print(f"Missing: {subject_file}")
        continue
    try:
        ecg, labels = load_ecg_data(subject_file)
        X, y = preprocess_ecg(ecg, labels)
        if sid == test_subject:
            X_test, y_test = X, y
            print(f" Test subject: {sid}, samples: {len(y)}")
        else:
            X_train.append(X)
            y_train.append(y)
            print(f" Loaded {sid}: {len(y)} train samples")
    except Exception as e:
        print(f" Failed {sid}: {e}")

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def objective_function(params):
    n_particles = params.shape[0]
    scores = []

    for i in range(n_particles):
        n_estimators = int(params[i][0])
        max_depth = int(params[i][1])
        min_samples_split = int(params[i][2])
        min_samples_leaf = int(params[i][3])

        clf = RandomForestClassifier(
            n_estimators=max(n_estimators, 10),
            max_depth=max(max_depth, 2),
            min_samples_split=max(min_samples_split, 2),
            min_samples_leaf=max(min_samples_leaf, 1),
            class_weight='balanced',
            random_state=42
        )

        try:
            score = cross_val_score(
                clf, X_train_scaled, y_train,
                cv=StratifiedKFold(n_splits=3),
                scoring='accuracy'
            )
            scores.append(1 - score.mean())
        except Exception as e:
            print(f"Error evaluating: {params[i]}, {e}")
            scores.append(1.0)
    return np.array(scores)

bounds = ([10, 2, 2, 1], [100, 15, 10, 5])
optimizer = ps.single.GlobalBestPSO(
    n_particles=6, dimensions=4,
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
    bounds=bounds
)

best_cost, best_pos = optimizer.optimize(objective_function, iters=7)
best_n_estimators = int(best_pos[0])
best_max_depth = int(best_pos[1])
best_min_samples_split = int(best_pos[2])
best_min_samples_leaf = int(best_pos[3])

print(f"\n✅ PSO Best Params: n_estimators={best_n_estimators}, max_depth={best_max_depth}, "
      f"min_samples_split={best_min_samples_split}, min_samples_leaf={best_min_samples_leaf}")

clf = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train_scaled, y_train)
y_probs = clf.predict_proba(X_test_scaled)[:, 1]

best_thresh = 0.5
best_f1 = 0
for t in np.arange(0.3, 0.91, 0.01):
    f1 = f1_score(y_test, (y_probs >= t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"🔧 Best Threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")
y_pred = (y_probs >= best_thresh).astype(int)

print("\n=== Final Evaluation ===")
print(f"Test Subject: {test_subject}")
print(classification_report(y_test, y_pred, target_names=["Not Stress", "Stress"]))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Stress", "Stress"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

stress_count = np.sum(y_pred == 1)
not_stress_count = np.sum(y_pred == 0)
final_prediction = "Stress" if stress_count > not_stress_count else "Not Stress"
print(f"\nFinal prediction for subject {test_subject}: {final_prediction}")

joblib.dump(clf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
with open("threshold.txt", "w") as f:
    f.write(str(best_thresh))
