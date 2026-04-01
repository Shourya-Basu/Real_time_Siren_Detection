import os
import numpy as np
import librosa
import joblib
from numpy.linalg import pinv

# =========================
# CONFIGURATION
# =========================
DATASET_PATH = "Datasets"
SR = 16000              # Sampling rate
DURATION = 1.0          # seconds
N_MFCC = 13
HIDDEN_NEURONS = 80     # Pi-friendly size

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    try:
        audio, sr = librosa.load(
            file_path,
            sr=SR,
            mono=True,
            duration=DURATION
        )

        if len(audio) < SR * DURATION:
            return None

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=400,
            hop_length=160
        )

        zcr = librosa.feature.zero_crossing_rate(audio)

        features = np.hstack([
            mfcc.mean(axis=1),   # 13
            mfcc.std(axis=1),    # 13
            zcr.mean()           # 1
        ])

        return features

    except Exception as e:
        print(f"❌ Error processing {file_path}")
        return None

# =========================
# LOAD DATASET (NO CSV)
# =========================
X = []
y = []

# ---- NON-SIREN ----
nonsiren_path = os.path.join(DATASET_PATH, "nonsiren")
for category in os.listdir(nonsiren_path):
    cat_path = os.path.join(nonsiren_path, category)
    for file in os.listdir(cat_path):
        if file.endswith(".wav"):
            feat = extract_features(os.path.join(cat_path, file))
            if feat is not None:
                X.append(feat)
                y.append(0)

# ---- SIREN ----
siren_path = os.path.join(DATASET_PATH, "siren")
for category in os.listdir(siren_path):
    cat_path = os.path.join(siren_path, category)
    for file in os.listdir(cat_path):
        if file.endswith(".wav"):
            feat = extract_features(os.path.join(cat_path, file))
            if feat is not None:
                X.append(feat)
                y.append(1)

X = np.array(X)
y = np.array(y).reshape(-1, 1)

print("✅ Dataset loaded")
print("X shape:", X.shape)  # (samples, 27)
print("y shape:", y.shape)

# =========================
# FEATURE NORMALIZATION
# =========================
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8
X = (X - mean) / std

# =========================
# TRAIN ELM
# =========================
input_size = X.shape[1]

W = np.random.randn(HIDDEN_NEURONS, input_size)
b = np.random.randn(HIDDEN_NEURONS)

H = 1 / (1 + np.exp(-(X @ W.T + b)))
beta = pinv(H) @ y

# =========================
# TRAINING ACCURACY
# =========================
pred = (H @ beta > 0.5).astype(int)
accuracy = (pred == y).mean() * 100
print(f"🎯 Training Accuracy: {accuracy:.2f}%")

# =========================
# SAVE MODEL
# =========================
joblib.dump({
    "W": W,
    "b": b,
    "beta": beta,
    "mean": mean,
    "std": std
}, "elm_siren_model.pkl")

print("💾 Model saved as elm_siren_model.pkl")
