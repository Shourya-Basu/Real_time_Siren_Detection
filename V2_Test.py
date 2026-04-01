import numpy as np
import librosa
import joblib

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "elm_siren_model.pkl"
TEST_AUDIO_PATH = "./Datasets/test_file_4(N).wav"   # <-- replace with your test file
SR = 16000
DURATION = 1.0
N_MFCC = 13
THRESHOLD = 0.5

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SR,
        mono=True,
        duration=DURATION
    )

    if len(audio) < SR * DURATION:
        raise ValueError("Audio file is too short (need at least 1 second).")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=400,        # 25 ms window
        hop_length=160    # 10 ms hop
    )

    zcr = librosa.feature.zero_crossing_rate(audio)

    features = np.hstack([
        mfcc.mean(axis=1),   # 13
        mfcc.std(axis=1),    # 13
        zcr.mean()           # 1
    ])

    return features

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)

W = model["W"]
b = model["b"]
beta = model["beta"]
mean = model["mean"]
std = model["std"]

print("✅ Model loaded successfully")

# =========================
# EXTRACT FEATURES
# =========================
features = extract_features(TEST_AUDIO_PATH)

# Normalize using training statistics
features = (features - mean) / std

# =========================
# ELM INFERENCE
# =========================
H = 1 / (1 + np.exp(-(features @ W.T + b)))

# ✅ SAFE scalar extraction (NO WARNING)
output = (H @ beta).item()

confidence = output * 100

# =========================
# RESULT
# =========================
print("-" * 40)

if output >= THRESHOLD:
    print("🚑 SIREN DETECTED")
else:
    print("🔇 NON-SIREN")

print(f"🔎 Siren Match Confidence: {confidence:.2f}%")
print("-" * 40)
