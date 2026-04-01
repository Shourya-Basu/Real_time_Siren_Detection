import numpy as np
import librosa
import joblib
import os

# =====================================================
# PATH HANDLING (FIXED FOR YOUR STRUCTURE)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "elm_siren_model.pkl")
TEST_AUDIO_PATH = os.path.join(BASE_DIR, "test_file_2(P).wav")  # change if needed

# =====================================================
# CONFIGURATION
# =====================================================
SR = 16000
DURATION = 1.0
N_MFCC = 13
THRESHOLD = 0.5

# =====================================================
# LOAD MODEL
# =====================================================
print("📦 Loading model from:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

W = model["W"]
b = model["b"]
beta = model["beta"]
mean = model["mean"]
std = model["std"]

print("✅ Model loaded successfully")

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features(file_path):
    print("🎧 Loading audio:", file_path)

    audio, sr = librosa.load(
        file_path,
        sr=SR,
        mono=True,
        duration=DURATION
    )

    if len(audio) < SR * DURATION:
        raise ValueError("Audio file must be at least 1 second long")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=400,
        hop_length=160
    )

    zcr = librosa.feature.zero_crossing_rate(audio)

    return np.hstack([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        zcr.mean()
    ])

# =====================================================
# PREDICTION
# =====================================================
features = extract_features(TEST_AUDIO_PATH)

# Normalize
features = (features - mean) / std

# ELM forward pass
H = 1 / (1 + np.exp(-(features @ W.T + b)))
output = (H @ beta).item()
confidence = output * 100

# =====================================================
# RESULT
# =====================================================
print("-" * 45)

if output >= THRESHOLD:
    print("🚑 SIREN DETECTED")
else:
    print("🔇 NON-SIREN")

print(f"🔎 Siren Match Confidence: {confidence:.2f}%")
print(f"📁 Tested File: {os.path.basename(TEST_AUDIO_PATH)}")

print("-" * 45)
