# ============================================================
# Emergency Siren Detection using MFCC + ZCR + ELM
# With Dominant Frequency Estimation
# Target: Raspberry Pi 4
# ============================================================

# -------------------- IMPORTS --------------------
import os
import time
import pickle
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import sounddevice as sd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------- AUDIO PARAMETERS --------------------
SAMPLE_RATE = 16000
FRAME_LENGTH = int(0.025 * SAMPLE_RATE)   # 400
HOP_LENGTH = int(0.010 * SAMPLE_RATE)     # 160
N_FFT = 512
N_MFCC = 13

# -------------------- FEATURE EXTRACTION --------------------
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Skip extremely short clips
        if len(y) < 0.1 * sr:
            return None

        # Pre-emphasis
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            win_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            window="hamming"
        )

        # Delta MFCC (safe width)
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=3)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        # Stack features
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2, zcr])

        # Temporal pooling
        feature_vector = np.hstack([
            np.mean(features, axis=1),
            np.std(features, axis=1)
        ])

        return feature_vector

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# -------------------- DOMINANT FREQUENCY --------------------
def get_dominant_frequency(y, sr):
    y = y * np.hanning(len(y))          # windowing
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    spectrum[0] = 0                     # remove DC
    return freqs[np.argmax(spectrum)]

# -------------------- ELM MODEL --------------------
class ELM:
    def __init__(self, input_size, hidden_size=300):
        self.W = np.random.randn(input_size, hidden_size)
        self.b = np.random.randn(hidden_size)

    def _sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        H = self._sigmoid(X @ self.W + self.b)
        self.beta = np.linalg.pinv(H) @ y

    def predict(self, X):
        H = self._sigmoid(X @ self.W + self.b)
        return H @ self.beta

    def predict_class(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)

# -------------------- TRAINING --------------------
def train_model(csv_path="dataset_labels.csv"):
    df = pd.read_csv(csv_path)

    X, y = [], []
    for _, row in df.iterrows():
        f = extract_features(row["filepath"])
        if f is not None:
            X.append(f)
            y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    print("Feature matrix:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    elm = ELM(input_size=X.shape[1], hidden_size=300)
    elm.fit(X_train, y_train)

    y_pred = elm.predict_class(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    with open("elm_model.pkl", "wb") as f:
        pickle.dump(elm, f)

    print("Model saved as elm_model.pkl")

# -------------------- LOAD MODEL --------------------
def load_model():
    with open("elm_model.pkl", "rb") as f:
        return pickle.load(f)

# -------------------- CONTINUOUS DETECTION --------------------
def continuous_detection(elm, threshold=0.4):
    print("🎧 Continuous Siren Detection Started")
    print("Press Ctrl+C to stop\n")

    freq_history = []

    while True:
        try:
            # Record audio
            audio = sd.rec(
                int(SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            y = audio.flatten()
            sf.write("temp_audio.wav", y, SAMPLE_RATE)

            # Extract features
            feature = extract_features("temp_audio.wav")
            if feature is None:
                continue

            feature = feature.reshape(1, -1)

            # Prediction
            pred = elm.predict_class(feature, threshold=threshold)[0]

            # Dominant frequency
            dom_freq = get_dominant_frequency(y, SAMPLE_RATE)
            freq_history.append(dom_freq)
            freq_history = freq_history[-5:]

            sweep = max(freq_history) - min(freq_history)

            # Output
            if pred == 1:
                print(f"🚑 SIREN | Freq: {dom_freq:.1f} Hz | Sweep: {sweep:.1f}")
            else:
                print(f"🔇 NO SIREN | Freq: {dom_freq:.1f} Hz")

            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n🛑 Detection stopped")
            break

# -------------------- MAIN --------------------
if __name__ == "__main__":

    # 🔴 RUN ONCE FOR TRAINING
    # train_model("dataset_labels.csv")

    # 🟢 LOAD TRAINED MODEL
    elm = load_model()

    # 🔵 START REAL-TIME DETECTION
    continuous_detection(elm, threshold=0.4)
