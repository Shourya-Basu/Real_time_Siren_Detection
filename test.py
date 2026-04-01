import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("dataset_labels.csv")

print("Total samples:", len(df))
print("Siren samples:", (df["label"] == 1).sum())
print("Non-siren samples:", (df["label"] == 0).sum())

X = []
y = []

for _, row in df.iterrows():
    features = extract_features(row["filepath"])
    X.append(features)
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)
