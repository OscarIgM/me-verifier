# scripts/train.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import json

embeddings_path = 'models/embeddings.npy'
labels_path = 'models/labels.csv'

X = np.load(embeddings_path)
df_labels = pd.read_csv(labels_path)
y = df_labels['label'].apply(lambda x: 1 if x == 'me' else 0).values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_val_scaled)
y_prob = clf.predict_proba(X_val_scaled)[:, 1]

accuracy = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)
report = classification_report(y_val, y_pred, target_names=['not_me','me'], output_dict=True)

print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
print("Classification report:", report)

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

joblib.dump(clf, 'models/model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

metrics = {
    "accuracy": accuracy,
    "auc": auc,
    "classification_report": report
}

with open('reports/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Modelo, scaler y métricas guardados en 'models/' y 'reports/'")
