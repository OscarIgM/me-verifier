# evaluate.py
import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report
)

embeddings = np.load('models/embeddings.npy')
labels_df = pd.read_csv('models/labels.csv')
y_true = labels_df['label'].apply(lambda x: 1 if x == 'me' else 0).values

clf = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')

X_scaled = scaler.transform(embeddings)

y_prob = clf.predict_proba(X_scaled)[:, 1]

precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
fscore = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = fscore.argmax()
best_threshold = pr_thresholds[best_idx]

print(f"✅ Umbral óptimo τ: {best_threshold:.2f}")

# --- Predicciones con umbral ---
y_pred = (y_prob >= best_threshold).astype(int)

# --- Matriz de confusión ---
cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:\n", cm)

# --- Curvas ROC y PR ---
fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
os.makedirs('reports', exist_ok=True)
plt.savefig('reports/roc_curve.png')
plt.close()

plt.figure(figsize=(6,6))
plt.plot(recall, precision, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('reports/pr_curve.png')
plt.close()

# --- Guardar métricas y matriz de confusión ---
report = classification_report(y_true, y_pred, target_names=['not_me','me'], output_dict=True)

metrics = {
    "best_threshold": float(best_threshold),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "roc_auc": float(roc_auc)
}

with open('reports/evaluation.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluación completa guardada en reports/")
