import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("fraud_data.csv")

df['Time'] = pd.to_datetime(df['Time'], unit='s')
df['hour'] = df['Time'].dt.hour
df['day'] = df['Time'].dt.day
df['month'] = df['Time'].dt.month
df.drop(columns=['Time'], inplace=True) 
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

X = df.drop(columns=['Class'])
y = df['Class']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = joblib.load("fraud_detection_model.pkl")  
y_pred_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

y_pred = (y_pred_proba > optimal_threshold).astype(int)

print(f"Optimal Threshold: {optimal_threshold}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()
