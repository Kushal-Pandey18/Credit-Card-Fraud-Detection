import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("fraud_data.csv")

df.drop_duplicates(inplace=True)

df['Time'] = pd.to_datetime(df['Time'], unit='s')
df['hour'] = df['Time'].dt.hour
df['day'] = df['Time'].dt.day
df['month'] = df['Time'].dt.month
df.drop(columns=['Time'], inplace=True)  


df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

# Define features and target
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_
best_model.fit(X_train_resampled, y_train_resampled)

joblib.dump(best_model, "fraud_detection_model.pkl")
print("Model saved as fraud_detection_model.pkl")

# Model evaluation
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold}")

with open("optimal_threshold.txt", "w") as f:
    f.write(str(optimal_threshold))

y_pred = (y_pred_proba > optimal_threshold).astype(int)

print("Best Model Parameters:", grid_search.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))

# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc_score(y_test, y_pred_proba):.4f}")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()
