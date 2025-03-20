import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from preprocessing import preprocess_data

file_path = "fraud_data.csv"
X, y = preprocess_data(file_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42), 
    param_grid, 
    cv=3, 
    scoring='roc_auc'
)
grid_search.fit(X_train, y_train)

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Find optimal threshold using ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold}")

y_pred = (y_pred_proba > optimal_threshold).astype(int)

print("Best Model Parameters:", grid_search.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))

joblib.dump(best_model, "fraud_detection_model.pkl")
print("Model saved as 'fraud_detection_model.pkl'")
