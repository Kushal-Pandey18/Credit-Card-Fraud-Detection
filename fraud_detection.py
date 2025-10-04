# fraud_detection.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

MODEL_FILE = "fraud_detection_model.pkl"
THRESHOLD_FILE = "optimal_threshold.txt"

# -------------------------------
# Check if model already exists
# -------------------------------
if os.path.exists(MODEL_FILE) and os.path.exists(THRESHOLD_FILE):
    model = joblib.load(MODEL_FILE)
    with open(THRESHOLD_FILE, "r") as f:
        optimal_threshold = float(f.read())
    print("Model and threshold loaded from disk.")
else:
    print("Training model as no existing model found...")

    # -------------------------------
    # Load & preprocess data
    # -------------------------------
    df = pd.read_csv("fraud_data.csv")
    df.drop_duplicates(inplace=True)

    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df['hour'] = df['Time'].dt.hour
    df['day'] = df['Time'].dt.day
    df['month'] = df['Time'].dt.month
    df.drop(columns=['Time'], inplace=True)

    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

    # Features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # -------------------------------
    # Hyperparameter tuning
    # -------------------------------
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42),
        param_grid, cv=3, scoring='roc_auc'
    )
    grid_search.fit(X_train_res, y_train_res)

    model = grid_search.best_estimator_
    model.fit(X_train_res, y_train_res)

    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved as {MODEL_FILE}")

    # Evaluate model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    with open(THRESHOLD_FILE, "w") as f:
        f.write(str(optimal_threshold))

    y_pred = (y_pred_proba > optimal_threshold).astype(int)
    print("Best Model Parameters:", grid_search.best_params_)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))

    # Optional: ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc_score(y_test, y_pred_proba):.4f}")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("roc_curve.png")  # optional, for later reference
    plt.close()

# -------------------------------
# Flask API endpoint
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df_input = pd.DataFrame([data])
        # Preprocess Amount if present
        if 'Amount' in df_input.columns:
            df_input['Amount'] = StandardScaler().fit_transform(df_input[['Amount']])
        pred_proba = model.predict_proba(df_input)[:, 1][0]
        pred_class = int(pred_proba > optimal_threshold)
        return jsonify({"prediction": pred_class, "probability": float(pred_proba)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
