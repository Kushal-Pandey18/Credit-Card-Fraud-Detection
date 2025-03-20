# Credit Card Fraud Detection

## Project Overview
This project builds a machine learning model to detect fraudulent credit card transactions efficiently while minimizing false positives. It includes data preprocessing, feature engineering, handling class imbalance, and training an optimized model.

## Dataset Information
Dataset

The dataset used for this project was obtained from Kaggle. You can download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset is highly imbalanced, requiring oversampling techniques.

## Project Structure
```
├── fraud_data.csv          # Dataset
├── preprocessing.py        # Data preprocessing & feature engineering
├── train_model.py          # Model training script
├── evaluate.py             # Model evaluation script
├── fraud_detection.py      # Main fraud detection script
├── requirements.txt        # Dependencies
├── README.txt              # Project documentation
```

## Installation and Setup
### Prerequisites
- Python 3.x
- Required libraries (listed in `requirements.txt`)

### Steps to Run
1. **Clone the repository**  
   ```sh
   git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```
2. **Install dependencies**  
   ```sh
   pip install -r requirements.txt
   ```
3. **Preprocess the data**  
   ```sh
   python preprocessing.py
   ```
4. **Train the model**  
   ```sh
   python train_model.py
   ```
5. **Evaluate the model**  
   ```sh
   python evaluate.py
   ```
6. **Run fraud detection**  
   ```sh
   python fraud_detection.py
   ```

## Feature Engineering
- **Time Features**: Extracted hour, day, and month.
- **Amount Normalization**: StandardScaler applied to 'Amount'.
- **Handling Class Imbalance**: Used SMOTE for synthetic data generation.

## Model Details
- **Algorithm**: XGBoost Classifier
- **Hyperparameter Tuning**: GridSearchCV optimized `learning_rate`, `max_depth`, and `n_estimators`.
- **Threshold Optimization**: ROC curve used to find the optimal classification threshold.

### Best Model Parameters:
```
{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 150}
```

## Evaluation Metrics
- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **AUC-ROC Score**
- **Optimal Threshold Selection**

Example Output:
```
Confusion Matrix:
 [[55524  1127]
 [   12    83]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.98      0.99     56651
           1       0.07      0.87      0.13        95

    accuracy                           0.98     56746
   macro avg       0.53      0.93      0.56     56746
weighted avg       1.00      0.98      0.99     56746

AUC-ROC Score: 0.969
```

## Expected Outcome
- A trained fraud detection model with high accuracy and minimal false positives.
- Clear documentation and structured code.
- A GitHub repository containing:
  - Preprocessing steps.
  - Model training and evaluation.
  - Fraud detection implementation.

## Contact
**Developer:** Kushal Pandey  
**Email:** kushal.pandey850@gmail.com
**GitHub:** https://github.com/Kushal-Pandey18

