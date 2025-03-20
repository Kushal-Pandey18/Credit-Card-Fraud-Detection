import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(file_path):

    df = pd.read_csv(file_path)
    
    df.drop_duplicates(inplace=True)
    
    return df

def feature_engineering(df):
    
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df['hour'] = df['Time'].dt.hour
    df['day'] = df['Time'].dt.day
    df['month'] = df['Time'].dt.month
    df.drop(columns=['Time'], inplace=True)  

    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    return df

def handle_class_imbalance(X, y):
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

def preprocess_data(file_path):
    
    df = load_data(file_path)
    df = feature_engineering(df)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Handle class imbalance
    X_resampled, y_resampled = handle_class_imbalance(X, y)
    
    return X_resampled, y_resampled

if __name__ == "__main__":
    file_path = "fraud_data.csv"
    X, y = preprocess_data(file_path)
    print("Data preprocessing completed. Processed dataset ready for model training.")

