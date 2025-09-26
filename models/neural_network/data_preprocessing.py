import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df, target_column="Product_Yield_PCT_Area_UV"):
    data = df.copy()
    y = data[target_column].values
    X = data.drop(
        [target_column]
        + [col for col in data.columns if "SMILES" in col or "ID" in col],
        axis=1,
        errors="ignore",
    )
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    X = X.fillna(X.mean(numeric_only=True))
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    return X.values, y, scaler, label_encoders


def prepare_data_loaders(X, y, test_size=0.2, batch_size=32, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader
