import optuna
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 1.0)
    gamma = trial.suggest_float("gamma", 0.0, 0.5)
    strategy = trial.suggest_categorical("strategy", ["mean", "median"])
    
    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        random_state=42,
        n_jobs=-1,
        enable_categorical=False,
    )
    model.fit(X_train_imputed, y_train)
    y_pred = model.predict(X_test_imputed)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    trial.set_user_attr("mae", mae)
    trial.set_user_attr("r2", r2)
    return mse


df = pd.read_csv("df_cleared.csv")
label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le
X = df.drop("Product_Yield_PCT_Area_UV", axis=1)
y = df["Product_Yield_PCT_Area_UV"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=150)
best_trial = study.best_trial
print("Best parameters:", best_trial.params)
print("Best MSE:", best_trial.value)
print("Best MAE:", best_trial.user_attrs["mae"])
print("Best R2:", best_trial.user_attrs["r2"])
