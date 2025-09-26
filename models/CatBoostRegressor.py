import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def objective(trial):
    iterations = trial.suggest_int("iterations", 100, 1000)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    depth = trial.suggest_int("depth", 4, 10)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
    random_strength = trial.suggest_float("random_strength", 0.1, 10)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", 1, 10)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        leaf_estimation_iterations=leaf_estimation_iterations,
        random_state=42,
        verbose=False,
        thread_count=-1,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features_indices,
        eval_set=(X_test, y_test),
        verbose=False,
    )
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    trial.set_user_attr("mae", mae)
    trial.set_user_attr("r2", r2)
    return mse


df = pd.read_csv("df_cleared.csv")
X = df.drop("Product_Yield_PCT_Area_UV", axis=1)
y = df["Product_Yield_PCT_Area_UV"]
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
for col in cat_features:
    X[col] = X[col].fillna("missing")
    X[col] = X[col].astype(str)
cat_features_indices = [X.columns.get_loc(col) for col in cat_features]
for col in cat_features:
    print(f"Признак {col}: NaN значений - {X[col].isnull().sum()}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best_trial = study.best_trial
print("Best parameters:", best_trial.params)
print("Best MSE:", best_trial.value)
print("Best MAE:", best_trial.user_attrs["mae"])
print("Best R2:", best_trial.user_attrs["r2"])
