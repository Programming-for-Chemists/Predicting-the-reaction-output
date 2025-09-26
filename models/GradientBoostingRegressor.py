import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    max_iter = trial.suggest_int("max_iter", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    l2_regularization = trial.suggest_float("l2_regularization", 0.0, 1.0)
    max_bins = trial.suggest_int("max_bins", 100, 255)
    
    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_bins=max_bins,
        random_state=42,
        early_stopping=False,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
    df[column] = le.fit_transform(df[column])
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
