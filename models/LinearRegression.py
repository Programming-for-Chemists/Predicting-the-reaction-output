import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def objective(trial):
    normalize = trial.suggest_categorical("normalize", [True, False])
    strategy = trial.suggest_categorical("strategy", ["mean", "median"])

    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    if normalize:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
    else:
        X_train_scaled = X_train_imputed
        X_test_scaled = X_test_imputed
        
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
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
