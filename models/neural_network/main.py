import pandas as pd
from data_preprocessing import preprocess_data, prepare_data_loaders
from model import ChemicalYieldPredictor
from train import train_model
from evaluate import evaluate_model, plot_results
from utils import save_model


def main():
    df = pd.read_csv("df_cleared.csv")
    X, y, scaler, label_encoders = preprocess_data(df)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader = (
        prepare_data_loaders(X, y)
    )
    input_size = X_train_tensor.shape[1]
    model = ChemicalYieldPredictor(input_size)
    train_losses = train_model(model, train_loader, epochs=300, learning_rate=0.01)
    save_model(
        model,
        "trained_model.pth",
    )
    y_pred, metrics = evaluate_model(model, X_test_tensor, y_test_tensor)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    plot_results(train_losses)


if __name__ == "__main__":
    main()
