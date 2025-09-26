import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
from torch import nn


def evaluate_model(model, X_test, y_test):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = criterion(predictions, y_test).item()
        r2 = r2_score(y_test.numpy(), predictions.numpy())
        mae = nn.L1Loss()(predictions, y_test).item()
        rmse = np.sqrt(mse)
    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    return predictions.numpy(), metrics


def plot_results(train_losses):
    plt.figure(figsize=(5, 4))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Learning Progess")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()
