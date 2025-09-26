import torch


def save_model(model, path="trained_model.pth"):
    torch.save(model.state_dict(), path)
