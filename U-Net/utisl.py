import torch
import torch.nn as nn

import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_states(model, optimizer, epoch, loss, path):
    print("=> Saving model")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "epoch": epoch,
        },
        path,
    )
    print(f"=> Model saved at {path}")


def load_states(model, path, optimizer=None, loss=None, epoch=None):
    print("=> Loading model")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss
