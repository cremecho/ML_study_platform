import torch
from torch import nn

def complete_settings(configs):
    if 'model_name' not in configs or 'save_path' not in configs or 'dataset_root' not in configs:
        raise ValueError("necessary config settings were missing")
    if "batch_size" not in configs:
        configs["batch_size"] = 32
    if "epoch" not in configs:
        configs["epoch"] = 50
    if "lr" not in configs:
        configs["lr"] = 1e-2
    if "val" not in configs:
        configs["val"] = True
    if "scale" not in configs:
        configs["scale"] = 1


def get_model(trainer):
    if trainer.model_name == 'lr':
        from models.BinaryLogesticRegression import BinaryLogisticRegression
        return BinaryLogisticRegression()


def get_criterion(trainer):
    if trainer.model_name == 'lr':
        return nn.BCELoss()
    elif ...:
        ...


def get_optimizer(trainer):
    if trainer.model_name == 'lr':
        return torch.optim.SGD(trainer.model.parameters(), lr=trainer.lr)
    elif ...:
        ...