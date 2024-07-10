import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader


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


def get_model_settings(trainer):
    if trainer.model_name == 'lr':
        # 1. get model
        from models.BinaryLogesticRegression import BinaryLogisticRegression
        _model = BinaryLogisticRegression()
        # 2. get dataset
        from dataset.BinaryClassificationPointsDataset import BCPDataset
        NUM_CLASSES = 2
        CLASS_LABELS = ['x1', 'x2']
        _dataset = BCPDataset(trainer.mode, trainer.dataset_root, NUM_CLASSES, CLASS_LABELS)
        # 3. set criterion
        _loss_func = nn.BCELoss()
        # 4. set optimizer, initializing in train_val since the model is not setting yet
        _optimizer = torch.optim.SGD
    elif trainer.model_name == 'multiclass_lr':
        # 1.
        from models.LogesticRegression import LogisticRegression
        _model = LogisticRegression(28*28, 10)
        # 2.
        from dataset.Mnist import MnistDataset
        NUM_CLASSES = 10
        CLASS_LABELS = ['0','1','2','3','4','5','6','7','8','9']
        _dataset = MnistDataset(trainer.mode, trainer.dataset_root, NUM_CLASSES, CLASS_LABELS)
        # 3.
        _loss_func = nn.CrossEntropyLoss()
        # 4.
        _optimizer = torch.optim.SGD
    elif trainer.model_name == 'knn':
        # for Knn and kd-tree Knn, run the
        raise NotImplementedError("for Knn and kd-tree Knn, run the main function in model implement script")
    else:
        raise NotImplementedError('model: ' + trainer.model_name + ' not implemented')
    return _model, _dataset, _loss_func, _optimizer


def get_loader(trainer, dataset, pin_memory, num_workers):
    if trainer.mode == 'train':
        if trainer.scale != 1.:
            size = int(trainer.scale * len(dataset))
            dataset, _ = data.random_split(dataset, [size, len(dataset) - size])
        if trainer.val:
            train_size = int(0.9 * len(dataset))
            train_set, val_set = data.random_split(dataset, [train_size, len(dataset) - train_size])
            image_size = 512
            train_loader = DataLoader(dataset=train_set, batch_size=trainer.batch_size, pin_memory=pin_memory,
                                      num_workers=num_workers, shuffle=True)
            val_loader = DataLoader(dataset=val_set, batch_size=trainer.batch_size, pin_memory=pin_memory, num_workers=num_workers,
                                    shuffle=False)
            return train_loader, val_loader
        else:
            train_loader = DataLoader(dataset=dataset, batch_size=trainer.batch_size, pin_memory=pin_memory,
                                      num_workers=num_workers, shuffle=True)
            return train_loader, None
    else:
        test_loader = DataLoader(dataset=dataset, batch_size=trainer.batch_size, pin_memory=pin_memory, num_workers=num_workers,
                                 shuffle=False)
        return test_loader

