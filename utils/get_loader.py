from torch.utils import data
from torch.utils.data import DataLoader
import os
from os.path import join
import glob
import torch
import torchvision.transforms as tr






def get_dataset(model_name, root, mode):
    if model_name == 'lr':
        from dataset.BinaryClassificationPointsDataset import BCPDataset
        return BCPDataset(mode)


def get_loader(model_name, scale, root, batch_size, val, pin_memory, num_workers, mode):
    if mode == 'train':
        dataset = get_dataset(model_name, root, mode)
        if scale != 1.:
            size = int(scale * len(dataset))
            dataset, _ = data.random_split(dataset, [size, len(dataset) - size])
        if val:
            train_size = int(0.9 * len(dataset))
            train_set, val_set = data.random_split(dataset, [train_size, len(dataset) - train_size])
            image_size = 512
            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, pin_memory=pin_memory,
                                      num_workers=num_workers, shuffle=True)
            val_loader = DataLoader(dataset=val_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                                    shuffle=False)
            return train_loader, val_loader
        else:
            train_loader = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
                                      num_workers=num_workers, shuffle=True)
            return train_loader
    else:
        dataset = get_dataset(model_name, root, mode)
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                                 shuffle=False)
        return test_loader

