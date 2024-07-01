from torch.utils.data.dataset import T_co
from torchvision.datasets import MNIST
from torch.utils import data
import torch
from dataset._common_Dataset_attributes import *

class MnistDataset(ClassificationDataset):
    def __init__(self, mode, dataset_root, NUM_CLASSES, CLASS_LABELS):
        super().__init__(mode, dataset_root, NUM_CLASSES, CLASS_LABELS)
        train = True if mode == 'train' else False
        dataset = MNIST(root=dataset_root, train=train, transform=None, download=False)
        data = dataset.train_data
        label = dataset.train_labels
        # for lr, squeeze image to 1d
        self.data = data.view(-1, 28*28)
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'x': self.data[index], 'y': self.label[index]}