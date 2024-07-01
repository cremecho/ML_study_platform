from torch.utils.data.dataset import T_co
from torchvision.datasets import MNIST
from torch.utils import data
import torch

class MnistDataset(data.Dataset):
    def __init__(self, trainer):
        train = True if trainer.mode == 'train' else False
        dataset = MNIST(root=trainer.dataset_root, train=train, transform=None, download=False)
        data = dataset.train_data
        label = dataset.train_labels
        # for lr, squeeze image to 1d
        self.data = data.view(-1, 28*28)
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'x': self.data[index], 'y': self.label[index]}