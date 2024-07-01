from torch.utils import data


class ClassificationDataset(data.Dataset):
    def __init__(self, mode, dataset_root, NUM_CLASSES, CLASS_LABELS):
        self.mode = mode
        self.dataset_root = dataset_root
        self.NUM_CLASSES = NUM_CLASSES
        self.CLASS_LABELS = CLASS_LABELS
        if mode == 'train':
            ...
        else:
            ...

    def __getitem__(self, index):
        pass