from dataset._common_Dataset_attributes import *

class EtpDataset(ClassificationDataset):
    def __init__(self, mode, dataset_root, NUM_CLASSES, CLASS_LABELS):
        super().__init__(mode, dataset_root, NUM_CLASSES, CLASS_LABELS)



    def __getitem__(self, index):
        return super().__getitem__(index)
