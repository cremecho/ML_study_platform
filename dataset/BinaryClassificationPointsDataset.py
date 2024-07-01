import random
import numpy as np
import torch
import os
from torch.utils import data
from dataset._common_Dataset_attributes import *

  # any random number


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


class BCPDataset(ClassificationDataset):
    def __init__(self, mode, dataset_root, NUM_CLASSES, CLASS_LABELS):
        super().__init__(mode, dataset_root, NUM_CLASSES, CLASS_LABELS)
        if mode == 'train':
            set_seed(103)
        else:
            set_seed(27)
        x1 = torch.randn(400) + 1.5  # 生成400个满足标准正态分布的数字，均值为“0”，方差为“1”
        x2 = torch.randn(400) - 1.5

        data = zip(x1.data.numpy(), x2.data.numpy())  # 转numpy，组成元组

        pos = []
        neg = []

        def classification(data):
            for i in data:
                if (i[0] > 1.5 + 0.1 * torch.rand(1).item() * (-1) ** torch.randint(1, 10, (
                        1, 1)).item()):  # item获取元素值，按照1.5分为左右两边
                    pos.append(i)
                else:
                    neg.append(i)

        classification(data)

        # 数据：pos and neg
        inputs = [[i[0], i[1]] for i in pos]  # 数据维度2，由x1和x2组成
        inputs.extend([[i[0], i[1]] for i in neg])  # extend 接受一个参数，这个参数总是一个 list，并且把这个 list 中的每个元素添加到原 list 中
        # inputs = torch.Tensor(inputs)  # torch.Tensor 生成单精度浮点类型的张量

        # 标签，真值，1 and 0
        label = [1 for i in range(len(pos))]
        label.extend(0 for i in range(len(neg)))
        # label = torch.Tensor(label)
        self.x = inputs
        self.y = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        inputs = torch.Tensor(x)
        # label = torch.Tensor(y)
        # tt = tr.ToTensor()
        # inputs = tt(inputs)
        return {'x': inputs, 'y': y}
