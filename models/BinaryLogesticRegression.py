# [【pytorch学习实战】第三篇：逻辑回归_3.pytorch实现逻辑回归-CSDN博客](https://blog.csdn.net/QLeelq/article/details/123375791)
import torch.nn as nn


class BinaryLogisticRegression(nn.Module):
    def __init__(self):
        super(BinaryLogisticRegression, self).__init__()
        self.linear = nn.Linear(2,1) # == torch.mm(X,w)+b
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return self.sigmoid(self.linear(x))