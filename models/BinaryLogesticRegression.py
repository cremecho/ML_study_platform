import torch.nn as nn


class BinaryLogisticRegression(nn.Module):
    def __init__(self):
        super(BinaryLogisticRegression, self).__init__()
        self.linear = nn.Linear(2,1) # == torch.mm(X,w)+b
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return self.sigmoid(self.linear(x))