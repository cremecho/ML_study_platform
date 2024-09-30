import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


import torch
import torch.nn.functional as F


class BasicLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BasicLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        # 权重和偏置初始化
        self.W_fh = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_fx = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))

        self.W_ih = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_ix = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))

        self.W_ch = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_cx = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))

        self.W_oh = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_ox = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_prev, c_prev):
        # 拼接输入x和前一隐藏状态h_prev
        combined = torch.cat((x, h_prev), dim=1)

        # 遗忘门
        f = torch.sigmoid(F.linear(x, self.W_fx, self.b_f) + F.linear(h_prev, self.W_fh))

        # 输入门
        i = torch.sigmoid(F.linear(x, self.W_ix, self.b_i) + F.linear(h_prev, self.W_ih))

        # 候选记忆单元
        c = torch.tanh(F.linear(x, self.W_cx, self.b_c) + F.linear(h_prev, self.W_ch))

        # 输出门
        o = torch.sigmoid(F.linear(x, self.W_ox, self.b_o) + F.linear(h_prev, self.W_oh))

        # 更新记忆单元
        c_new = f * c_prev + i * c
        # 更新隐藏状态
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class BasicLSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = BasicLSTMCell(input_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        for t in range(x.size(1)):
            h, c = self.lstm_cell(x[:, t, :], h, c)

        out = self.fc(h)
        return out


# 示例用法
input_size = 20
hidden_size = 20
output_size = 1

model = BasicLSTMModel(input_size, hidden_size, output_size)
x = torch.randn(5, 1, input_size)  # (batch_size, sequence_length, input_size)
output = model(x)
print(output)
