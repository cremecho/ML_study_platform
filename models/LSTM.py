import torch.nn as nn
import numpy as np
import torch

path1 = r''
path2 = r''



test_input = torch.randint(90, 105, (1, 100, 24), dtype=torch.float)
test_output = torch.randint(0,3, (1,100,))

lstm = nn.LSTM(input_size=24, hidden_size=48, proj_size = 3, batch_first=True)
criteria = nn.CrossEntropyLoss()
opt = torch.optim.SGD(lstm.parameters(), lr=1e-3)

for i in range(1000):
    model_output, (hn, cn) = lstm(test_input)
    model_output = model_output.transpose(1,2)
    loss = criteria(model_output, test_output)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(loss)
