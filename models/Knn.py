import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys

# As Knn isn't Unsupervised Model, the calling line should be modified to:
# output = self.model(x, y)
class Knn(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.l1 = nn.Linear(2,2)
        self.K = K

    def euclid_distance(self, p1, p2):
        d = sum((p1 - p2).pow(2))
        return d

    def forward(self, x, y):
        # x.shape [400,2]
        n, features = x.shape
        k_distances = 999 * torch.ones((n, self.K))
        k_labels = torch.zeros((n, self.K))
        output = torch.zeros((n,))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                point1 = x[i, :]
                point2 = x[j, :]
                distance = self.euclid_distance(point1, point2)
                if distance < k_distances[i, -1]:
                    k_distances[i, -1] = distance
                    k_labels[i, -1] = y[j]
                    st = k_distances[i, :].sort()
                    k_distances[i, :] = st.values
                    k_labels[i, :] = k_labels[i, st.indices]
        mask = k_labels.sum(dim=1) > (self.K) / 2
        output[mask] = 1

        x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
        output = output.numpy()
        acc = np.sum(output == y) / len(y)
        print("Acc: %.3f" % acc)

        # org plot
        plt.figure(1)
        for i in range(len(y)):
            if y[i] == 0:
                plt.scatter(x[i,0], x[i,1], c='r', marker="*")
            else:
                plt.scatter(x[i, 0], x[i, 1],c='b', marker="^")
        plt.show()
        plt.close()

        plt.figure(2)
        for i in range(len(y)):
            if output[i] == 0:
                plt.scatter(x[i,0], x[i,1], c='r', marker="*")
            else:
                plt.scatter(x[i, 0], x[i, 1],c='b', marker="^")
        plt.show()
        plt.close()

        sys.exit()

        output.cuda()
        return output