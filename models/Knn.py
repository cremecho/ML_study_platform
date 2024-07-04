import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys

import time
import random
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


class BCPDataset(object):
    def __init__(self, mode):
        if mode == 'train':
            set_seed(103)
        else:
            set_seed(27)
        x1 = torch.randn(2000) + 1.5  # 生成400个满足标准正态分布的数字，均值为“0”，方差为“1”
        x2 = torch.randn(2000) - 1.5

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

# As Knn isn't Unsupervised Model, the calling line should be modified to:
# output = self.model(x, y)
class Knn(object):
    def __init__(self,xy, K):
        super().__init__()
        self.l1 = nn.Linear(2,2)
        self.xy = xy
        self.K = K
        self.k_near_dis = np.ones((K)) * 999
        self.k_near_point = np.zeros((K, 2))
        self.k_near_label = np.zeros((K))

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

    def euclid_distance_np(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def predict(self, target):
        for i in range(len(self.xy)):
            cur_point = self.xy[i][:2]
            cur_label = self.xy[i][-1]
            dis = self.euclid_distance_np(target, cur_point)

            if dis < self.k_near_dis[-1]:
                self.k_near_dis[-1] = dis
                self.k_near_point[-1] = cur_point
                self.k_near_label[-1] = cur_label

                idx = np.argsort(self.k_near_dis)
                self.k_near_dis = self.k_near_dis[idx]
                self.k_near_label = self.k_near_label[idx]
                self.k_near_point = self.k_near_point[idx]

        return1, return2, return3 = self.k_near_point, self.k_near_dis, self.k_near_label
        self.k_near_dis = np.ones((self.K)) * 999
        self.k_near_point = np.zeros((self.K, 2))
        self.k_near_label = np.zeros((self.K))

        return return1, return2, return3



if __name__ == '__main__':
    dataset_train = BCPDataset('train')
    dataset_test = BCPDataset('test')

    train_xy = [[a[0], a[1], b] for a, b in zip(dataset_train.x, dataset_train.y)]
    K = 3
    knn = Knn(train_xy, K=K)

    added_labels = set()
    confusion_matrix = [0,0,0,0]
    plt.figure(1)
    total_t = 0.
    for target, label in zip(dataset_test.x, dataset_test.y):
        t1 = time.time()
        nearest_point, nearest_dis, nearest_label = knn.predict(target)
        delta_t = time.time() - t1
        total_t += delta_t
        pred_label = 1 if sum(nearest_label) > K / 2 else 0

        if label == 1 and pred_label == 1:
            confusion_matrix[0] = confusion_matrix[0] + 1
            scatter_label = 'TP' if 'TP' not in added_labels else None
            added_labels.add('TP')
            plt.scatter(target[0], target[1], c='b', label=scatter_label)
        elif label == 1 and pred_label == 0:
            confusion_matrix[1] = confusion_matrix[1] + 1
            scatter_label = 'FN' if 'FN' not in added_labels else None
            added_labels.add('FN')
            plt.scatter(target[0], target[1], c='g', label=scatter_label)
        elif label == 0 and pred_label == 1:
            confusion_matrix[2] = confusion_matrix[2] + 1
            scatter_label = 'FP' if 'FP' not in added_labels else None
            added_labels.add('FP')
            plt.scatter(target[0], target[1], c='orange', label=scatter_label)
        elif label == 0 and pred_label == 0:
            confusion_matrix[3] = confusion_matrix[3] + 1
            scatter_label = 'TN' if 'TN' not in added_labels else None
            added_labels.add('TN')
            plt.scatter(target[0], target[1], c='r', label=scatter_label)

    plt.legend()
    plt.title('KNN Classification Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    print('Acc: %.3f' % ((confusion_matrix[0]+ confusion_matrix[3]) / sum(confusion_matrix)))
    print('Time: %.3f' % total_t)