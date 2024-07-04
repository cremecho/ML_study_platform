import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


class Node(object):
    def __init__(self, value, label, left=None, right=None):
        self.val = value
        self.label = label
        self.left = left
        self.right = right



class KDTree(object):
    def __init__(self, K, K_knn):
        super().__init__()
        self.l1 = nn.Linear(2,2)
        self.K = K
        self.K_knn = K_knn
        self.tree = None
        self.k_near_dis = np.ones((K_knn)) * 999
        self.k_near_point = np.zeros((K_knn, K))
        self.k_near_label = np.zeros((K_knn))

    # 建树
    def build_tree(self, x, xy, depth):
        l = len(x)
        if l == 0:
            return None
        split = depth % self.K  # 当前分割维度
        sorted_data = sorted(x, key=lambda x: x[split])  # 按照某个维度数据进行排序
        sorted_label = sorted(xy, key=lambda x: x[split])

        mid_idx = l // 2
        left_data = sorted_data[:mid_idx]
        right_data = sorted_data[mid_idx+1:]  # mid_idx已经作为当前层根节点了，所以不能加mid_idx这个数据对
        left_label = sorted_label[:mid_idx]
        right_label = sorted_label[mid_idx+1:]

        cur_node = Node(sorted_data[mid_idx], sorted_label[mid_idx][-1])  # 将中位数作为当前根节点
        cur_node.left = self.build_tree(left_data, left_label, depth + 1)  # 递归构建当前根节点左子树
        cur_node.right = self.build_tree(right_data, right_label, depth + 1)  # 递归构建当前根节点右子树

        self.tree = cur_node
        return cur_node

    # 计算欧氏距离
    def euclid_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # 最近邻搜索（1个target）
    def search_nn(self, tree, target):
        def dfs(node, depth):
            if not node:
                return

            # 第一步要先找到叶子节点，之后再进行回溯
            split = depth % self.K
            if target[split] < node.val[split]:
                dfs(node.left, depth + 1)  # 该行包括下面的dfs行，第一个参数输入都是某个节点node.left或node.right，所以本质上该节点node并没有改变，每次改变的都是函数输入参数，所以当dfs递归结束并回退到该位置时，该节点没有变化，以此实现回溯上一个节点的过程！
            else:
                dfs(node.right, depth + 1)
            # ======= 到这结束就已经找到了某个路径下最终叶子节点 =======

            # 开始回溯，以当前叶子节点和目标节点的距离作为初始的最小距离
            dis = self.euclid_distance(node.val, target)
            if not node.val in self.k_near_point and dis < self.k_near_dis[-1]:
                self.k_near_dis[-1] = dis
                self.k_near_label[-1] = node.label
                self.k_near_point[-1,:] = node.val

                idx = np.argsort(self.k_near_dis)
                self.k_near_dis = self.k_near_dis[idx]
                self.k_near_label = self.k_near_label[idx]
                self.k_near_point = self.k_near_point[idx]
                # label

            # 判断是否遍历该节点另一边子树
            if abs(node.val[split] - target[split]) < self.k_near_dis[-1]:
                if target[split] < node.val[split]:  # 第一次从上往下遍历时目标节点值小于当前节点值时，就遍历左边；回溯时因为要遍历另一边，所以刚好相反！
                    dfs(node.right, depth + 1)
                else:
                    dfs(node.left, depth + 1)

        dfs(tree, 0)
        return1, return2, return3 = self.k_near_point, self.k_near_dis, self.k_near_label
        self.k_near_dis = np.ones((self.K_knn)) * 999
        self.k_near_point = np.zeros((self.K_knn, self.K))
        self.k_near_label = np.zeros((self.K_knn))

        return return1, return2, return3




if __name__ == "__main__":
    # dataset, target, K = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]], [1, 5], 2
    dataset_train = BCPDataset('train')
    dataset_test = BCPDataset('test')

    K = 2
    K_knn = 3

    tree = KDTree(K, K_knn)
    train_xy = [[a[0], a[1], b] for a,b in zip(dataset_train.x, dataset_train.y)]
    tree.build_tree(dataset_train.x, train_xy, 0)

    added_labels = set()
    confusion_matrix = [0,0,0,0]
    plt.figure(1)
    total_t = 0
    for target, label in zip(dataset_test.x, dataset_test.y):
        t1 = time.time()
        nearest_point, nearest_dis, nearest_label = tree.search_nn(tree.tree, target)
        delta_t = time.time() - t1
        total_t += delta_t
        pred_label = 1 if sum(nearest_label) > K_knn / 2 else 0

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