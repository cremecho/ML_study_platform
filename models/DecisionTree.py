import pandas as pd
import numpy as np
from scipy.stats import entropy
import random
from sklearn.tree import DecisionTreeClassifier

class Node(object):
    def __init__(self, value, leaves=None, leaves_value=None):
        self.attribute_name = value
        self.leaves = leaves
        self.leaves_attribute_value = leaves_value


class DecisionTree(object):
    def __init__(self, data=None, max_depth=10, method='c45', min_samples_split=2):
        self.tree = None
        self.max_depth = max_depth
        self.method = method
        self.min_samples_split = min_samples_split
        if data is not None:
            data = self.pre_pruning(data)
            self.build_tree(data, depth=0)

    def pre_pruning(self, data):
        data_df = data.copy()
        data_df.loc[data_df['Temperature'] < 70, 'Temperature'] = 0
        data_df.loc[data_df['Temperature'] > 80, 'Temperature'] = 2
        data_df.loc[(data_df['Temperature'] <= 80) & (data_df['Temperature'] >= 70), 'Temperature'] = 1

        data_df.loc[data_df['Humidity'] <= 80, 'Humidity'] = 0
        data_df.loc[data_df['Humidity'] > 80, 'Humidity'] = 1
        return data_df

    def build_tree(self, data, depth):
        # 提前终止条件：树的深度达到最大深度
        if depth >= self.max_depth:
            most_common_label = data.iloc[:, -1].mode()[0]
            return Node(most_common_label)

        # 提前终止条件：节点中的数据全部属于同一类
        if len(data.iloc[:, -1].unique()) == 1:
            return Node(data.iloc[0, -1])

        # 提前终止条件：节点中的数据数量小于最小分裂样本数
        if len(data) < self.min_samples_split:
            most_common_label = data.iloc[:, -1].mode()[0]
            return Node(most_common_label)

        l = data.shape[1]
        if l == 1:
            return Node(data.values[0])

        attribute_names, label_name = data.columns.values[:-1], data.columns.values[-1]
        attributes_entropy = []
        for name in attribute_names:
            gb = data.groupby(name)[label_name]
            subset = []
            for attribute_value, index in gb.groups.items():
                try:
                    subset.append(data.loc[index][label_name].array)
                except:
                    ...
            etp = self.compute_entropy(subset)
            attributes_entropy.append([name, etp])
        attributes_entropy.sort(key=lambda x: x[1])

        name, entropy_value = attributes_entropy[0]
        cur_node = Node(name)
        gb = data.groupby(name)
        cur_node.leaves = []
        cur_node.leaves_attribute_value = []
        for attribute_value, index in gb.groups.items():
            leaf_data = data.loc[index].drop(columns=name)
            cur_node.leaves.append(self.build_tree(leaf_data, depth + 1))
            cur_node.leaves_attribute_value.append(attribute_value)
        self.tree = cur_node
        return cur_node

    def predict(self, x):
        def predict_helper(row, node):
            if len(row) == 0:
                # last leaf is the value of predict not name
                return node.attribute_name
            divide_attribute_name = node.attribute_name
            divide_value = row.get(divide_attribute_name)
            try:
                idx = node.leaves_attribute_value.index(divide_value)
            except:
                return np.array([-1])
            sub_row = row.drop(divide_attribute_name)
            sub_node = node.leaves[idx]
            predict = predict_helper(sub_row, sub_node)
            return predict

        y_hat = []
        for index, row in x.iterrows():
            predict = predict_helper(row, self.tree)

            if len(predict) != 1:
                predict = np.random.choice(predict.squeeze(), 1)
            predict = predict.squeeze()
            y_hat.append(predict)
        return y_hat

    def compute_entropy(self, subset):
        if self.method == 'id3':
            etp = 0.
            for s in subset:
                counts = s.value_counts()
                probability = [freq / counts.sum() for freq in counts.array]
                etp += entropy(probability)
            return etp


if __name__ == '__main__':
    root = r'../dataset/golf/golf_dataset_the_origin.csv'
    data_df = pd.read_csv(root)

    tree = DecisionTree(data_df, method='id3')

    # --- Testing ---
    root_test = r'../dataset/golf/golf_dataset_mini/golf_dataset_mini_original_with_testset.csv'
    test_df = pd.read_csv(root_test)
    one_hot_columns = test_df.iloc[:, 0:3]
    # 定义映射字典，将列名索引映射为原来的分类
    mapping = {'Outlook_sunny': 'sunny', 'Outlook_overcast': 'overcast', 'Outlook_rain': 'rain'}
    # 使用idxmax找到每一行中最大值的索引，并使用映射字典将索引转换为分类
    test_df['Outlook'] = one_hot_columns.idxmax(axis=1).map(mapping)
    test_df.drop(test_df.columns[0:3], axis=1, inplace=True)
    # 将Outlook列放到最前面
    cols = ['Outlook'] + [col for col in test_df if col != 'Outlook']
    test_df = test_df[cols]
    # pre-pruning
    test_df = tree.pre_pruning(test_df)
    x, y = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    y_hat = tree.predict(x)
    y = pd.DataFrame([y, pd.Series(y_hat, name="Predict")])
    y = y.transpose()

    print(y)
    correct_predict = y.loc[y['Play'] == y['Predict']]
    print("Acc: %.3f" % (len(correct_predict) / len(y)))

if __name__ == '__main__':
    root = r'../dataset/golf/golf_dataset_the_origin.csv'
    data_df = pd.read_csv(root)
    onehot = data_df.loc[:,'Outlook']
    dummy = pd.get_dummies(onehot, prefix='Outlook', dtype=float)
    data_df = data_df.drop("Outlook", axis=1)
    data_df = pd.concat([dummy, data_df], axis=1)
    data_df = data_df.iloc[:, [2,0,1,3,4,5,6]]

    root_test = r'../dataset/golf/golf_dataset_mini/golf_dataset_mini_original_with_testset.csv'
    test_df = pd.read_csv(root_test)

    x,y = data_df.iloc[:, :-1], data_df.iloc[:, -1]
    tree_sklearn = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    tree_sklearn.fit(x,y)
    test_x, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    y_hat = tree_sklearn.predict(test_x)
    y = pd.DataFrame([test_y, y_hat],index=['Play', 'Predict'])
    y = y.transpose()
    print('---')
    print(y)
    correct_predict = y.loc[y['Play'] == y['Predict']]
    print("Acc: %.3f" % (len(correct_predict) / len(y)))