import pandas as pd
import numpy as np
from scipy.stats import entropy
import random

class Node(object):
    def __init__(self, value, leaves = None, leaves_value = None):
        self.attribute_name = value
        self.leaves = leaves
        self.leaves_attribute_value = leaves_value
        # self.left = left
        # self.right = right


class DecisionTree(object):
    def __init__(self, data = None, max_depth = 10, method = 'c45'):
        self.tree = None
        self.max_depth = max_depth
        self.method = method
        if data is not None:
            data = self.pre_pruning(data)
            self.build_tree(data)

    def pre_pruning(self, data):
        data_df = data
        data_df.loc[data_df['Temperature'] < 70, 'Temperature'] = 0
        data_df.loc[data_df['Temperature'] > 80, 'Temperature'] = 2
        data_df.loc[(data_df['Temperature'] <= 80) & (data_df['Temperature'] >= 70), 'Temperature'] = 1

        data_df.loc[data_df['Humidity'] <= 80, 'Humidity'] = 0
        data_df.loc[data_df['Humidity'] > 80, 'Humidity'] = 1
        return data_df

    def post_pruning(self):
        ...

    def build_tree(self, data):
        # TODO:maybe is data.shape[2] == 1
        l = data.shape[1]
        if l == 1:
            return Node(data.values)
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
            entropy = self.compute_entropy(subset)
            attributes_entropy.append([name, entropy])
        attributes_entropy.sort(key=lambda x:x[1])

        name, entropy = attributes_entropy[0]
        cur_node = Node(name)
        gb = data.groupby(name)
        cur_node.leaves = []
        cur_node.leaves_attribute_value = []
        for attribute_value, index in gb.groups.items():
            leaf_data = data.loc[index].drop(columns=name)
            cur_node.leaves.append(self.build_tree(leaf_data))
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



import numpy as np
import pandas as pd

class Node:
    def __init__(self, attribute_name=None, value=None, leaves=None):
        self.attribute_name = attribute_name
        self.value = value
        self.leaves = leaves if leaves is not None else []
        self.leaves_attribute_value = []

class DecisionTree:
    def __init__(self, data=None, max_depth=10, method='cart'):
        self.tree = None
        self.max_depth = max_depth
        self.method = method
        if data is not None:
            self.build_tree(data)

    def build_tree(self, data, depth=0):
        if depth == self.max_depth or len(data.columns) == 1:
            return Node(value=data.iloc[:, -1].mean())

        attribute_names = data.columns.values[:-1]
        label_name = data.columns.values[-1]

        best_split = self.find_best_split(data, attribute_names, label_name)
        if best_split is None:
            return Node(value=data[label_name].mean())

        split_attribute, split_value = best_split
        left_subset = data[data[split_attribute] <= split_value]
        right_subset = data[data[split_attribute] > split_value]

        if left_subset.empty or right_subset.empty:
            return Node(value=data[label_name].mean())

        cur_node = Node(attribute_name=split_attribute, value=split_value)
        cur_node.leaves.append(self.build_tree(left_subset, depth + 1))
        cur_node.leaves.append(self.build_tree(right_subset, depth + 1))

        return cur_node

    def find_best_split(self, data, attribute_names, label_name):
        best_mse = float('inf')
        best_split = None

        for attribute in attribute_names:
            values = data[attribute].unique()
            for value in values:
                left_subset = data[data[attribute] <= value]
                right_subset = data[data[attribute] > value]
                mse = self.compute_mse(left_subset[label_name], right_subset[label_name])

                if mse < best_mse:
                    best_mse = mse
                    best_split = (attribute, value)

        return best_split

    def compute_mse(self, left, right):
        left_mse = ((left - left.mean()) ** 2).mean() if len(left) > 0 else 0
        right_mse = ((right - right.mean()) ** 2).mean() if len(right) > 0 else 0
        total_mse = left_mse * len(left) + right_mse * len(right)
        total_mse /= (len(left) + len(right))
        return total_mse

    def predict(self, x):
        def predict_helper(row, node):
            if node.leaves == []:
                return node.value

            if row[node.attribute_name] <= node.value:
                return predict_helper(row, node.leaves[0])
            else:
                return predict_helper(row, node.leaves[1])

        y_hat = []
        for index, row in x.iterrows():
            y_hat.append(predict_helper(row, self.tree))
        return y_hat

# 示例数据
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6],
    'feature2': [2, 3, 4, 5, 6, 7],
    'target': [1, 3, 2, 5, 4, 6]
})

# 构建并训练回归树
tree = DecisionTree(data=data, max_depth=3, method='cart')

# 预测
test_data = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [2, 3, 4]
})

predictions = tree.predict(test_data)
print(predictions)


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
    x, y = test_df.iloc[:,:-1], test_df.iloc[:,-1]
    y_hat = tree.predict(x)
    y = pd.DataFrame([y, pd.Series(y_hat,name="Predict")])
    y = y.transpose()

    print(y)
    correct_predict = y.loc[y['Play'] == y['Predict']]
    print("Acc: %.3f" % (len(correct_predict) / len(y)))