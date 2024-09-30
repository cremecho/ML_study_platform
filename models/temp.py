# import pandas as pd
# import numpy as np
# from scipy.stats import entropy
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.tree import export_text
#
# # root = '../dataset/golf/golf_dataset_mini/golf_dataset_mini_original_numerical_with_testset.csv'
# # df = pd.read_csv(root)
# # X,y = df.iloc[:,:-1], df.iloc[:,-1]
# #
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.33, random_state=42)
# #
# # tree = DecisionTreeRegressor()
# # tree.fit(X_train, y_train)
# # y_hat = tree.predict(X_test)
# #
# # y = pd.DataFrame([y_test, pd.Series(y_hat, name="Predict")])
# # y = y.transpose()
# # print(y_test.values)
# # print(y_hat)
# #
# # loss = sum(pow((y_test.values - y_hat), 2)) / len(y_hat)
# # print("MSE: %.3f" % loss)
#
# from sklearn.datasets import fetch_openml
# mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
#
# # 查看测试器和标签
# X, y = mnist['data'], mnist['target']
# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# # 对数据进行洗牌，防止输入许多相似实例导致的执行性能不佳
# shuffle_index = np.random.permutation(60000)
# X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
#
# def add_sum_as_feature(X):
#     # 计算每个实例的784个元素的和，得到形状为 (60000, 1) 或 (10000, 1) 的数组
#     sums = X.sum(axis=1, keepdims=True)
#     # 将 sums 数组与原始的 X 拼接，得到形状为 (60000, 785) 或 (10000, 785) 的新数组
#     X_with_sums = np.hstack((X, sums))
#     return X_with_sums
#
# # 对 X_train 和 X_test 进行处理
# X_train = add_sum_as_feature(X_train)
# X_test = add_sum_as_feature(X_test)
#
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# rfc = DecisionTreeClassifier()
# rfc.fit(X_train, y_train)
# text = export_text(rfc)
# print(text)
#
# y_hat = rfc.predict(X_test)
# test = [1 if a==b else 0 for a,b in zip(y_hat, y_test)]
# print('Acc = %.3f' % (sum(test) / len(test)))
# ...

import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# 加载MNIST数据集
mnist = datasets.fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target


# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 使用Bagging SVM进行训练和预测，并测量时间
start_time = time.time()
bagging_svm_clf = BaggingClassifier(base_estimator=SVC(kernel='rbf', C=1, gamma=0.05),
                                    n_estimators=10, random_state=42)
bagging_svm_clf.fit(X_train_scaled, y_train)
bagging_svm_training_time = time.time() - start_time
y_pred_bagging_svm = bagging_svm_clf.predict(X_test_scaled)
accuracy_bagging_svm = accuracy_score(y_test, y_pred_bagging_svm)

# 输出结果

print(f"Bagging SVM Accuracy: {accuracy_bagging_svm:.4f}")
print(f"Bagging SVM Training Time: {bagging_svm_training_time:.2f} seconds")