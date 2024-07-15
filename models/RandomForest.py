import numpy as np
from sklearn.tree import DecisionTreeRegressor

class RFR:
    """
    随机森林回归器
    """

    def __init__(self, n_estimators = 100, random_state = 0):
        # 随机森林的大小
        self.n_estimators = n_estimators
        # 随机森林的随机种子
        self.random_state = random_state

    def fit(self, X, y):
        dts = []
        n = X.shape[0]
        rs = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            dt = DecisionTreeRegressor(random_state=rs.randint(np.iinfo(np.int32).max), max_features = "auto")
            dt.fit(X, y, sample_weight=np.bincount(rs.randint(0, n, n), minlength = n))
            dts.append(dt)
        self.trees = dts

    def predict(self, X):

        ys = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            dt = self.trees[i]
            ys[i] = dt.predict(X)
        pred = np.argmax(ys)
        return pred

from sklearn.ensemble import RandomForestRegressor
import pandas as pd

rfr = RandomForestRegressor()
root = r"../dataset\golf\golf_dataset_mini\golf_dataset_mini_original_numerical_with_testset.csv"
df = pd.read_csv(root)
train, test = df.iloc[:-10,:], df.iloc[-10:,:]
X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]
X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]

rfr.fit(X_train, y_train)
y_hat = rfr.predict(X_test)


ls = [list(y_test), list(y_hat)]
print(ls[0])
print(ls[1])
mse = sum([(y1-y2)**2 for y1,y2 in zip(y_test, y_hat)]) / len(ls[0])
print("MSE: %.3f" % mse)

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
y_hat = gbr.predict(X_test)

ls = [list(y_test), list(y_hat)]
print(ls[0])
print(ls[1])
mse = sum([(y1-y2)**2 for y1,y2 in zip(y_test, y_hat)]) / len(ls[0])
print("MSE: %.3f" % mse)