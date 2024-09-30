class AdaBoostClassifier_:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):
        sample_weight = np.ones(len(X)) / len(X)  # 初始化样本权重为 1/N
        for _ in range(self.n_estimators):
            dtc = DecisionTreeClassifierWithWeight().fit(X, y, sample_weight)  # 训练弱学习器
            alpha = 1/2 * np.log((1-dtc.best_err)/dtc.best_err)  # 权重系数
            y_pred = dtc.predict(X)
            sample_weight *= np.exp(-alpha*y_pred*y)  # 更新迭代样本权重
            sample_weight /= np.sum(sample_weight)  # 样本权重归一化
            self.estimators.append(dtc)
            self.alphas.append(alpha)
        return self

    def predict(self, X):
        y_pred = np.empty((len(X), self.n_estimators))  # 预测结果二维数组，其中每一列代表一个弱学习器的预测结果
        for i in range(self.n_estimators):
            y_pred[:, i] = self.estimators[i].predict(X)
        y_pred = y_pred * np.array(self.alphas)  # 将预测结果与训练权重乘积作为集成预测结果
        return 2*(np.sum(y_pred, axis=1)>0)-1  # 以0为阈值，判断并映射为-1和1

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred==y)