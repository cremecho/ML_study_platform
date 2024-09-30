import numpy as np
import pandas as pd

class NaiveBayes(object):
    def __init__(self, lam = 1):
        self.x_yc = {}
        self.p_y = {}
        self.lam = lam

    def pre_pruning(self, data):
        data_df = data
        data_df.loc[data_df['Temperature'] < 75, 'Temperature'] = 0
        data_df.loc[data_df['Temperature'] >= 75, 'Temperature'] = 1


        data_df.loc[data_df['Humidity'] < 80, 'Humidity'] = 0
        data_df.loc[data_df['Humidity'] >= 80, 'Humidity'] = 1
        return data_df

    # 然后计算每个属性的概率
    def compute_p_x_yc(self, df):
        gb = df.groupby(df.columns[-1])
        for col in df.columns:
            attr = gb[col]
            if attr == df.columns[-1]:
                continue
            value_counts = attr.value_counts()
            label_attr_pairs = value_counts.index
            attr_n = [0] * len(self.p_y)
            for pair in label_attr_pairs:
                self.x_yc[pair] = value_counts[pair]
                attr_n[pair[0]] += value_counts[pair]
            for label_attr in self.x_yc.keys():
                label = label_attr[0]
                n = attr_n[label]
                self.x_yc[label_attr] /= n

    # 先计算每一个label的概率
    def compute_p_y(self, df):
        gb = df.groupby(df.columns[-1])
        N = df.shape[0]
        self.N = N
        for label, indices in gb.groups.items():
            p_label = len(indices) / N
            self.p_y[label] = p_label


    def fit(self, X, y):
        X['Label'] = y
        self.compute_p_y(X)
        self.compute_p_x_yc(X)


    def predict(self, X):
        y_hat = []
        for row in X.iter_rows():
            prob = {}
            K = len(self.p_y)
            for label in self.p_y.keys():
                py = self.p_y[label]
                prob[label] = py
                for attr in X.columns:
                    label_attr_pair = (label, attr)
                    if not label_attr_pair in self.x_yc:
                        prob = 0
                    else:
                        prob = self.x_yc[label_attr_pair]
                    prob[label] *= prob
                # laplace smoothing
                prob[label] = (prob[label] + self.lam) / (self.N + K * self.lam)
            ans = max(prob, key=lambda x: prob[x])
            y_hat.append(ans)
        return y_hat

if __name__ == '__main__':
    root_train = r"../dataset\golf\golf_dataset_the_origin.csv"
    root_test = r"../dataset\golf\golf_dataset_mini\golf_dataset_mini_original_with_testset.csv"

    nbc = NaiveBayes()

    df = pd.read_csv(root_train)
    df = nbc.pre_pruning(df)
    X_train, y_train = df.iloc[:,:-1], df.iloc[:,-1]


    nbc.fit(X_train, y_train)
    print(df.value_counts())
    ...