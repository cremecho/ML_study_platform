# # -*- coding: utf-8 -*-
#
# import numpy as np
# import math
# import copy
# import matplotlib.pyplot as plt
# from joblib.numpy_pickle_utils import xrange
#
# isdebug = True
#
#
# # 参考文献：机器学习TomM.Mitchell P.137
# # 代码参考http://blog.csdn.net/chasdmeng/article/details/38709063
#
# # 指定k个高斯分布参数，这里指定k=2。注意2个高斯分布具有相同均方差Sigma，均值分别为Mu1,Mu2。
# def init_data(Sigma, Mu1, Mu2, k, N):
#     global X
#     global Mu
#     global Expectations
#     X = np.zeros((1, N))
#     Mu = np.random.random(k)
#     Expectations = np.zeros((N, k))
#     for i in xrange(0, N):
#         if np.random.random(1) > 0.5:
#             X[0, i] = np.random.normal(Mu1, Sigma)
#         else:
#             X[0, i] = np.random.normal(Mu2, Sigma)
#     if isdebug:
#         print("***********")
#         print("初始观测数据X：")
#         print(X)
#
#
# # EM算法：步骤1，计算E[zij]
# def e_step(Sigma, k, N):
#     global Expectations
#     global Mu
#     global X
#     for i in xrange(0, N):
#         Denom = 0
#         Numer = [0.0] * k
#         for j in xrange(0, k):
#             Numer[j] = math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)
#             Denom += Numer[j]
#         for j in xrange(0, k):
#             Expectations[i, j] = Numer[j] / Denom
#     if isdebug:
#         print("***********")
#         print("隐藏变量E（Z）：")
#         print(Expectations)
#
#
# # EM算法：步骤2，求最大化E[zij]的参数Mu
# def m_step(k, N):
#     global Expectations
#     global X
#     for j in xrange(0, k):
#         Numer = 0
#         Denom = 0
#         for i in xrange(0, N):
#             Numer += Expectations[i, j] * X[0, i]
#             Denom += Expectations[i, j]
#         Mu[j] = Numer / Denom
#
#
# # 算法迭代iter_num次，或达到精度Epsilon停止迭代
# def run(Sigma, Mu1, Mu2, k, N, iter_num, Epsilon):
#     init_data(Sigma, Mu1, Mu2, k, N)
#     print("初始<u1,u2>:", Mu)
#     for i in range(iter_num):
#         Old_Mu = copy.deepcopy(Mu)
#         e_step(Sigma, k, N)
#         m_step(k, N)
#         print(i, Mu)
#         if sum(abs(Mu - Old_Mu)) < Epsilon:
#             break
#
#
# if __name__ == '__main__':
#     sigma = 6  # 高斯分布具有相同的方差
#     mu1 = 40  # 第一个高斯分布的均值 用于产生样本
#     mu2 = 20  # 第二个高斯分布的均值 用于产生样本
#     k = 2  # 高斯分布的个数
#     N = 1000  # 样本个数
#     iter_num = 1000  # 最大迭代次数
#     epsilon = 0.0001  # 当两次误差小于这个时退出
#     run(sigma, mu1, mu2, k, N, iter_num, epsilon)
#
#     plt.hist(X[0, :], 50)
#     plt.show()
#
# # -*- coding: utf-8 -*-
# import math
#
# def cal_u(pi, p, q, xi):
#     """
#       u值计算
#     :param pi: 下一次迭代开始的 pi
#     :param p:  下一次迭代开始的 p
#     :param q:  下一次迭代开始的 q
#     :param xi: 观察数据第i个值，从0开始
#     :return:
#     """
#     return pi * math.pow(p, xi) * math.pow(1 - p, 1 - xi) / \
#            float(pi * math.pow(p, xi) * math.pow(1 - p, 1 - xi) +
#                  (1 - pi) * math.pow(q, xi) * math.pow(1 - q, 1 - xi))
#
# def e_step(pi,p,q,x):
#     """
#         e步计算
#     :param pi: 下一次迭代开始的 pi
#     :param p:  下一次迭代开始的 p
#     :param q:  下一次迭代开始的 q
#     :param x: 观察数据
#     :return:
#     """
#     return [cal_u(pi,p,q,xi) for xi in x]
#
# def m_step(u,x):
#     """
#      m步计算
#     :param u:  m步计算的u
#     :param x:  观察数据
#     :return:
#     """
#     pi1=sum(u)/len(u)
#     p1=sum([u[i]*x[i] for i in range(len(u))]) / sum(u)
#     q1=sum([(1-u[i])*x[i] for i in range(len(u))]) / sum([1-u[i] for i in range(len(u))])
#     return [pi1,p1,q1]
#
# def run(observed_x, start_pi, start_p, start_q, iter_num):
#     """
#
#     :param observed_x:  观察数据
#     :param start_pi:  下一次迭代开始的pi $\pi$
#     :param start_p:  下一次迭代开始的p
#     :param start_q:  下一次迭代开始的q
#     :param iter_num:  迭代次数
#     :return:
#     """
#     for i in range(iter_num):
#         u=e_step(start_pi, start_p, start_q, observed_x)
#         print (i,[start_pi,start_p,start_q])
#         if [start_pi,start_p,start_q]==m_step(u, observed_x):
#             break
#         else:
#             [start_pi,start_p,start_q]=m_step(u, observed_x)
# if __name__ =="__main__":
#     # 观察数据
#     x = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
#     # 初始化 pi，p q
#     [pi, p, q] = [0.7, 0.2, 0.7]
#     # 迭代计算
#     run(x,pi,p,q,100)


# --- my implementation ---
# GMM: unknown:
#   theta = {mu1, mu2, ..., sigma1, sigma2, ...}
#   Z = {pi1, pi2, ...}

import numpy as np
from joblib.numpy_pickle_utils import xrange
import math
import matplotlib.pyplot as plt
from numpy import seterr


class GMM1d_EM(object):
    def __init__(self, K, N, max_iter, epsilon):
        self.K = K
        self.N = N

        self.init_data()
        self.init_para()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def init_data(self):
        X = np.zeros((self.N))
        Mu = 10 * np.random.random(self.K)
        Sigma = np.random.random(self.K)
        print("GT mu:")
        print(Mu)
        print("GT sigma:")
        print(Sigma)
        for i in xrange(0, self.N):
            cls = np.random.randint(0,self.K)
            X[i] = np.random.normal(Mu[cls], Sigma[cls])
        self.X = X
        plt.hist(X,50)
        plt.show()

    def init_para(self):
        self.expect = np.zeros((self.N, self.K))
        from sklearn.cluster import KMeans
        km = KMeans(3)
        km.fit(self.X.reshape(-1,1))
        self.sigma = np.zeros((3,))
        group_counts = np.zeros((3,))
        self.mu = km.cluster_centers_.squeeze()
        for x,group in zip(self.X, km.labels_):
            self.sigma[group] += np.sqrt((x-self.mu[group])**2)
            group_counts[group] += 1
        self.sigma /= group_counts

    def E_step(self):
        left = 1 /( (np.sqrt(2 * np.pi) * self.sigma) +1e-10)
        x_minus_mu = np.array([self.X - mu for mu in self.mu])
        e_exp = -1 * (x_minus_mu**2).T / ( (2*(self.sigma**2))  +1e-10)
        right = np.power(np.e, e_exp)
        expect = left * right
        self.expect = expect.T / (np.sum(expect, axis=1)    +1e-10)


    def M_step(self):
        # update mu
        numer = np.dot(self.expect, self.X)
        denom = np.sum(self.expect, axis=1)

        if np.isnan(numer.any()) or np.isnan(denom.any()):
            ...
        self.mu = numer / denom


        # update sigma
        x_minus_mu = np.array([self.X - mu for mu in self.mu])
        numer = np.sum(x_minus_mu**2 * self.expect, axis=1)
        self.sigma = numer / denom


    def fit(self):
        for i in range(self.max_iter):
            curr_mu, curr_simga = self.mu, self.sigma
            self.E_step()
            self.M_step()
            if sum(abs(curr_mu - self.mu)) < self.epsilon and sum(abs(curr_simga - self.sigma)) < self.epsilon:
                break
        print("iter %d" % i)

if __name__ == '__main__':
    k = 3   # num of distribution
    N = 2000    # num of total points
    max_iter = 1000
    epsilon = 1e-3
    gmm = GMM1d_EM(k,N,max_iter,epsilon)
    print("init: mu")
    print(gmm.mu)
    print("init: sigma")
    print(gmm.sigma)
    gmm.fit()
    print("final: mu")
    print(gmm.mu)
    print("final: sigma")
    print(gmm.sigma)


    from sklearn.mixture import GaussianMixture
    gm_sk = GaussianMixture(3)
    y = gm_sk.fit(gmm.X.reshape(-1,1))
    print(y.means_.squeeze())
    print(y.covariances_.squeeze())