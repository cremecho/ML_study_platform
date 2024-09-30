import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln
from sklearn.cluster import kmeans_plusplus


class GMM1d_VarBayes(object):
    def __init__(self, K, N, max_iter=100, epsilon=1e-3):
        self.K = K
        self.N = N
        self.init_data()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def init_data(self):
        X = np.zeros((self.N))
        Mu = 10 * np.random.random(self.K)
        Sigma = np.random.random(self.K)
        print("Init mu:")
        print(Mu)
        print("Init Sigma:")
        print(Sigma)
        for i in range(self.N):
            cls = np.random.randint(0, self.K)
            X[i] = np.random.normal(Mu[cls], Sigma[cls])
        self.X = X
        plt.hist(X, 50)
        plt.show()

    def initialize_parameters(self):
        self.alpha0 = np.ones(self.K)
        self.beta0 = 1.0
        self.m0 = np.mean(self.X)
        self.W0 = 1.0
        self.v0 = 1.0

        self.alpha = np.ones(self.K)
        self.beta = np.ones(self.K)
        self.m = np.random.choice(self.X, self.K)
        self.W = np.ones(self.K)
        self.v = np.ones(self.K)
        self.resp = np.zeros((self.N, self.K))

    def e_step(self):
        for n in range(self.N):
            for k in range(self.K):
                ln_lambda = digamma(0.5 * (self.v[k] + 1)) - np.log(self.W[k])
                E_ln_pi = digamma(self.alpha[k]) - digamma(np.sum(self.alpha))
                E_ln_lambda = 0.5 * ln_lambda
                diff = self.X[n] - self.m[k]
                E_ln_x = -0.5 * (self.v[k] * self.W[k] * (diff ** 2 + 1 / self.beta[k]))

                self.resp[n, k] = np.exp(E_ln_pi + E_ln_lambda + E_ln_x)

        self.resp /= np.sum(self.resp, axis=1, keepdims=True)

    def m_step(self):
        Nk = np.sum(self.resp, axis=0)
        xk = np.dot(self.resp.T, self.X) / Nk

        self.alpha = self.alpha0 + Nk
        self.beta = self.beta0 + Nk
        self.m = (self.beta0 * self.m0 + Nk * xk) / self.beta
        self.W = 1 / (1 / self.W0 + Nk * np.var(self.X, axis=0))
        self.v = self.v0 + Nk

    def fit(self):
        self.initialize_parameters()
        for iteration in range(self.max_iter):
            prev_alpha = np.copy(self.alpha)
            self.e_step()
            self.m_step()

            if np.all(np.abs(self.alpha - prev_alpha) < self.epsilon):
                print(f"Converged in {iteration} iterations.")
                break
        else:
            print("Reached maximum iterations without convergence.")

        self.mu = self.m
        self.sigma = np.sqrt(1 / (self.v * self.W))

    def get_parameters(self):
        return self.mu, self.sigma

# Example usage
gmm = GMM1d_VarBayes(K=3, N=1000, max_iter=100, epsilon=1e-3)
gmm.fit()
mu, sigma = gmm.get_parameters()
print("Mu:", mu)
print("Sigma:", sigma)
