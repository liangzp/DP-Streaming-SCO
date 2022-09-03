import time
import numpy as np
from math import sqrt
from utils import paras

class SCO_steam_env:
    def __init__(self, params):
        paras(self, params)
        self.random_seed = params['env']['random_seed']
        self.x_std = params['env']['x_std']
        self.y_std = params['env']['y_std']
        self.p = params['env']['p']
        self.noise_norm = params['env']['noise_norm']
        if np.isinf(self.p):
            self.q = 1
        else:
            self.q = self.p / (self.p - 1)
        self.algo = params['algo']['type']
        self.logger = params['logger']

    def run(self):

        # init theta
        theta = np.zeros(shape=self.size_SCO)  
        theta[:self.s] = 1
        theta /= np.linalg.norm(theta.flatten(), self.p)

        # initialize the baseline risk
        self.algo.logger = self.logger
        super(type(self.algo), self.algo).train(theta)

        self.logger.record('time', time.time())

        #theta = np.random.normal(0, self.x_std, size = (self.d, 1))
        xs = np.random.normal(0, self.x_std, size=(self.T, self.d))
            # x /= np.abs(x).max()
        xs /= np.linalg.norm(xs, self.q, axis = 1, keepdims= True)
        # print(xs.shape, xs.max().max())
        # ys = xs.dot(theta) + np.random.normal(0, self.y_std, size=(self.T, 1))
        # here we clip the noise to [-1, 1], to guarantee our lipschitz
        noise = np.random.normal(0, self.y_std, size=(self.T, 1))
        noise = noise.clip(-self.noise_norm, self.noise_norm)
        ys = xs.dot(theta) + noise


        for t in range(self.T):
            x, y = xs[[t]], ys[[t]]
            self.algo.update(x.T, y, self.T, t)
            self.logger.record('est_error', np.linalg.norm(self.algo.theta_hat - theta) ** 2)

        self.theta_hat = self.algo.theta_hat

        # testing
        X = np.random.normal(0, self.x_std, size=(10000, self.d))
        X = np.array([x/np.linalg.norm(x, self.q) for x in X])
        y = X.dot(theta) + np.random.normal(0, self.y_std, size=(10000, 1))
        self.risk = ((X.dot(self.theta_hat) - y) ** 2).sum() / 10000
        self.baseline = ((X.dot(np.zeros(self.size_SCO)) - y) ** 2).sum() / 10000
        self.logger.record('baseline', self.baseline)
        self.logger.record('end time', time.time())





class SCO_batch_env:
    def __init__(self, params):
        paras(self, params)
        self.random_seed = params['env']['random_seed']
        self.x_std = params['env']['x_std']
        self.y_std = params['env']['y_std']
        self.algo = params['algo']['type']
        self.p = params['env']['p']
        self.noise_norm = params['env']['noise_norm']
        if np.isinf(self.p):
            self.q = 1
        else:
            self.q = self.p / (self.p - 1)
        self.logger = params['logger']
    
    def run(self):
        # init theta
        theta = np.zeros(shape = self.size_SCO)
        theta[:self.s] = 1
        theta /= np.linalg.norm(theta.flatten(), self.p)
        
        X = np.random.normal(0, self.x_std, size = (self.T, self.d))
        X = np.array([x/np.linalg.norm(x, self.q) for x in X])
        # here we clip the noise to [-1, 1], to guarantee our lipschitz
        noise = np.random.normal(0, self.y_std, size=(self.T, 1))
        noise = noise.clip(-self.noise_norm, self.noise_norm)
        y = X.dot(theta) + noise
        S = np.hstack((X,y))
        self.theta_hat = self.algo.train(S, theta, self.logger)


class bandits_env:
    def __init__(self, params):
        paras(self, params)
        self.random_seed = params['env']['random_seed']
        self.algo = params['algo']['type']
        self.logger = params['logger']
        self.logger.record('time', time.time())
        self.multi = params['bandit']['multi']

    def run(self):
        np.random.seed(self.random_seed)
        
        # init theta
        if self.multi:
            theta = np.zeros(shape = (self.d, self.k)) # np.random.normal(0, 0.05, size = size_1)
            for i in range(self.k):
                indice = np.random.choice(self.d, self.s)
                theta[indice, i] = 1
                theta[indice, i] /= np.linalg.norm(theta[indice, i], 1)

            xs = np.random.normal(0, self.x_std, size=(self.T, self.d))
            xs /= np.linalg.norm(xs, np.inf, axis = 1, keepdims= True)
            ys = xs.dot(theta)+ np.random.normal(0, self.y_std, size=(self.T, 1))

            for t in range(self.T):
                X = xs[[t]]
                X /= np.linalg.norm(X, np.inf)
                X = X.T
                y = X.T.dot(theta) + np.random.normal(0, 0.05, size = (1, self.k))
                at = self.algo.decide(X.T, t)
                self.algo.update(X, y[:, [at]], t, at)
                est_error = np.linalg.norm(self.algo.theta_hat - theta)**2
                regret = (np.max(theta.T.dot(X)) - theta[:, [at]].T.dot(X))[0]
                self.logger.record('record', [t, est_error, regret, time.time() - self.logger.dict['time'][0]])
        else:
            theta = np.zeros(shape = (self.d, self.k)) 
            for i in range(self.k):
                indice = np.random.choice(self.d, self.s)
                theta[indice, i] = 1
                theta[indice, i] /= np.linalg.norm(theta[indice, i], 1)


            Xs = np.random.normal(0, 0.05, size=(self.T, self.d))
            Xs /= np.linalg.norm(Xs, np.inf, axis = 1, keepdims= True)
            Ys = Xs.dot(theta)+ np.random.normal(0, 0.05, size=(self.T, self.k))

            theta = theta.reshape((self.d*self.k))

            for t in range(self.T):
                x = Xs[[t]]
                X = np.zeros((self.d*self.k, self.k))
                for i in range(self.k):
                    X[i*self.d:(i+1)*self.d, i] = x.ravel()
                y = Ys[[t]]
                at = self.algo.decide(X, t)
                self.algo.update(X[:, [at]], y[:, [at]], t)
                est_error = np.linalg.norm(self.algo.theta_hat - theta)**2
                regret = np.max(X.T.dot(theta)) - X[:, [at]].T.dot(theta)
                self.logger.record('record', [t, est_error, regret, time.time() - self.logger.dict['time'][0]])