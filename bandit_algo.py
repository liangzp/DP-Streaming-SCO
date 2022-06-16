
from cmath import nan
import numpy as np
from math import log, sqrt, floor, log2, ceil
from utils import compute_linear_gradient, paras
from scipy.optimize import NonlinearConstraint, minimize
import copy
import matplotlib.pyplot as plt
import time
import functools

eps = 1e-5

class Algo(object):
    def __init__(self, params):
        paras(self, params)

    def update(self, theta0, theta1, x, y):
        pass

    def train(self,theta):
        self.start_time = time.time()
        X = np.random.normal(0, self.x_std, size=(self.test_size, self.d))
        self.X = np.array([x/np.linalg.norm(x, self.q) for x in X])
        self.y = self.X.dot(theta) + np.random.normal(0, self.y_std, size=(self.test_size, 1))
        self.baseline = ((self.X.dot(np.zeros(self.size_SCO)) - self.y) ** 2).sum() / self.test_size
        self.opt_risk = ((self.X.dot(theta) - self.y) ** 2).sum() / self.test_size
        self.logger.record('baseline', self.baseline)

    def test(self, t, theta_hat):
        self.risk = ((self.X.dot(theta_hat) - self.y) ** 2).sum() / self.test_size
        self.subopt = (self.risk - self.opt_risk)/(self.baseline-self.opt_risk)
        self.logger.record('record', [t, self.risk, self.subopt])


class DPUCB(Algo):
    def __init__(self, params):
        super().__init__(params)
        self.L = sqrt(self.d*self.k)
        self.alpha = 1/self.T
        self.m = ceil(log(self.T, 2) +1)
        self.sigma = 0.05
        self.S = 1
        self.sigma_noise = sqrt(16*self.m*self.L**4*log(4/self.delta)**2/self.eps**2)
        self.Gamma = self.sigma_noise*sqrt(2*self.m)*(4*sqrt(self.d*self.k)+2*log(2*self.T/self.alpha))
        self.rho_min, self.rho_max = self.Gamma, self.Gamma*3
        self.G, self.u  = np.zeros(shape = (self.d*self.k, self.d*self.k)), np.zeros(shape = (self.d*self.k, 1))
        # self.G += np.eye(self.d)
        self.gamma = self.sigma_noise*sqrt(self.m/self.Gamma)*(sqrt(self.d*self.k) + sqrt(2*log(2*self.T/self.alpha))) # sqrt(self.m*self.L**2*(sqrt(self.d) + 2*log(2*log(2*self.T/self.alpha)))/(sqrt(2)*self.eps))
        self.Zs = np.random.normal(0, self.sigma_noise, size = (self.T, self.d*self.k, self.d*self.k))
        self.us = np.random.normal(0, self.sigma_noise, size = (self.T, self.d*self.k, 1))

    def update(self, x, y, t):
        self.G += x.dot(x.T)
        self.u += y*x

    def decide(self, X, t):
        reward = list()
        Z = self.Zs[t]
        self.V = self.G + (Z + Z.T)/sqrt(2)
        self.ut = self.u + self.us[t]
        invV = np.linalg.inv(self.V)
        self.theta_hat = invV.dot(self.ut)
        self.betat = self.sigma*sqrt(2*log(2/self.alpha)+self.d*log(self.rho_max/self.rho_min + (t+1)*self.L**2/(self.d*self.rho_min)))+self.S*sqrt(self.rho_max)+self.gamma
        for i in range(self.k):
            reward.append(self.theta_hat.T.dot(X[:, i]) + self.betat*X[:, i].T.dot(invV).dot(X[:, i]))
        at = np.argmax(reward)
        return at

class OFW_peq1(Algo):
    '''
    Algorithm for p=1. 
    '''
    def __init__(self, params):
        paras(self, params)
        super().__init__(params)
        self.dt = np.zeros(self.size_SCO)
        self.etat, self.rhot = 1, 1
        self.theta_hat_old, self.theta_hat, self.theta_hat0 = np.zeros((self.d, self.k)), np.zeros((self.d, self.k)), np.zeros((self.d, self.k))
        self.Gt = np.zeros(self.size_SCO)  # clean summation of g_i up to t
        self.l1_radius = self.lr_scale # since our theta is normalized to have norm = 1
        self.l1_ball = np.eye(self.d)
        self.t0 = int(log(self.d*self.T)*log(self.T)/self.eps**2)
        self.hsub = 1

    def update(self, x, y, t, at):
        t = t + 1  # python starting from 0, while our algo starts from 1
        self.etat, self.rhot = 1 / (t + 1), 1 / (t + 1)

        gradient0, gradient1 = compute_linear_gradient(self.theta_hat_old[:, at], x, y), compute_linear_gradient(
            self.theta_hat[:, at], x, y)
        self.dt = gradient1 + (1-self.rhot)*(self.dt - gradient0) # when t=1, dt = 0. Thus I ignore the classification for t = 1 and t neq 1
        Lap = 4*sqrt(log(self.T)*log(1/self.delta))/(self.eps*sqrt((t+1)))
        max_indice = np.argmax(np.abs(self.dt + np.random.laplace(0, Lap, size = self.dt.shape))) # 
        v = (-np.sign(self.dt[max_indice]) * self.l1_radius * self.l1_ball[max_indice]).reshape((self.d, 1))
        self.theta_hat_old[:, at] = copy.copy(self.theta_hat[:, at])
        self.theta_hat[:, [at]] = self.theta_hat[:, [at]] + self.etat*(v-self.theta_hat[:, [at]])
  
    def get_theta0(self):
        for i in range(self.k):
            self.theta_hat0[:, i] = copy.copy(self.theta_hat[:, i])

    def decide(self, X, t):
        if t== self.k*self.t0+1:
            self.get_theta0()
        if t<= self.k*self.t0:
            return t%self.k
        else:
            rewards = list()
            psreward_max = max([X.dot(self.theta_hat0[:, [i]]) for i in range(self.k)])
            for i in range(self.k):
                reward = X.dot(self.theta_hat[:, i])
                if reward>psreward_max - self.hsub/2:
                    rewards.append(reward)
                else:
                    rewards.append(float("-inf"))
            at = np.argmax(rewards)
            return at