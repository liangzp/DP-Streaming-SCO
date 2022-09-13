from cmath import nan
import numpy as np
from math import log, sqrt
from utils import compute_linear_gradient, paras, GGPlus, lp_projection, Proj_inf, clip
from scipy.optimize import NonlinearConstraint, minimize
import copy
import matplotlib.pyplot as plt
import time
import functools
import tqdm 

eps = 1e-5

def argmin(grad, p, r):
    q = p/(p-1)
    return -np.sign(grad)*np.power(np.abs(grad), q-1)*r/(np.power(np.abs(grad), q).sum()**((q-1)/q)+eps) 


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


class NoisySFW(Algo):
    '''
    Algorithm 3 in Bassily 2021 for 1<p<2
    '''
    def __init__(self, params):
        paras(self, params)
        self.p = params['env']['p']
        self.q = self.p/(self.p - 1)
        self.kappa_q = min(self.q - 1, np.exp(1) ** 2 * (np.log(self.d) - 1))
        self.kappa_q_plus = min(self.q - 1, np.log(self.d) - 1)
        
        
    def train(self, S, theta, logger):
        self.logger = logger
        super().train(theta)
        r = 2 * np.linalg.norm(theta.flatten(), self.p) 
        M = 2 * r
        nabla = np.zeros(shape = (self.d, 1))
        n = S.shape[0]
        eta = log(n)/(2*sqrt(n)) * self.lr_scale
        self.logger.record('time', time.time())

        theta0 = np.zeros(shape = self.size_SCO)

        B0 = S[: len(S)//2]
        Shat = S[len(S)//2:]
        sigma = 4 * 2 **0.5 * sqrt(self.kappa_q) * self.L0 * log(1/self.delta) ** 0.5 / (n * self.eps)
        if self.noise_free: 
            GGNoise = np.zeros(shape = self.size_SCO)
        else:
            GGNoise = GGPlus(self.d, self.kappa_q_plus, sigma) 
        nabla = np.mean([clip(compute_linear_gradient(theta0, s[:-1], s[-1]), self.L0, self.q) for s in B0], axis = 0).reshape((-1, 1))\
                + GGNoise.reshape((-1, 1))
        theta1 = (1-eta)*theta0 + eta * argmin(nabla, self.p, r)
        sigma = sqrt((32*self.kappa_q*self.L1**2*M**2*eta**2*log(1/self.delta))/(n*self.eps**2))
        theta_list = list()

        for t in range(1, int(sqrt(n))):
            B = Shat[(t-1)*int(sqrt(n)//2):t*int(sqrt(n)//2)]
            if self.noise_free: 
                GGNoise = np.zeros(shape = self.size_SCO)
            else:
                GGNoise = GGPlus(self.d, self.kappa_q_plus, sigma).reshape((-1, 1)) 
            grad = np.mean([clip(compute_linear_gradient(theta1, s[:-1], s[-1]), self.L0, self.q) - clip(compute_linear_gradient(theta0, s[:-1], s[-1]), self.L0, self.q) for s in B], axis = 0).reshape((-1, 1))\
                + GGNoise
            nabla += grad
            theta0 = copy.deepcopy(theta1)
            theta1 = (1-eta)*theta1 + eta*argmin(nabla, self.p, r)
            if (t%max(1, int(sqrt(n))//self.test_freq)==0) and (self.test_flag==True):    
                self.test(t, theta1)
            self.logger.record('iteration', t)
            self.logger.record('time', time.time() - self.logger.dict['time'][0])
        return theta1

class NoisySGD(Algo):
    '''
    Algorithm 4 in Stability of Stochastic Gradient Descent on Nonsmooth Convex Losses for 2<p<=infty
    Learning rate should be <=1. Be careful, the noise is in scaling of learning rate
    '''
    def __init__(self, params):
        paras(self, params)
        self.random_seed = params['env']['random_seed']
        self.kappa = min(1/(self.d-1), 2*log(self.d)) 
        self.p = params['env']['p']
        if np.isinf(self.p):
            self.q = 1
        else:
            self.q = self.p / (self.p - 1)
        
    def train(self, S, theta, logger):
        self.logger = logger
        super().train(theta)
        r = np.linalg.norm(theta.flatten(), self.p)
        n = S.shape[0]
        sigma = sqrt(8*self.L0**2*log(1/self.delta)/self.eps**2)
        theta1 = np.zeros(shape = (self.d, 1))
        # theta_list = list()
        theta_avg = np.zeros(shape=(self.d, 1))
        if self.noise_free: 
            noise_mechanism = functools.partial(int, 0)
        else:
            noise_mechanism = functools.partial(np.random.normal, loc = 0, scale = sigma, size = (self.d, 1))

        if self.p < np.inf:
            proj = functools.partial(lp_projection, r = 2*r,  p = self.p)
        else:
            proj = functools.partial(Proj_inf, r = r)

        # initial time
        self.logger.record('time', time.time())
        # indices = np.random.choice(n, n**2)
        # GGNoises = [noise_mechanism() for _ in range(n**2-1)]
        X = S[:, :-1]
        Y = S[:, -1]
        # for t in range(n**2-1):
        for t in range(1, n**2+1):
            i = np.random.choice(n, 1)
            GGNoise = noise_mechanism()
            etat = r/(self.L0*n*max(sqrt(n), sqrt(self.d*log(1/self.delta))/self.eps))*self.lr_scale
            gradient = compute_linear_gradient(theta1, X[i].reshape((-1, 1)), Y[i]).reshape((-1, 1))
            gradient = clip(gradient, self.L0, self.q)
            noisy_gradient = gradient + GGNoise
            theta1 = proj(noisy_gradient)
            # theta_list.append(theta1)
            theta_avg += theta1 / n**2
            if (self.test_flag==True) and t%(n**2//self.test_freq)==0:
                self.test(t, (theta_avg * n**2)/(t+1))

        self.logger.record('time', time.time() - self.logger.dict['time'][0])
        # return np.mean(theta_list, axis = 0)
        return theta_avg


class OFW_ple2(Algo):
    '''
    Algorithm for 1<p<=2. Learning rate should be <=1
    '''
    def __init__(self, params):
        super().__init__(params)
        self.dt = np.zeros(self.size_SCO)
        self.etat, self.rhot = 1, 1
        self.theta_hat_old, self.theta_hat = np.zeros(self.size_SCO), np.zeros(self.size_SCO)
        self.Gt = np.zeros(self.size_SCO)  # clean summation of g_i up to t
        self.r = 2 # since our theta is normalized to have norm=1


        self.q = self.p / (self.p - 1)
        self.kappa_q = min(self.q - 1, np.exp(1) ** 2 * (np.log(self.d) - 1))
        self.kappa_q_plus = min(self.q - 1, np.log(self.d) - 1)

        self.noises = self.get_noise()
        self.noise_by_step = self.get_noise_by_step()

    def lr_scheduler(self, t):
        etat = 1 / (t + 1)
        etat = etat * self.lr_scale
        return etat

    def get_sigma_plus(self, t):
        etat = self.lr_scheduler(t)
        logn = np.ceil(np.log2(self.T)) + 1
        sigma_plus_2 = 8 * logn ** 2 * self.kappa_q * np.log(logn / self.delta) * (
                    self.L1 * 2 * self.r * (t+1) * etat + self.L0) ** 2 / self.eps ** 2
        sigma_plus = sigma_plus_2 ** 0.5
        return sigma_plus

    def get_noise(self):
        noises = np.zeros((self.d, self.T))
        for i in range(self.T):
            sigma_plus = self.get_sigma_plus(i)
            noises[:, i] = GGPlus(self.d, self.kappa_q_plus, sigma_plus)
        return noises

    def push_gt(self, gt):
        self.Gt += gt

    def get_noise_idx(self, t):
        # in each round t, we need to know the indices of nodes we need to get the partial sum
        # these indices will allow use the get the corresponding fixed noises of thoese nodes
        # for example, when t=7, as need node A4+A6+A7,
        # where A4 = g1+g2+g3+g4 + noise, A6 = g5 + g6 + noise, A7 = g7 + noise
        # so we need to get the noise from A4, A6, A7. This function will return [4,6,7] for us.
        # The idea is like, we decompose a number 7 into 4+2+1, the index for node would be 4, 4+2, 4+2+1
        bin_t = format(t, 'b')
        node_idx = []
        for i in range(len(bin_t)):
            if bin_t[i] == '1':
                idx = '1' + '0' * (len(bin_t) - i - 1)
                if len(node_idx) == 0:
                    node_idx.append(int(idx, 2))
                else:
                    node_idx.append(node_idx[-1] + int(idx, 2))
        return node_idx

    # def get_Gt(self, t):
    #     noise_idx = np.array(self.get_noise_idx(t)) - 1
    #     if self.noise_free:
    #         noise = np.zeros(shape=self.size_SCO)
    #     else:
    #         noise = self.noises[:, noise_idx].sum(axis=1)
    #     return self.Gt + noise.reshape(self.size_SCO)

    def get_noise_by_step(self):
        noise_by_step = []

        for t in range(1, self.T+1):
            noise_idx = np.array(self.get_noise_idx(t)) - 1
            if self.noise_free:
                noise = np.zeros(shape=self.size_SCO)
            else:
                noise = self.noises[:, noise_idx].sum(axis=1)
            noise_by_step.append(noise)

        return np.array(noise_by_step)

    def get_Gt(self, t):
        return self.Gt + self.noise_by_step[t-1].reshape(self.size_SCO)


    def update(self, x, y, T, t):

        t = t + 1  # python starting from 0, while our algo starts from 1
        self.etat, self.rhot = 1 / (t + 1), 1 / (t + 1)

        self.etat = self.etat * self.lr_scale
        self.rhot = self.rhot * self.lr_scale

        gradient0, gradient1 = compute_linear_gradient(self.theta_hat_old, x, y), compute_linear_gradient(
            self.theta_hat, x, y)
        gradient0 = clip(gradient0, self.L0, self.q)
        gradient1 = clip(gradient1, self.L0, self.q)
        gt = (t + 1) * gradient1 - t * gradient0
        self.push_gt(gt)
        Gt = self.get_Gt(t)
        self.dt = Gt / (1 + t)
        v = -1 * np.sign(self.dt) * np.abs(self.dt) ** (self.q - 1) * self.r / np.linalg.norm(self.dt.flatten(), self.q) ** (self.q - 1)
        self.theta_hat_old = copy.copy(self.theta_hat)
        self.theta_hat = self.theta_hat + self.etat * (v - self.theta_hat)

        # we control the total number of testing time to be self.test_freq
        if (self.test_flag==True) and (t % max(1, (T // self.test_freq)) == 0):
            self.test(t, self.theta_hat)
        self.logger.record('iteration', t)
        self.logger.record('time', time.time() - self.logger.dict['time'][0])

    def decide(self, X, t):
        reward = list()
        for i in range(self.k):
            reward.append(self.theta_hat.T.dot(X[:, i]))
        at = np.argmax(reward)
        return at

class OFW_peq1(Algo):
    '''
    Algorithm for p=1. 
    '''
    def __init__(self, params):
        super().__init__(params)
        self.dt = np.zeros(self.size_SCO)
        self.etat, self.rhot = 1, 1
        self.theta_hat_old, self.theta_hat = np.zeros(self.size_SCO), np.zeros(self.size_SCO)
        self.Gt = np.zeros(self.size_SCO)  # clean summation of g_i up to t
        self.r = 2 # since our theta is normalized to have norm=1

    def update(self, x, y, T, t):
        t = t + 1  # python starting from 0, while our algo starts from 1
        self.etat, self.rhot = 1 / (t + 1), 1 / (t + 1)

        self.etat = self.etat * self.lr_scale
        self.rhot = self.rhot * self.lr_scale

        gradient0, gradient1 = compute_linear_gradient(self.theta_hat_old, x, y), compute_linear_gradient(
            self.theta_hat, x, y)
        gradient0 = clip(gradient0, self.L0, np.inf)
        gradient1 = clip(gradient1, self.L0, np.inf)
        gt = (t + 1) * gradient1 - t * gradient0
        self.push_gt(gt)
        Gt = self.get_Gt(t)
        self.dt = gradient1 + (1-self.rhot)*(self.dt - gradient0) # when t=1, dt = 0. Thus I ignore the classification for t = 1 and t neq 1
        Lap = 4*sqrt(log(self.T)*log(1/self.delta))/(self.eps*sqrt((t+1)))
        max_indice = np.argmax(np.abs(self.dt + np.random.laplace(0, Lap, size = self.dt.shape))) # 
        v = (-np.sign(self.dt[max_indice]) * self.l1_radius * self.l1_ball[max_indice]).reshape(self.theta_hat.shape)
        self.theta_hat_old = copy.copy(self.theta_hat)
        self.theta_hat = self.theta_hat + self.etat*(v-self.theta_hat)

        # we control the total number of testing time to be self.test_freq
        if (self.test_flag==True) and (t % max(1, (T // self.test_freq)) == 0):
            self.test(t, self.theta_hat)
        self.logger.record('iteration', t)
        self.logger.record('time', time.time() - self.logger.dict['time'][0])

    def decide(self, X, t):
        reward = list()
        for i in range(self.k):
            reward.append(self.theta_hat.T.dot(X[:, i]))
        at = np.argmax(reward)
        return at


class Local_MD(Algo):
    '''
    Algorithm 6 (embedding Algo 5) in Asi, 2021
    Private Stochastic Convex Optimization: Optimal Rates in l1 Geometry
    it is for 1<p<=2
    '''

    def __init__(self, params):
        super().__init__(params)
        self.q = self.p / (self.p - 1)
        self.est_error = []

    def h(self, x, x0):
        return np.linalg.norm(x - x0, self.p) ** 2 / (2 * (self.p - 1))


    def h_grad(self, y, x0):
        diff = y - x0
        grad = diff * (np.abs(diff) + eps) ** (self.p-2) / ((self.p-1) * np.linalg.norm(np.abs(diff) + eps, self.p) ** (self.p-2))
        return grad


    def D_h(self, x, y, x0):
        grad = self.h_grad(y, x0)
        dh = self.h(x, x0) - self.h(y, x0) - ((x-y) * grad).sum()
        return dh

    def train(self, S, theta, logger):
        self.logger = logger
        super().train(theta)
        r = 2 * np.linalg.norm(theta.flatten(), self.p)  # M is diameter, r is radius
        D = 2 * r

        n = S.shape[0]
        kk = int(np.log2(n))

        eta_1 = 1 / ((self.p - 1) * n) ** 0.5
        eta_2 = self.eps / (self.d * np.log(1 / self.delta) * (1 + np.log(self.d))) ** 0.5
        eta = self.lr_scale * D / self.L0 * min(eta_1, eta_2)

        theta1 = np.zeros(shape=self.size_SCO)

        idx = 0

        self.logger.record('time', time.time())

        for i in range(1, kk + 1):
            eps_i = 2 ** (-i) * self.eps
            n_i = int(2 ** (-i) * n)
            eta_i = 2 ** (-4 * i) * eta

            # apply algorithm 5
            b_i = max((n_i / np.log(self.d)) ** 0.5, (self.d / eps_i) ** 0.5) ## b_i may be event larger than n_i
            b_i = int(max(b_i, 1))

            T = int(n_i ** 2 / b_i ** 2)
            if T <= 1: # there is an issue of the original paper that: b_i > n_i
                break

            con = lambda x: np.linalg.norm((x - theta1.flatten()), self.p)
            r = 2 * self.L0 * eta_i * n_i * (self.p - 1)
            nlc = NonlinearConstraint(con, 0, r)

            # run iteration on a subset
            data_idx = np.arange(idx, idx + n_i).astype(int)
            idx = idx + n_i
            subS = S[data_idx, :]

            theta_list = []
            theta0 = copy.copy(theta1)

            for j in range(1, T + 1):
                subsubS = np.random.choice(range(0, subS.shape[0]), b_i, replace=False)
                grad = np.mean([clip(compute_linear_gradient(theta0, subS[s, :-1], subS[s, -1]), self.L0, self.q) for s in subsubS],
                               axis=0).reshape((-1, 1))
                sigma = 100 * self.L0 * (self.d ** (1 - 2 / self.q) * np.log(1 / self.delta)) ** 0.5 / (b_i * eps_i)

                if self.noise_free:
                    noisy_grad = grad
                else:
                    noisy_grad = grad + np.random.normal(0, sigma, size=grad.shape)

                fun = lambda x: (noisy_grad.flatten() * (x - theta0.flatten())).sum() + self.D_h(x, theta0.flatten(), theta1.flatten()) / eta_i
                x_init = copy.copy(theta0).flatten()
                theta0 = minimize(fun, x_init, constraints=nlc).x
                theta0 = theta0.reshape((theta0.shape[0], 1))

                theta_list.append(copy.copy(theta0))

            theta1 = np.mean(theta_list, axis=0)

            if self.test_flag == True:
                self.test(i, theta1)
            self.logger.record('iteration', i)
            self.logger.record('time', time.time() - self.logger.dict['time'][0])

            self.est_error.append(np.linalg.norm((theta1 - theta).flatten(), 2) ** 2)
        return theta1

    def decide(self, X, t):
        reward = list()
        for i in range(self.k):
            reward.append(self.theta_hat.T.dot(X[:, i]))
        at = np.argmax(reward)
        return at



class OFW_pge2(Algo):
    '''
    Algorithm for p>2. Learning rate should be <=1
    '''
    def __init__(self, params):
        super().__init__(params)
        self.dt = np.zeros(self.size_SCO)
        self.etat, self.rhot = 1, 1
        self.theta_hat_old, self.theta_hat = np.zeros(self.size_SCO), np.zeros(self.size_SCO)
        self.Gt = np.zeros(self.size_SCO)  # clean summation of g_i up to t
        self.r = 2 # since our theta is normalized to have norm=1

        if np.isinf(self.p):
            self.q = 1
        else:
            self.q = self.p / (self.p - 1)
        self.kappa_q = self.d ** (1 - 2/self.p)
        # self.kappa_q_plus = 1

        self.noises = self.get_noise()
        self.noise_by_step = self.get_noise_by_step()

    def lr_scheduler(self, t):
        etat = 1 / (t + 1)
        etat = etat * self.lr_scale
        return etat

    def get_sigma_plus(self, t):
        etat = self.lr_scheduler(t)
        logn = np.ceil(np.log2(self.T)) + 1
        sigma_plus_2 = 8 * logn ** 2 * self.kappa_q * np.log(logn / self.delta) * (
                self.L1 * 2 * self.r * (t+1) * etat + self.L0) ** 2 / self.eps ** 2
        sigma_plus = sigma_plus_2 ** 0.5
        d_pow = 1 / 2 - 1 / self.p
        sigma_plus = sigma_plus / self.d ** d_pow

        return sigma_plus

    def get_noise(self):
        noises = np.zeros((self.d, self.T))
        for i in range(self.T):
            sigma_plus = self.get_sigma_plus(i)
            noises[:, i] = GGPlus(self.d, 2, sigma_plus)
        return noises

    def push_gt(self, gt):
        self.Gt += gt

    def get_noise_idx(self, t):
        # in each round t, we need to know the indices of nodes we need to get the partial sum
        # these indices will allow use the get the corresponding fixed noises of thoese nodes
        # for example, when t=7, as need node A4+A6+A7,
        # where A4 = g1+g2+g3+g4 + noise, A6 = g5 + g6 + noise, A7 = g7 + noise
        # so we need to get the noise from A4, A6, A7. This function will return [4,6,7] for us.
        # The idea is like, we decompose a number 7 into 4+2+1, the index for node would be 4, 4+2, 4+2+1
        bin_t = format(t, 'b')
        node_idx = []
        for i in range(len(bin_t)):
            if bin_t[i] == '1':
                idx = '1' + '0' * (len(bin_t) - i - 1)
                if len(node_idx) == 0:
                    node_idx.append(int(idx, 2))
                else:
                    node_idx.append(node_idx[-1] + int(idx, 2))
        return node_idx

    # def get_Gt(self, t):
    #     noise_idx = np.array(self.get_noise_idx(t)) - 1
    #     if self.noise_free:
    #         noise = np.zeros(shape=self.size_SCO)
    #     else:
    #         noise = self.noises[:, noise_idx].sum(axis=1)
    #     return self.Gt + noise.reshape(self.size_SCO)

    def get_noise_by_step(self):
        noise_by_step = []

        for t in range(1, self.T+1):
            if self.noise_free:
                noise = np.zeros(shape=self.size_SCO)
            else:
                noise_idx = np.array(self.get_noise_idx(t)) - 1
                noise = self.noises[:, noise_idx].sum(axis=1)
            noise_by_step.append(noise)

        return np.array(noise_by_step)

    def get_Gt(self, t):
        return self.Gt + self.noise_by_step[t-1].reshape(self.size_SCO)


    def update(self, x, y, T, t):

        t = t + 1  # python starting from 0, while our algo starts from 1
        self.rhot = 1 / (t + 1)
        self.rhot = self.rhot * self.lr_scale

        self.etat = self.lr_scheduler(t)

        gradient0, gradient1 = compute_linear_gradient(self.theta_hat_old, x, y), compute_linear_gradient(
            self.theta_hat, x, y)
        gradient0 = clip(gradient0, self.L0, self.q)
        gradient1 = clip(gradient1, self.L0, self.q)
        gt = (t + 1) * gradient1 - t * gradient0
        self.push_gt(gt)
        Gt = self.get_Gt(t)
        self.dt = Gt / (1 + t)
        if np.isinf(self.p):
            v = - np.sign(self.dt) * self.r
        else:
            v = -1 * np.sign(self.dt) * np.abs(self.dt) ** (self.q - 1) * self.r / np.linalg.norm(self.dt.flatten(), self.q) ** (self.q - 1)
        self.theta_hat_old = copy.copy(self.theta_hat)
        self.theta_hat = self.theta_hat + self.etat * (v - self.theta_hat)

        # we control the total number of testing time to be self.test_freq
        if (self.test_flag==True) and (t % max(1, (T // self.test_freq)) == 0):
            self.test(t, self.theta_hat)
        self.logger.record('iteration', t)
        self.logger.record('time', time.time() - self.logger.dict['time'][0])

    def decide(self, X, t):
        reward = list()
        for i in range(self.k):
            reward.append(self.theta_hat.T.dot(X[:, i]))
        at = np.argmax(reward)
        return at