from math import sqrt, log, sqrt
from scipy.stats import gennorm
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

p_inf = 10000

def GGPlus(d, p, sigma_plus):
    '''
    https://stats.stackexchange.com/questions/352668/generate-uniform-noise-from-a-p-norm-ball-x-p-leq-r
    '''
    shape = d/2
    scale = 2*sigma_plus**2
    r = np.random.gamma(shape = shape, scale = scale, size = (1)) ** 0.5
    eps = gennorm.rvs(p, size=(d)) + 1/float(p)
    s = np.random.binomial(1, 1/2, size = (d)) * 2 - 1
    x = eps * s
    y = r*x/(np.power(np.abs(x), p).sum())**(1/p)
    return y    

def paras(self, params):
    self.d = params['env']['d']
    self.T = params['env']['T']
    self.s = params['env']['s']
    self.k = params['env']['k']
    self.p = params['env']['p']
    self.test_size = params['env']['test_size']
    self.test_freq = params['env']['test_freq']
    self.lr_scale = params['algo']['lr_scale']
    self.noise_free = params['algo']['noise_free']
    self.test_flag = params['algo']['test_flag']
    self.x_std = params['env']['x_std']
    self.y_std = params['env']['y_std']
    self.L0 = params['env']['lip'] # Lipschitz
    self.L1 = 1 # smoothness
    self.size_SCO = (self.d, 1)
    self.size_bandits = (self.d, self.k)
    self.eps, self.delta = params['prv']['eps'], params['prv']['delta']
    self.random_seed = params['env']['random_seed']
    self.logger = params['logger']


def clip(grad, lip, q):
    size = grad.shape
    grad = grad.flatten()
    if np.linalg.norm(grad, q) > lip:
        con = lambda x: np.linalg.norm(x, q)
        nlc = NonlinearConstraint(con, 0, lip)
        fun = lambda x: np.linalg.norm(x-grad, q)
        # x_init = np.zeros_like(grad)
        x_init = grad / np.linalg.norm(grad, q)
        clip_grad = minimize(fun, x_init, constraints=nlc).x
    else:
        clip_grad = grad
    clip_grad = clip_grad.reshape(size)
    return clip_grad


def compute_linear_gradient(theta_, x, y):
    return 2 * (theta_.T.dot(x) - y) * x


def lp_projection(x0, r, p):
    shape = x0.shape
    x0 = x0.flatten()
    fun = lambda x: np.linalg.norm(x-x0, 2) 
    x_init = np.zeros(shape = (x0.shape[0]))
    con = lambda x: np.linalg.norm(x, p)
    nlc = NonlinearConstraint(con, 0, r)
    res = minimize(fun, x_init, constraints = nlc).x
    return res.reshape(shape)

def Proj_inf(x0, r):
    return np.clip(x0, -r, r)

class Logger:
    def __init__(self):
        self.dict = dict()

    def record(self, key, value):
        if key not in self.dict:
            self.dict[key] = list()
        self.dict[key].append(value)