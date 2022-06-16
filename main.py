import sys
import os
import pickle
import argparse
import numpy as np
from env import *
from algo import *
from utils import Logger
from bandit_algo import *
from itertools import product
import matplotlib.pyplot as plt
from multiprocessing import Pool
sys.path.append(os.path.abspath(__file__))

def generate_params(algo, seed, p, d = 5, T = int(1e2), scale = 1, noise_free = False, test_flag = False
):  
    params = dict()
    params['env'] = {"random_seed": seed,
                     "T": T,
                     "d": d,
                     "s": 5,  
                     "k": 5,
                     "p": p, 
                     "x_std": 0.05,
                     "y_std": 0.05,
                     "test_size": 1000, 
                     "test_freq" : 100,
                     }

    np.random.seed(seed)
    
    logger = Logger()
    params['prv'] = {"eps": 1,
                     "delta": 1/params['env']['T']}
    params['logger'] = logger
    params['algo'] = dict()
    params['algo']['lr_scale'] = scale
    params['algo']['noise_free'] = noise_free
    params['algo']['test_flag'] = test_flag
    params['bandit'] = dict()
    if algo == 'OFW_ple2':
        params['algo']['type'] = OFW_ple2(params)
        env = SCO_steam_env(params)
    elif algo == 'OFW_pge2':
        params['algo']['type'] = OFW_pge2(params)
        env = SCO_steam_env(params)
    elif algo == 'Local_MD':
        params['algo']['type'] = Local_MD(params)
        env = SCO_batch_env(params)
    elif algo=='NoisySFW':
        params['algo']['type'] = NoisySFW(params)
        env = SCO_batch_env(params)
    elif algo=='NoisySGD':
        params['algo']['type'] = NoisySGD(params)
        env = SCO_batch_env(params)
    elif algo=='DPUCB':
        params['algo']['type'] = DPUCB(params)
        params['bandit']['multi'] = False
        env = bandits_env(params)
    elif algo=='OFW_peq1':
        params['algo']['type'] = OFW_peq1(params)
        params['bandit']['multi'] = True
        env = bandits_env(params)
        
    env.run()
    params['result'] = env.logger.dict
    return p, d, T, algo, str(scale), str(noise_free), str(test_flag), params

if __name__ == '__main__':
    # parser argument in the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type = str, nargs = '+')
    parser.add_argument('--T', type = float, default = [1e2], nargs = '+',
                        help='sample size')  
    parser.add_argument('--p', type = str, default = ['1.5'], nargs = '+',
                        help='geometries')   
    parser.add_argument('--d', type = int, default = [5], nargs = '+',
                        help='dimension')
    parser.add_argument('--n_random_seeds', type = int, default = 5,
                        help='total amount of random seeds')
    parser.add_argument('--grid_scale', type=float, default=[1e-1], nargs='+',
                        help='learning rate grid scale')
    parser.add_argument('--noise_free', type=lambda x:bool(int(x)), default=False, nargs='+',
                        help='we do not set noise when noise_free==1')

    args = parser.parse_args()
    p = [np.inf if (p_ =='inf') else float(p_) for p_ in args.p]
    T = [int(T_) for T_ in args.T]
    d = [int(d_) for d_ in args.d]
    grid_scale = args.grid_scale
    noises = args.noise_free
    test_flags = [True, False] 
    random_seeds = list(range(args.n_random_seeds))
    algo = args.algo
    n_process = 5

    # parallel run
    print(f'using {n_process} processes')
    with Pool(processes = n_process) as pool:
        collection_sources = pool.starmap(generate_params, product(algo, random_seeds, p, d, T, grid_scale, noises, test_flags))

    # output experiments result to the disk
    results_dict = dict()
    for collection_source in collection_sources:
        p, d, T, algo, scale, noise_free, test_flag = collection_source[0], collection_source[1], collection_source[2], collection_source[3], collection_source[4], collection_source[5], collection_source[6]
        params = collection_source[-1]
        if abs(p-1.0)>1e-5 or str(p)=='inf':
            result_folder = 'dpsco-results'
        else:
            result_folder = 'bandits-results-2'
        folder_name = "../{}/p={}-d={}-T={}-scale={}-noise_free={}-test_flag={}/".format(result_folder, str(p), str(d), str(T), str(scale),str(noise_free),str(test_flag))
        if not folder_name in results_dict:
            results_dict[folder_name] = dict()
        if not algo in results_dict[folder_name]:
            results_dict[folder_name][algo] = list()
        results_dict[folder_name][algo].append(params)
    
    for folder_name in results_dict.keys():
        print(folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for algo in results_dict[folder_name].keys():
            with open(folder_name + algo + '.pkl', 'wb') as f:
                pickle.dump(results_dict[folder_name][algo], f)
