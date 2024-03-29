### old version
###
 # @Author: liangzp zliangao@connect.ust.hk
 # @Date: 2022-09-10 13:21:14
 # @LastEditors: liangzp zliangao@connect.ust.hk
 # @LastEditTime: 2023-02-26 08:16:58
 # @FilePath: /DP-Streaming-SCO/execute.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# nohup /usr/bin/python3 main.py --p 1.5 --grid_scale 0.4 0.6 1.0 --noise_free 0 --algo OFW_ple2 --d 5 10 20 50 --T 1000 2000 5000 10000;
# nohup /usr/bin/python3 main.py --p 1.5 --grid_scale 0.1 0.2 0.3 0.4 --noise_free 1 --algo OFW_ple2 --d 5 10 20 50 --T 1000 2000 5000 10000;
# /usr/bin/python3 main.py --p 1.5 --grid_scale 0.01 0.02 0.04 0.06 0.08 0.1 --noise_free 0 --algo NoisySFW --d 5 10 20 50 --T 1000 2000 5000 10000; 
# /usr/bin/python3 main.py --p 1.5 --grid_scale 0.10 0.20 0.30 0.40 --noise_free 1 --algo NoisySFW --d 5 10 20 50 --T 1000 2000 5000 10000; 
# /usr/bin/python3 main.py --p 1.5 --grid_scale 1e2 1e3 1e4 --noise_free 0 --algo Local_MD --d 5 10 20 50 --T 1000 2000 5000 10000; 
# /usr/bin/python3 main.py --p 1.5 --grid_scale 1e2 1e3 1e4 --noise_free 1 --algo Local_MD --d 5 10 20 50 --T 1000 2000 5000 10000;
# /usr/bin/python3 main.py --p inf --grid_scale 0.6 0.7 0.8 1 --noise_free 0 --algo OFW_pge2 --d 5 10 20 50 --T 1000 2000 5000 10000;
# /usr/bin/python3 main.py --p inf --grid_scale 0.6 0.7 0.8 1 --noise_free 1 --algo OFW_pge2 --d 5 10 20 50 --T 1000 2000 5000 10000;
# /usr/bin/python3 main.py --p inf --grid_scale 0.1 0.4 0.8 1 1.2 2 --noise_free 0 --algo NoisySGD --d 5 10 20 50 --T 1000 2000 5000 10000;
# /usr/bin/python3 main.py --p inf --grid_scale 0.1 0.4 0.8 1 1.2 2 --noise_free 1 --algo NoisySGD --d 5 10 20 50 --T 1000 2000 5000 10000; 

# bandit
#nohup /usr/bin/python3 main.py --p 1 --grid_scale 1 --noise_free 0  --grid_scale 1 2 3 4 5 --algo OFW_peq1 --d 10 20 50 --T 1000 2000 5000 10000&
#nohup /usr/bin/python3 main.py --p 1 --grid_scale 1 --noise_free 0  --grid_scale 1 --algo DPUCB --d 10 20 50 --T 1000 2000 5000 10000&


### new version

# # SCO
# /usr/bin/python3 main.py --p 1.5 --noise_free 0 --algo OFW_ple2 --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;
# # /usr/bin/python3 main.py --p 1.5 --noise_free 1 --algo OFW_ple2 --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;

# /usr/bin/python3 main.py --p 1.5 --noise_free 0 --algo NoisySFW --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;
# # /usr/bin/python3 main.py --p 1.5 --noise_free 1 --algo NoisySFW --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;

# /usr/bin/python3 main.py --p inf --noise_free 0 --algo OFW_pge2 --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;
# # /usr/bin/python3 main.py --p inf --noise_free 1 --algo OFW_pge2 --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;

# /usr/bin/python3 main.py --p inf --noise_free 0 --algo NoisySGD --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;
# # /usr/bin/python3 main.py --p inf --noise_free 1 --algo NoisySGD --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;

# # bandit
# nohup /usr/bin/python3 main.py --p 1 --noise_free 0  --algo OFW_peq1 --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;
# nohup /usr/bin/python3 main.py --p 1 --noise_free 0  --algo DPUCB    --n_random_seeds 10 --d 5 10 20 50 --T 1000 2000 5000 10000;


# # SCO
/usr/bin/python3 main.py --p 1.5 --noise_free 0 --algo OFW_ple2 --n_random_seeds 10 --d 5 10 15 20 25 50 --T 1000 2000;
/usr/bin/python3 main.py --p 1.5 --noise_free 0 --algo NoisySFW --n_random_seeds 10 --d 5 10 15 20 25 50 --T 1000 2000;

/usr/bin/python3 main.py --p 1.5 --noise_free 0 --algo OFW_pge2 --n_random_seeds 10 --d 5 10 15 20 25 50 --T 1000 2000;
/usr/bin/python3 main.py --p 1.5 --noise_free 0 --algo NoisySGD --n_random_seeds 10 --d 5 10 15 20 25 50 --T 1000 2000;