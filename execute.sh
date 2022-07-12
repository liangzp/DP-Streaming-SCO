nohup /usr/bin/python3 main.py --p 1.5 --grid_scale 0.4 0.6 1.0 --noise_free 0 --algo OFW_ple2 --d 5 10 20 50 --T 1000 2000 5000 10000;
nohup /usr/bin/python3 main.py --p 1.5 --grid_scale 0.1 0.2 0.3 0.4 --noise_free 1 --algo OFW_ple2 --d 5 10 20 50 --T 1000 2000 5000 10000;
/usr/bin/python3 main.py --p 1.5 --grid_scale 0.01 0.02 0.04 0.06 0.08 0.1 --noise_free 0 --algo NoisySFW --d 5 10 20 50 --T 1000 2000 5000 10000; 
/usr/bin/python3 main.py --p 1.5 --grid_scale 0.10 0.20 0.30 0.40 --noise_free 1 --algo NoisySFW --d 5 10 20 50 --T 1000 2000 5000 10000; 
/usr/bin/python3 main.py --p 1.5 --grid_scale 1e2 1e3 1e4 --noise_free 0 --algo Local_MD --d 5 10 20 50 --T 1000 2000 5000 10000; 
/usr/bin/python3 main.py --p 1.5 --grid_scale 1e2 1e3 1e4 --noise_free 1 --algo Local_MD --d 5 10 20 50 --T 1000 2000 5000 10000;
/usr/bin/python3 main.py --p inf --grid_scale 0.6 0.7 0.8 1 --noise_free 0 --algo OFW_pge2 --d 5 10 20 50 --T 1000 2000 5000 10000;
/usr/bin/python3 main.py --p inf --grid_scale 0.6 0.7 0.8 1 --noise_free 1 --algo OFW_pge2 --d 5 10 20 50 --T 1000 2000 5000 10000;
/usr/bin/python3 main.py --p inf --grid_scale 0.1 0.4 0.8 1 1.2 2 --noise_free 0 --algo NoisySGD --d 5 10 20 50 --T 1000 2000 5000 10000;
/usr/bin/python3 main.py --p inf --grid_scale 0.1 0.4 0.8 1 1.2 2 --noise_free 1 --algo NoisySGD --d 5 10 20 50 --T 1000 2000 5000 10000; 


# bandit
nohup /usr/bin/python3 main.py --p 1 --grid_scale 1 --noise_free 0  --grid_scale 1 2 3 4 5 --algo OFW_peq1 --d 10 20 50 --T 1000 2000 5000 10000&
nohup /usr/bin/python3 main.py --p 1 --grid_scale 1 --noise_free 0  --grid_scale 1 --algo DPUCB --d 10 20 50 --T 1000 2000 5000 10000&