import glob 
import sys
import os
import numpy as np
import _pickle as cPickle
sys.path.append(os.path.abspath(__file__))

for file_name in glob.glob('../results/*/*.pkl'):
    with open(file_name, "rb")  as input_file:
        dict_list = cPickle.load(input_file)
        if dict_list[0]['algo']['test_flag']==True:
            risk_mean = np.mean([logger_dict['result']['record'][-1][0] for logger_dict in dict_list])
            risk_std = np.std([logger_dict['result']['record'][-1][0] for logger_dict in dict_list])
            subopt = np.mean([logger_dict['result']['record'][-1][1] for logger_dict in dict_list])
            time_mean, time_std = -1, -1
        else:
            time_mean = np.mean([logger_dict['result']['time'][-1] for logger_dict in dict_list])
            time_std = np.std([logger_dict['result']['time'][-1] for logger_dict in dict_list])
            risk_mean, risk_std, subopt = -1, -1, -1

        par_path = os.path.abspath(os.path.join(file_name, os.pardir))
        summary_file = par_path + '/summary.csv'
        if not os.path.exists(summary_file):
            out = open(summary_file, 'w')
            out.write(',risk_mean,risk_std,time_mean,time_std, improvement\n')
        else:
            out = open(summary_file, 'a')

        out.write("{},{},{},{},{},{}\n".format(file_name.split('/')[-1][:-4], str(risk_mean), str(risk_std), str(time_mean), str(time_std), str(subopt)))
        out.close()

# time vs risk
# T vs risk
# d vs risk