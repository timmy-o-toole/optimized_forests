import pycatch22
import os
import numpy as np
import matplotlib.pyplot as plt
import utils

benchmark_rf_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\fred__75__rf__h_1__4.csv")
benchmark_rf_errors = np.loadtxt(benchmark_rf_path, encoding='utf-8-sig')
utils.mse_from_error_vec(benchmark_rf_errors, plot=True, verbose=True)
#plt.hist(benchmark_rf, bins=100, edgecolor='k')
#plt.show()


data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv")
dataset = np.loadtxt(data_file_path, delimiter=";", encoding='utf-8-sig')
col_names = [str(i) for i in range(dataset.shape[1])]

print(dataset.shape)

#tsData = [0,1,2,3,4]
#pycatch22.DN_HistogramMode_5(dataset)