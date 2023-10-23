import pycatch22
import os
import numpy as np
import matplotlib.pyplot as plt
import utils

benchmark_rf_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\fred__75__rf__h_1__4.csv")
benchmark_rf_errors = np.loadtxt(benchmark_rf_path, encoding='utf-8-sig')

print("Benchmark RF:")
utils.mse_from_error_vec(benchmark_rf_errors, plot=False, verbose=True)

import pandas as pd
#own_rf_path = os.path.join(os.getcwd(), "trained_models\\BaggedTree_experiments_summary_expid1122_predvarid103.csv")
#own_rf_df = pd.read_csv(own_rf_path, delimiter=";")
#own_rf_errors = own_rf_df["test_point_err"].values
#print("\nOwn RF:")
#utils.mse_from_error_vec(own_rf_errors, plot=False, verbose=True)

own_boosted_path = os.path.join(os.getcwd(), "trained_models\\XGBoost_experiments_summary_expid204_predvarid103.csv") # 1
own_boosted_df = pd.read_csv(own_boosted_path, delimiter=";")
own_boosted_errors  = own_boosted_df["test_point_err"].values
print("Boosted Approach:")
utils.mse_from_error_vec(own_boosted_errors, plot=False, verbose=True)

plt.plot(benchmark_rf_errors, label="Benchmark RF")
#plt.plot(own_rf_errors, label="Own RF")
plt.plot(own_boosted_errors, label="Boosted Approach")
#own_boosted_errors = np.concatenate([own_boosted_errors, np.zeros(len(benchmark_rf_errors) - len(own_boosted_errors))])
#own_boosted_errors = list(own_boosted_errors).extend([0 for _ in range(len(benchmark_rf_errors) - len(own_boosted_errors))])
plt.plot(np.zeros(len(benchmark_rf_errors)))
#plt.plot(abs(own_boosted_errors)-abs(benchmark_rf_errors))
plt.show()

data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv")
dataset = np.loadtxt(data_file_path, delimiter=";", encoding='utf-8-sig')
col_names = [str(i) for i in range(dataset.shape[1])]
print(dataset.shape)

#tsData = [0,1,2,3,4]
#pycatch22.DN_HistogramMode_5(dataset)