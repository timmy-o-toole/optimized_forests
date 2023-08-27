import os
import numpy as np
import utils

if __name__ == "__main__":

    # load the data
    data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv")
    dataset = np.loadtxt(data_file_path, delimiter=";", encoding='utf-8-sig')
    print(dataset.shape)

    opt = utils.opt()
    
    utils.expanding_window(data=dataset, ind_f_vars=[104], col_names="",num_factors=4,num_lags=4, opt=opt)
    # TODO: write a loop with moving window that trains the trees (using hyper opt classes)
    # -> see: code\models\forecast_rf.m   