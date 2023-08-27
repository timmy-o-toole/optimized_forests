import os
import numpy as np
import utils

if __name__ == "__main__":

    # load the data
    data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv")
    dataset = np.loadtxt(data_file_path, delimiter=";", encoding='utf-8-sig')
    print(dataset.shape)

    # add lags
    dataset_with_lags = utils.add_lags(dataset = dataset, lags=3)
    print(dataset_with_lags.shape)

    # TODO: setup conda
    # TODO: install the tree libraries and hyper-opt
    
    # TODO: write a loop with moving window that trains the trees (using hyper opt classes)
    # -> see: code\models\forecast_rf.m

    