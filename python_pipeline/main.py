import os
import numpy as np
import utils
import BayesianHyperOptClasses as bho

if __name__ == "__main__":

    # load the data
    data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv")
    dataset = np.loadtxt(data_file_path, delimiter=";", encoding='utf-8-sig')
    print(dataset.shape)

    opt = utils.opt()

    # Create an instance of the class that contains the ML model that should be optimized
    search_space_cat = {'iterations': (100, 1000),
                        'depth': (1, 8),
                        'learning_rate': (0.01, 1.0),
                        'random_strength': (1e-9, 10),
                        'bagging_temperature': (0.0, 1.0),
                        'border_count': (1, 255),
                        'l2_leaf_reg': (2, 30),
                        }
    trainabale_model = bho.CatBoost_HyperOpt(X = np.zeros((1,1)), y = np.zeros((1,1)),
                            train_test_split_perc = 0.8, search_space = search_space_cat,
                            is_reg_task = "True", perf_metric = "rmse", max_or_min = "min", 
                            init_points=2, n_iter=4, device="CPU")
    
    # handover the "trainable model" to the expanding window method
    utils.expanding_window(data=dataset, model=trainabale_model, ind_f_vars=[104], col_names=[""],num_factors=4,num_lags=4, opt=opt)
