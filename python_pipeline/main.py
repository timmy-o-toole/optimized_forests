import os
import numpy as np
import utils
import BayesianHyperOptClasses as bho
import warnings
warnings.filterwarnings("error")

if __name__ == "__main__":

    # load the data
    data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv")
    dataset = np.loadtxt(data_file_path, delimiter=";", encoding='utf-8-sig')
    print("Data points:\t", dataset.shape[0])
    print("Data dimensions:", dataset.shape[1])

    opt = utils.opt()

    # Create an instance of the class that contains the ML model that should be optimized
    fake_data = np.zeros((1,1))
    # CatBoost
    search_space_cat = {'iterations': (100, 1000),
                        'depth': (2, 9),
                        'learning_rate': (0.01, 1.0),
                        'random_strength': (1e-9, 10),
                        'bagging_temperature': (0.0, 1.0),
                        'border_count': (1, 255),
                        'l2_leaf_reg': (2, 30),
                        }
    trainabale_model_cat = bho.CatBoost_HyperOpt(X = fake_data, y = fake_data,
                            train_test_split_perc = 0.8, search_space = search_space_cat,
                            is_reg_task = "True", perf_metric = "RMSE", max_or_min = "min", 
                            init_points=2, n_iter=4, device="CPU")
    
    # LightGBM
    search_space_lgbm = {
        #"n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": (0.01, 0.3), # float
        "num_leaves": (10, 3000), # int
        "max_depth": (2, 9), #int
        "min_data_in_leaf": (5, 50), #int
        "lambda_l1": (0, 100), #int
        "lambda_l2": (0, 100), #int
        "min_gain_to_split": (0, 15), #float
        "bagging_fraction": (0.2, 0.95), #float
        #"bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": (0.1, 0.95) #float
    }
    trainabale_model_lgbm = bho.LightGBM_HyperOpt(X = fake_data, y = fake_data,
                         train_test_split_perc = 0.8, search_space = search_space_lgbm,
                         is_reg_task = "True", perf_metric = "RMSE", max_or_min = "min",
                         init_points=2, n_iter=4, device="CPU")
    
    # XGBoost
    search_space_xgb = {"lambda_": (1e-9, 1.0),
                    "alpha": (1e-9, 1.0),
                    "max_depth": (2, 9),
                    "eta": (1e-9, 1.0),
                    "gamma": (1e-8, 1.0)
                   }
    trainabale_model_xgb = bho.XGBoost_HyperOpt(X = fake_data, y = fake_data,
                            train_test_split_perc = 0.8, search_space = search_space_xgb,
                            is_reg_task = "True", perf_metric = "RMSE", max_or_min = "min",
                            init_points=2, n_iter=4, device="CPU")

    # BaggedTree
    trainabale_model_bagged = bho.BaggedTree(X = fake_data, y = fake_data, n_estimators=5)

    # handover the "trainable model" to the expanding window method
    errors = utils.expanding_window(data=dataset, model=trainabale_model_cat, ind_f_vars=[104], col_names=[""],
                           num_factors=4, num_lags=2, opt=opt, min_window_size=565)
    print("Sum of squared errors: ", sum([e*e for e in errors[0]]))

    # Next up:
    # TODO: add functions to trainable_model class that store and load instances of the class
    # TODO: visualize the optimization process during hyper-opt (use hyp-opt library)
    # TODO: create a procedure for evaluating and comparing performance
    # TODO: check more in detail if the implementation of BaggedTree is as in Matlab
    # TODO: write a hyper opt version of BaggedTree
    # TODO: check correctness of expanding_window() - did not understand Matlab code fully