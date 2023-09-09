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
    print("--------------------------------------")
    opt = utils.opt()


    # Create an instance of the class that contains the ML model that should be optimized
    
    
    ### XGBoost
    search_space_xgb = {'lag_to_add': (0, 6),
                        "lambda_": (1e-9, 1.0),
                        "alpha": (1e-9, 1.0),
                        "max_depth": (2, 9),
                        "eta": (1e-9, 1.0),
                        "gamma": (1e-8, 1.0)
                        }
    trainabale_model_xgb = bho.XGBoost_HyperOpt(experiment_id=0, train_test_split_perc = 0.8, search_space = search_space_xgb,
                            is_reg_task = "True", perf_metric = "RMSE", max_or_min = "min",
                            init_points=1, n_iter=2, device="CPU", optimize_lag=True)
    ###### testing XGBoost
    errors_xgb = trainabale_model_xgb.expanding_window(lagless_data=dataset, ind_f_vars=[104], col_names=[""],
                                                       num_factors=4, num_lags=2, opt=opt, min_window_size=550, verbose=0)
    print("Parameters: ", trainabale_model_xgb.optimal_params)
    print("Sum of squared errors - XGBoost: ", sum([e*e for e in errors_xgb[0]]))


    ### CatBoost
    search_space_cat = {'lag_to_add': (0, 6),
                        'iterations': (100, 1000),
                        'depth': (2, 9),
                        'learning_rate': (0.01, 1.0),
                        'random_strength': (1e-9, 10),
                        'bagging_temperature': (0.0, 1.0),
                        'border_count': (1, 255),
                        'l2_leaf_reg': (2, 30),
                        }
    trainabale_model_cat = bho.CatBoost_HyperOpt(experiment_id=0, train_test_split_perc = 0.8, search_space = search_space_cat,
                            is_reg_task = "True", perf_metric = "RMSE", max_or_min = "min", 
                            init_points=1, n_iter=1, device="CPU", optimize_lag=True)
    ###### handover the "trainable model" to the expanding window method
    errors_cat = trainabale_model_cat.expanding_window(lagless_data=dataset, ind_f_vars=[104], col_names=[""],
                           num_factors=4, num_lags=2, opt=opt, min_window_size=570, verbose=0)
    best_model_dict = trainabale_model_cat.optimal_params
    optimal_lag = best_model_dict["lag_to_add"]
    print(best_model_dict)
    print("Sum of squared errors - cat: ", sum([e*e for e in errors_cat[0]]))


    # LightGBM
    search_space_lgbm = {
        #"n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        'lag_to_add': (0, 6),
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
    trainabale_model_lgbm = bho.LightGBM_HyperOpt(experiment_id=0, train_test_split_perc = 0.8, search_space = search_space_lgbm,
                         is_reg_task = "True", perf_metric = "RMSE", max_or_min = "min",
                         init_points=2, n_iter=4, device="CPU")

    # BaggedTree
    trainabale_model_bagged = bho.BaggedTree(n_estimators=5)

    #pred_some_stuff = utils.add_lags(dataset, 2)
    #print(trainabale_model_cat.predict_with_trained_model(pred_some_stuff))

    # Next up:
    # TODO: check if data flow is correct (self.extra_X, self.test_X, self.lagless_X, self.X)
    # TODO: add functions to trainable_model class that store and load instances of the class
        # to make sure that can stop and start the process midway! -> split along datapoints, not optimizers
        # document the parameters used at each point in time -> lags, random_seeds, ...
    # TODO: write a moving window function
    # TODO: visualize the optimization process during hyper-opt (use hyp-opt library)
    # TODO: create a procedure for evaluating and comparing performance
    # TODO: check more in detail if the implementation of BaggedTree is as in Matlab
    # TODO: write a hyper opt version of BaggedTree
    # TODO: check correctness of expanding_window() - did not understand Matlab code fully
    