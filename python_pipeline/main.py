import os
import numpy as np
import utils
import BayesianHyperOptClasses as bho
import warnings

if __name__ == "__main__":

    # load the data
    data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv")  # Inflation idx: 103
    #data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_trans_outlier_and_na_removed.csv")    # Inflation idx: 104
    dataset = np.loadtxt(data_file_path, delimiter=";", encoding='utf-8-sig')
    # TODO: need to reverse the time series (letzte reihe ist der neuste datenpunkt!)
    col_names = [str(i) for i in range(dataset.shape[1])]

    print("Data points:\t", dataset.shape[0])
    print("Data dimensions:", dataset.shape[1])
    print("--------------------------------------")

    #enriched_dataset, enriched_col_names, transformers = utils.enrich_dataset(dataset, col_names,
    #                                                            nr_pca_factors = 4)
    
    opt = utils.opt()

    enrich_dataset_settings = {"nr_pca_factors": 4,
                               "nr_umap_dim": 0,
                               "min_mean_max_idx": 103,
                               "fft_idx": -1,
                               "ema_idx": 103,
                               "stl_idx": 103}
    
    '''
    ### BaggedTree
    trainabale_model_bagged = bho.BaggedTree(experiment_id=434, n_estimators=500)
    errors_bagged = trainabale_model_bagged.expanding_window(lagless_data=dataset, ind_f_vars=[103], col_names=col_names,
                           num_lags=4, opt=opt, min_window_size=383, verbose=0, enrich_dataset_settings=enrich_dataset_settings)
    exit()
    '''

    '''
    ### BaggedTree HyperOpt
    search_space_bagged = {'lag_to_add': (0, 6),
                           'n_estimators': (3, 10),
                           'max_features': (0.1, 0.7),  # max nr of features to train each estimator on (in percent)
                           'max_samples': (0.1, 1.0),   # max nr of samples to train each estimator on (in percent)
                           }
    trainabale_model_bagged = bho.BaggedTree_HyperOpt(experiment_id=99, train_test_split_perc=0.8,
                                                      search_space=search_space_bagged)
    trainabale_model_bagged.expanding_window(lagless_data=enriched_dataset, ind_f_vars=[104, 105], col_names=[""],
                                             num_factors=4, num_lags=2, opt=opt, min_window_size=565, verbose=0)
    '''

    ### XGBoost HyperOpt
    search_space_xgb = {'lag_to_add': (1, 5),
                        "lambda_": (1e-5, 100), # L2 reg
                        "alpha": (1e-5, 100),   # L1 reg
                        "max_depth": (3, 11),
                        "eta": (0.001, 0.1),     # learning rate
                        "gamma": (0, 2),
                        "min_child_weight": (5, 20),
                        "subsample": (0.01, 0.4),
                        "colsample_bytree": (0.01, 0.4)
                        }
    trainabale_model_xgb = bho.XGBoost_HyperOpt(experiment_id=204, test_split_perc = 0.1, search_space = search_space_xgb,
                            is_reg_task = True, perf_metric = "AIC", max_or_min = "min",
                            init_points=12, n_iter=100, device="CPU", optimize_lag=False)
    ###### testing XGBoost
    errors_xgb = trainabale_model_xgb.expanding_window(lagless_data=dataset, ind_f_vars=[103], col_names=col_names,
                                                       num_lags=4, opt=opt, min_window_size=383, verbose=0,
                                                       enrich_dataset_settings=enrich_dataset_settings)
    print("Sum of squared errors - XGBoost: ", [sum([e*e for e in errors_per_var]) for errors_per_var in errors_xgb])

    exit()
    '''
    # LightGBM HyperOpt
    search_space_lgbm = {
        "lag_to_add": (0, 6),
        "n_estimators": (10, 15),
        "learning_rate": (0.01, 0.3), # float
        "max_depth": (3, 15), #int
        "bagging_freq": (5, 15),
        "bagging_fraction": (0., 1.),
        #"num_leaves": (10, 3000), # int
        #"min_data_in_leaf": (5, 50), #int
        #"lambda_l1": (0, 100), #int
        #"lambda_l2": (0, 100), #ints
        #"min_gain_to_split": (0, 15), #float
        #"bagging_fraction": (0.2, 0.95), #float
        #"bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        #"feature_fraction": (0.1, 0.95) #float
    }
    trainabale_model_lgbm = bho.LightGBM_HyperOpt(experiment_id=0, test_split_perc = 0.2, search_space = search_space_lgbm,
                         is_reg_task = True, perf_metric = "MSE", max_or_min = "min",
                         init_points=2, n_iter=4, device="CPU")
    ###### testing LightGBM
    errors_xgb = trainabale_model_lgbm.expanding_window(lagless_data=dataset, ind_f_vars=[104], col_names=col_names,
                                                       num_factors=4, num_lags=2, opt=opt, min_window_size=565, verbose=0)
    print("Sum of squared errors - LightGBM: ", [sum([e*e for e in errors_per_var]) for errors_per_var in errors_xgb])
    '''

    ''''''
    ### CatBoost HyperOpt
    search_space_cat = {'lag_to_add': (0, 6),
                        'iterations': (100, 1000),
                        'depth': (2, 7),
                        'learning_rate': (0.01, 1.0),
                        'random_strength': (1e-9, 10),
                        'bagging_temperature': (0.0, 1.0),
                        'border_count': (1, 255),
                        'l2_leaf_reg': (2, 30),
                        }
    trainabale_model_cat = bho.CatBoost_HyperOpt(experiment_id=0, test_split_perc = 0.2, search_space = search_space_cat,
                            is_reg_task = True, perf_metric = "RMSE", max_or_min = "min", 
                            init_points=1, n_iter=1, device="CPU", optimize_lag=True)
    

    ###### handover the "trainable model" to the expanding window method
    errors_cat = trainabale_model_cat.expanding_window(lagless_data=dataset, ind_f_vars=[104, 105], col_names=col_names,
                           num_factors=4, num_lags=2, opt=opt, min_window_size=570, verbose=0)
    print("Sum of squared errors - cat: ", [sum([e*e for e in errors_per_var]) for errors_per_var in errors_cat])



    #pred_some_stuff = utils.add_lags(dataset, 2)
    #print(trainabale_model_cat.predict_with_trained_model(pred_some_stuff))

    # Next up:
    # TODO: create a procedure for evaluating and comparing performance
        # -> show actual development and predictions in the same graph.
    # TODO: check if data flow is correct (self.extra_X, self.test_X, self.lagless_X, self.X)
    # TODO: write a moving window function
    # TODO: include fields in the trainable_model class that contain all predictions, errors etc.
    # TODO: visualize the optimization process during hyper-opt (use hyp-opt library)
    # TODO: check more in detail if the implementation of BaggedTree is as in Matlab
    # TODO: check correctness of expanding_window() - did not understand Matlab code fully
    # TODO: consider using e.g. XGBoost_HyperOpt to find the optimal parameters for a tree algo. 
        # -> then use these paras and that model and train a Bagged version of multiple of these 
        # on different parameter and datapoint subsets. Should be better than a straight forward bagged regressor.
        # -> alternatively: Every Regressor in the BaggedRegressor is optimized on that individual subset (HyperOpt)


    '''
    # Testing the lagging the data code
    example_dataset = np.array([[i, i*10, i*100] for i in range(1, 10)])
    example_col_names = ["i", "i*10", "i*100"]
    print(example_dataset.shape)
    assert example_dataset.shape[1] == len(example_col_names)
    print(example_dataset)
    lagged_data, lagged_names = trainabale_model_xgb.add_lags(example_dataset, 3, example_col_names)
    print(lagged_data)
    print(lagged_data.shape)
    print(lagged_names)
    print(len(lagged_names))
    assert 0 == 1
    '''
