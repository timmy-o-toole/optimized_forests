# This file implements a class that can be used as parent class when doing a bayesian hyperparameter optimization 
# for a machine learning algorithm. It makes use of the "bayesian-optimization" library.

import copy
import time
from typing import Tuple, List
import numpy as np
from utils import opt
from tqdm import tqdm
import csv
import json
import datetime
import os
import warnings

# Models for which child-classes are created
import xgboost as xgb
import catboost as cat
import lightgbm as lgb
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor   
from sklearn.tree import DecisionTreeClassifier

#pip install bayesian-optimization
import utils
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


class trainable_model:
    def __init__(self, experiment_id: int, model_name: str, device: str="CPU", incrementally_trainable: bool=False, train_incrementally: bool=False,
                 summary_file_path: str="summary_file_path.csv"):
        self.X = None
        self.y = None
        self.extra_X = None
        self.extra_y = None
        self.num_lags = None
        self.SUMMARY_FILE_PATH = f"{summary_file_path.split('.')[0]}_expid{experiment_id}_predvaridXYZ.csv"
        self.RESTART_FILE_PATH = f"{'/'.join(summary_file_path.split('/')[:-1])}/restart_file.json"
        self.experiment_id = experiment_id

        if os.path.exists(self.RESTART_FILE_PATH):
            with open(self.RESTART_FILE_PATH, "r") as file:
                experiments_so_far = json.load(file)
            if self.SUMMARY_FILE_PATH.split(".")[0] in experiments_so_far.keys():
                if experiments_so_far[self.SUMMARY_FILE_PATH.split(".")[0]]['status_of_experiment'] == "FINISHED":
                    raise ValueError(f"You are trying to run an experiment with an ID that has already run to completion. See {experiments_so_far[self.SUMMARY_FILE_PATH.split('.')[0]]['experiment_summary_file']} for the results. Please choose a different experiment_id when initializing the trainable_model instance.")
        self.model_name = model_name
        self.verbose = 0
        self.device = device
        self.incrementally_trainable = incrementally_trainable
        self.train_incrementally = train_incrementally
        if not incrementally_trainable and train_incrementally:
            self.train_incrementally = False
            print("Incremental training is not available or not implemented for this model.")
        self.trained_model = None

    def store_model_to_path(self, model, path: str):
        '''
        MAYBE HAS TO BE OVERWRITTEN BY CHILD CLASS
        '''
        if not os.path.exists("/".join(path.split("/")[:-1])):
            os.makedirs("/".join(path.split("/")[:-1]))
        model.save_model(path)

    def load_model_from_path(self, model, path: str):
        '''
        MAYBE HAS TO BE OVERWRITTEN BY CHILD CLASS
        '''
        model.load_model(path)

    def manage_prediction_storing(self, test_point_idx: int, horizon: int, model, model_path: str, test_point_pred: float, 
                                  test_point_error: float, in_sample_error_avg: float, in_sample_error_var: float, 
                                  used_lags: int, summary_file_path: str):
        '''
        Creates and manages a .csv file for an experiment (e.g. "trained_models/XGBoost_experiments_summary_expid1_predvarid104.csv")
        by (per datapoint idx) adding a row to the .csv that summarizes the most important information about the trained model and 
        resulting performance ("test_point_idx", "datetime", "test_point_err", "in_sample_error_avg", "in_sample_error_var", "used_lags",
        "trained_model_file", "model_params").

        Can easily be extended to track more information!
        '''
        # Create tracking .csv file in case it doesn't exist yet
        if not os.path.exists("/".join(summary_file_path.split("/")[:-1])):
            os.makedirs("/".join(summary_file_path.split("/")[:-1]))
        if not os.path.exists(summary_file_path):
            header = ["test_point_idx", "horizon", "datetime", "test_point_pred", "test_point_err", "in_sample_error_avg",
                      "in_sample_error_var", "used_lags", "trained_model_file", "model_params"]
            with open(summary_file_path, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(header)

        # Store the trained model that was used for the prediction of datapoint test_point_idx
        self.store_model_to_path(model, model_path)
        # Summarize some stats about the model that was used for the prediction
        model_summary=[test_point_idx, horizon, str(datetime.datetime.now()), test_point_pred, test_point_error, in_sample_error_avg,
                       in_sample_error_var, used_lags, model_path, model.get_params()]
        with open(summary_file_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(model_summary)

    def track_progress(self, just_finished_idx: int, is_last_idx: bool):
        '''
        Takes care of tracking experiments that are/were done in a "restart file" (e.g. restart_file.json).
        For each experiment (identified by "experiment_id") it stores information like if its "RUNNING"/"FINISHED",
        what datapoint (idx) was processed last and where the corresponding experiment .csv 
        (managed by self.manage_prediction_storing()) is stored. 

        This "restart file" is used to track the current progress and after potential interruption at an
        intermediate datapoint idx continue from that last processed idx - instead of having to iterated over
        all datapoints from the beginning again.
        '''
        mode = "r"
        if not os.path.exists("/".join(self.RESTART_FILE_PATH.split("/")[:-1])):
            os.makedirs("/".join(self.RESTART_FILE_PATH.split("/")[:-1]))
        if not os.path.exists(self.RESTART_FILE_PATH):
            experiments = {}
            with open(self.RESTART_FILE_PATH, "w") as file:
                json.dump(experiments, file, indent=4)
        else:
            with open(self.RESTART_FILE_PATH, mode) as file:
                experiments = json.load(file)
        # check if experiment is already being tracked
        if not (self.SUMMARY_FILE_PATH.split(".")[0] in experiments.keys()):
            this_experiment = {"experiment_id": self.experiment_id,
                               "experiment_summary_file": self.SUMMARY_FILE_PATH,
                               "last_handled_idx": just_finished_idx,
                               "time_of_last_update": str(datetime.datetime.now()),
                               "handled_idxs": [just_finished_idx],
                               }
        else:
            this_experiment = experiments[self.SUMMARY_FILE_PATH.split(".")[0]]
            if this_experiment["status_of_experiment"] == "FINISHED":
                raise ValueError("Trying to update finished experiment!")
            this_experiment["last_handled_idx"] = just_finished_idx
            this_experiment["handled_idxs"].append(just_finished_idx)
            this_experiment["time_of_last_update"] = str(datetime.datetime.now())
        if is_last_idx:
            this_experiment["status_of_experiment"] = "FINISHED"
        else:
            this_experiment["status_of_experiment"] = "RUNNING"
        experiments[self.SUMMARY_FILE_PATH.split(".")[0]] = this_experiment
        # save the updated .json 
        with open(self.RESTART_FILE_PATH, "w") as file:
            json.dump(experiments, file, indent=4)

    def experiment_already_finished(self, experiment_file_path: str):
        '''
        Check if an experiment is marked as being "FINISHED" in the "restart file" managed
        by self.track_progress().
        '''
        if os.path.exists(self.RESTART_FILE_PATH):
            with open(self.RESTART_FILE_PATH, "r") as file:
                experiment = json.load(file)
            if experiment_file_path in experiment.keys():
                if experiment[experiment_file_path]["status_of_experiment"] == "FINISHED":
                    return True
        return False
    
    def does_experiment_exist(self) -> Tuple[bool, int]:
        '''
        Checks if an experiment is already contained in the "restart file" managed by self.track_progress().

        The experiment that is check is the one thats identified by self.SUMMARY_FILE_PATH 
        (e.g. "trained_models/XGBoost_experiments_summary_expid1_predvarid104.csv").

        Returns:
            exp_exists (bool): contains True if there is an entry in restart file
                    for the experiment in question, else is False
            last_handled_idx (int or None): if experiment is contained (exp_exists==True),
                    gives the last idx that was handled during that experiment
        '''
        last_handled_idx = None
        exp_exists = False
        if not os.path.exists(self.RESTART_FILE_PATH):
            return False, None
        
        with open(self.RESTART_FILE_PATH, "r") as file:
            experiments = json.load(file)

        if self.SUMMARY_FILE_PATH.split(".")[0] in experiments.keys():
            exp_exists = True
            last_handled_idx = experiments[self.SUMMARY_FILE_PATH.split(".")[0]]["last_handled_idx"]
        return exp_exists, last_handled_idx
     
    def add_extraXy_to_dataset(self):
        '''
        Helperfunction for concatenating an INDIVIDUAL extra datapoint stored in self.extra_X, self.extra_y
        to the dataset stored in self.lagless_X, self.lagless_y.
        '''
        if not (self.extra_X is None or self.extra_y is None):
            self.lagless_X = np.concatenate([self.lagless_X, self.extra_X.reshape(1, self.extra_X.size)], axis=0)
            self.lagless_y = np.concatenate([self.lagless_y, self.extra_y], axis=None)
            self.extra_X = None
            self.extra_y = None
            assert self.lagless_y.shape[0] == self.lagless_X.shape[0]

    def add_lags(self, dataset: np.ndarray, lags: int, col_names: List[str]) -> np.ndarray:
        '''
        For a time-series of datapoints, adds "lags"-many lags to each datapoint.
        Since for the first "lags"-many ts points there is no data for the lags,
        the dataset that is returned contains "lags"-many datapoints less.
        '''
        # datapoints for which we are able to add the requested number of lags
        lagged_col_names = copy.deepcopy(col_names)
        if lags == 0:
            lagged_dataset = dataset[:, :]
            return lagged_dataset, lagged_col_names
        lagged_dataset = dataset[:-lags, :]
        for this_lag in range(1, lags+1):
            if this_lag == lags:
                this_subset = dataset[this_lag:, :]
            else:
                this_subset = dataset[this_lag:-lags+this_lag, :]
            lagged_dataset = np.concatenate([lagged_dataset, this_subset], axis=1)
            lagged_col_names.extend([f"{col_names[i]}_laggedBy{this_lag}" for i in range(len(col_names))])
        assert lagged_dataset.shape[1] == len(lagged_col_names)
        return lagged_dataset, lagged_col_names

    def manage_lags(self, lag_to_add: int=None):
        if lag_to_add is None:
            raise ValueError("There is no specification given regarding the lag size! Check if the search_space variable is correct.")
        self.X, self.lagged_col_names = self.add_lags(self.lagless_X, lag_to_add, self.lagless_col_names)
        
        if lag_to_add == 0:
            self.y = self.lagless_y[:]
        else:
            self.y = self.lagless_y[:-lag_to_add]
    
        assert self.X.shape[0] == self.y.shape[0]
        assert self.X.shape[1] == self.lagless_X.shape[1] * (lag_to_add+1)

    def compute_performance_stats(self, list_of_errors: List[float]) -> dict:
        '''
        Computes some performance statistics over a given list of errors.
        This is included in the experiment .csv that is managed by self.manage_prediction_storing().

        Can easily be extended to compute and return more statistics!
        '''
        average = sum(list_of_errors) / len(list_of_errors)
        variance = sum((x - average) ** 2 for x in list_of_errors) / len(list_of_errors)
        return {"average": average, "variance": variance}

    def predict_with_trained_model(self, X: np.ndarray) -> np.ndarray:
        '''
        Takes the model stored in self.trained_model and uses it to make a prediction.
        '''
        model = self.trained_model
        if model is None:
            raise ValueError(f"There is no optimal_model stored that can be used to make a prediction.")
        return self.predict(X, model)
    
    def expanding_window(self, lagless_data: np.ndarray, ind_f_vars: List[int], col_names: List[str],
                    num_factors: int, num_lags: int, opt: opt, min_window_size: int = 100, verbose: int = 0) -> np.ndarray:

        '''
        %% function that forecast fred_md data using an extending window approach
        % Input: data - Data for estimation
        %  ind_f_vars - Indices of variables to forecast
        %           c - 1, include intercept, else do not insert
        %           m - insample window size
        %           h - forecast horizon
        %          ic - information criterion: either, 'aic', 'bic', 'hq'
        %      direct - 1, if direct forecast, otherwise iterative forecast


        function results = forecast_rf(data, ind_f_vars, m, h, col_names, num_factors, num_lags, opt)
        T = size(data, 1);
        d = length(ind_f_vars);
        err_ar = nan(m, d); % oos periods, variables
        p_opt = nan(m, d); % oos periods, variables
        num_trees = 500; % Medeiros default in R package
        var_imp = cell(m, d); % oos periods, variables
        var_names_all = cell(m, d);
        '''

        # make the num_lags available as class attribute
        self.num_lags = num_lags
        self.lagged_col_names = None
        self.verbose = verbose

        # ind_f_vars - Indices of variables to forecast
        T = lagless_data.shape[0]   # T - number of points in dataset
        h = opt.h[0]   # h - forecast horizon


        self.lagless_data = lagless_data
        self.original_col_names = col_names
        self.lagless_col_names = col_names
        assert len(self.lagless_col_names) == self.lagless_data.shape[1]
        self.X = None
        self.y = None
 
        error_list_per_var = []
        for pred_var_id in ind_f_vars:

            # window grows from min_window_size 
            min_last_window_idx = min_window_size     ## TODO: +h??
            max_last_window_idx = T-h-1  # all datapoints available up to last one that can be tested

            # to whole dataset size (minus forecast horizon)
            self.SUMMARY_FILE_PATH = "_".join(self.SUMMARY_FILE_PATH.split("_")[:-1]) + f"_predvarid{pred_var_id}.csv"
            if self.experiment_already_finished(self.SUMMARY_FILE_PATH.split(".")[0]):
                print(f"Experiment ({self.SUMMARY_FILE_PATH}) already finished, skipping to next pred_var_id.")
                continue
            exp_exists, last_handled_idx = self.does_experiment_exist()
            if exp_exists:
                min_last_window_idx = last_handled_idx+1
                print("Last handled idx was: ", last_handled_idx, ". Starting from: ", min_last_window_idx)

            print(f"Now starting: {self.SUMMARY_FILE_PATH}")
            error_list = []

            # first window
            assert min_last_window_idx+h < self.lagless_data.shape[0]
            self.lagless_X = self.lagless_data[:min_last_window_idx, :]
            self.lagless_y = self.lagless_data[h:min_last_window_idx+h, pred_var_id]
            assert self.lagless_X.shape[0] == self.lagless_y.shape[0]

            # enrich the available data at this point in time with additional features
            #modified_lagless_data, dataset_col_names, transformers = utils.enrich_dataset(self.lagless_X, col_names, nr_pca_factors = 4)
            #self.lagless_X = modified_lagless_data
            #self.lagless_col_names = dataset_col_names

            # make sure model trains only on the newly defined lagless data (in self.lagless_X, self.lagless_y)
            assert self.X is None and self.y is None
            self.train_final_model()    # stores some in_sample stats into self.in_sample_stats
            
            # next datapoint in timeframe has to be predicted
            new_testpoint_idx = min_last_window_idx
            # this is the individual datapoint that is added this iteration ("one row")
            single_test_X = self.lagless_data[new_testpoint_idx, :]
            single_test_y = self.lagless_data[new_testpoint_idx+h, pred_var_id]

            
            # create a lagged datapoint to ensure correct format when handing over to predict_with_trained_model()
            used_num_lags = self.num_lags
            assert used_num_lags >= 0 and used_num_lags+1 < new_testpoint_idx
            # create lagged test datapoint for testpoint at index "new_testpoint_idx".
            lagged_test_X_orig = self.lagless_data[new_testpoint_idx-used_num_lags:new_testpoint_idx+1, :]  
            #lagged_transformed_X_orig = utils.apply_transformers_to_data(lagged_test_X_orig, transformers)
            lagged_test_X, test_lagged_col_names = self.add_lags(lagged_test_X_orig, used_num_lags, self.lagless_col_names)
            lagged_test_X = np.reshape(lagged_test_X, (1, lagged_test_X.size))
            assert test_lagged_col_names == self.lagged_col_names

            predictions = self.predict_with_trained_model(lagged_test_X)
            this_test_error = predictions - single_test_y
            error_list.extend(this_test_error)
            
            store_model_path = f"{self.SUMMARY_FILE_PATH.split('.')[0]}_models/{self.model_name}_predvarid{pred_var_id}_testpointidx{new_testpoint_idx}_expid{self.experiment_id}_{datetime.date.today()}.json"
            self.manage_prediction_storing(new_testpoint_idx+h, h, self.trained_model, store_model_path, predictions[0], this_test_error[0], 
                                           self.in_sample_stats["average"], self.in_sample_stats["variance"], 
                                           used_num_lags, self.SUMMARY_FILE_PATH)
            self.track_progress(new_testpoint_idx, min_last_window_idx+1 > max_last_window_idx)
            
            self.in_sample_stats=None
            for new_testpoint_idx in tqdm(range(min_last_window_idx+1, max_last_window_idx)):
                # make sure model trains only on the newly defined lagless data 
                self.X = None
                self.y = None

                # append extra_X and extra_y to the dataset (used to enable iterative training)
                self.add_extraXy_to_dataset()
                # add the additional features to the data
                
                # PROBLEM: self.lagless_X will get the PCA, UMAP column newly/additionally added not only once but for each idx!
                # WILL HAVE TO GET RID OF extraXY to make it less messy!
                #modified_lagless_data, dataset_col_names, transformers = utils.enrich_dataset(self.lagless_X, col_names, nr_pca_factors = 4)
                #self.lagless_X = modified_lagless_data
                #self.lagless_col_names = dataset_col_names

                # the "extension of the window" for this iteration
                self.extra_X = single_test_X    # add old test point as extra data of this iteration 
                self.extra_y = np.array([single_test_y])
                
                # Current dataset shape that is used for the model training
                if self.verbose > 0:
                    print(f"Shape of lagless dataset available in this window: {self.lagless_X.shape}")

                # train the model using only the modified self.lagless_X, self.lagless_y
                assert self.X is None and self.y is None
                assert self.extra_X is None or (self.extra_X.ndim == 1 and self.extra_y.size == 1)
                assert self.in_sample_stats is None
                self.train_final_model()

                # set a new test point
                used_num_lags = self.num_lags
                assert used_num_lags >= 0 and used_num_lags+1 <= new_testpoint_idx
                # load new datapoint that will be tested
                single_test_X = self.lagless_data[new_testpoint_idx, :]
                single_test_y = self.lagless_data[new_testpoint_idx+h, pred_var_id]
                # create a test datapoint with the lag used by the model
                lagged_test_X_orig = self.lagless_data[new_testpoint_idx-used_num_lags:new_testpoint_idx+1, :]  
                lagged_test_X, test_lagged_col_names = self.add_lags(lagged_test_X_orig, used_num_lags, self.lagless_col_names)
                lagged_test_X = np.reshape(lagged_test_X, (1, lagged_test_X.size))
                assert test_lagged_col_names == self.lagged_col_names

                # evaluate the model
                predictions = self.predict_with_trained_model(lagged_test_X)
                this_test_error = predictions - single_test_y
                error_list.extend(this_test_error)
                store_model_path = f"{self.SUMMARY_FILE_PATH.split('.')[0]}_models/{self.model_name}_predvarid{pred_var_id}_testpointidx{new_testpoint_idx}_expid{self.experiment_id}_{datetime.date.today()}.json"
                self.manage_prediction_storing(new_testpoint_idx+h, h, self.trained_model, store_model_path, predictions[0], this_test_error[0], 
                                           self.in_sample_stats["average"], self.in_sample_stats["variance"], 
                                           used_num_lags, self.SUMMARY_FILE_PATH)
                #print("Is last testpoint: ", new_testpoint_idx >= max_last_window_idx-1)
                self.track_progress(new_testpoint_idx, new_testpoint_idx >= max_last_window_idx-1)
                self.in_sample_stats = None

            error_list_per_var.append(error_list)
            self.X = None
            self.y = None
        return error_list_per_var

    def reset_model_training(self):
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS

        Resets the class to a state where it is like it hasn't done
        any training so far (e.g. self.trained_model is deleted)
        E.g. for BayerisanHyperOpt the optimizer is reset
        '''
        pass

    def train_final_model(self):
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS

        Takes features self.lagless_X and labels self.lagless_y 
        and saves a trained model to self.trained_model.

        For self.expanding_window() it should also use the latest datapoint which is
        stored in self.extra_X, self.extra_y. Can use self.add_extraXy_to_dataset()
        to add that inidividual additional datapoint to the dataset in self.lagless_X, self.lagless_y.
        
        Has to store some in-sample stats into self.in_sample_stats
        (use self.compute_performance_stats()).
        '''
        pass

    def predict(self, X: np.ndarray, model) -> np.ndarray:
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS

        Takes the given model or the model in self.trained_model and
        makes a prediction on the data given in X.
        '''
        pass


class Bayesian_Optimizer(trainable_model):
    '''
    The parent class for classes that want to implement bayesian hyperparameter optimization.
    To effectively implement a child of this class, we need to create the methods:
    black_box_function_adapter(), transform_params(), train_model() in the child class.
    For examples, see the classes: XGBoost_HyperOpt, CatBoost_HyperOpt, LightGBM_HyperOpt.
    
    Parameters that are optimized can be found in self.black_box_function_adapter().
    '''
    def __init__(self, experiment_id: int, test_split_perc: float, search_space: dict,
                 is_reg_task: bool, pred_perf_metric: str, max_or_min: str, name: str,
                 init_points: int, n_iter: int, device: str, optimize_lag: bool, summary_file_path: str,
                 incrementally_trainable: bool=False, train_incrementally: bool=False):
        
        super().__init__(experiment_id=experiment_id, model_name=name, device=device, incrementally_trainable=incrementally_trainable,
                         train_incrementally=train_incrementally, summary_file_path=summary_file_path)
        # True -> its a regression task, False -> its a classification task
        self.is_reg_task = is_reg_task

        # to decide how much of the given data is to be used for each training and
        # how much for the following performance evaluation -> better than single train set
        self.test_percentage = test_split_perc
        # Over how many different train-test splits the performance of a set of params should be evaluated
        self.amt_train_per_params = 4
        # TODO: make this k-fold cross val instead of random draws
        # TODO: does this even make sense for time-series data? How is cross-val done in TS data?

        #for the "randomized" train-test split draws
        self.random_state = 0
        
        # Defining your search space (dictionary), has to contain the key "lag_to_add"
        if optimize_lag:
            assert "lag_to_add" in search_space.keys()
        self.search_space = search_space
        
        # The BayesianOptimization object and the optimal found parameters
        self.optimizer = None
        self.optimal_params = None

        # can decide to also include the nr of lags as a hyperparameter in the optimization
        self.optimize_lag = optimize_lag

        # Track the performance at different choices of hyperparameters
        self.model_history = {}

        # The metric used to evaluate which model performed best
        self.pred_perf_metric = pred_perf_metric

        # minimize or maximize performance metric
        if not max_or_min in ["max", "min"]:
            raise ValueError(f"max_or_min has to be set to either 'max' or 'min', not: '{max_or_min}'.")
        self.max_or_min = max_or_min

        # Number of random exploration steps
        self.init_points = init_points
        # Number of bayesian optimization steps
        self.n_iter = n_iter
        # Number of bayesian optimization steps per additional datapoint iteration
        # when using incremental training
        self.incremental_train_n_iter = 3

    def reset_model_training(self):
        self.optimizer = None
        self.optimal_params = None
        self.model_history = {}
        self.trained_model = None
        
    def train_final_model(self):
        '''
        Uses the bayesian hyperpara optimization to find good parameters
        and then returns a model trained on the whole dataset using those parameters.
        '''
        if self.lagless_X.shape[0] != self.lagless_y.shape[0]:
            raise ValueError("lagless_X and lagless_y don't have the same amount of datapoints.")
        
        if self.train_incrementally:
            if not (self.extra_X is None or self.extra_y is None):
                if self.extra_X.shape[0] != self.extra_y.size:
                    raise ValueError("extra_X and extra_y don't have the same amount of datapoints.")
            self.optimize_hyperparameters(init_points=0, n_iter=self.incremental_train_n_iter)
        else: # regular training
            self.reset_model_training()
            # add additionally available datapoint to the dataset so its used during training
            self.add_extraXy_to_dataset()
            self.optimize_hyperparameters(init_points=self.init_points, n_iter=self.n_iter)
        if self.verbose > 0:
            print("Train optimal model...")
            print("--------------------------------------")
        trained_model = self.train_optimal_model()
        return trained_model

    def predict(self, X: np.ndarray, model=None) -> np.ndarray:
        '''
        Usese the given model to make a prediction on given data.
        '''
        assert X.shape[1] == self.X.shape[1]
        return model.predict(X)

    def store_bayes_optimizer(self, file_path: str):
        # TODO
        pass

    def add_to_model_history(self, trained_model, para_dict: dict, perf_score: float):
        '''
        Keeps track of parameters that have already been tested; and the resulting performances.
        '''
        self.model_history[len(self.model_history)] = {"model": "trained_model", "params": para_dict, "perf": perf_score}

    def check_para_already_tested(self, para_dict: dict) -> Tuple[bool, float]:
        '''
        Tests if a model instantiated with the given parameter dictionary has already been tested.
        '''
        for dict_id in self.model_history:
            stored_dict = self.model_history[dict_id]
            if stored_dict["params"] == para_dict:
                return True, stored_dict["perf"]
        return False, None

    def optimize_hyperparameters(self, init_points: int=None, n_iter: int=None):
        '''
        The heart of the class that calls the actual hyperparameter optimization using Bayesian Optimization.
        
        Input:
            init_points (int): How many steps of random exploration you want to perform. 
                Random exploration can help by diversifying the exploration space.
            n_iter (int): How many steps of bayesian optimization you want to perform. 
                The more steps the more likely to find a good maximum you are.
        
        Output:
            Sets the self.optimal_params field to the best found parameters.
        '''

        # print("Now right before setting the optimal_params field")
        if init_points is None:
            init_points = self.init_points
        if n_iter is None:
            n_iter = self.n_iter

        self.optimizer = BayesianOptimization(f = self.black_box_function_adapter,
                                pbounds = self.search_space,
                                random_state = 1,
                                verbose = 0)
        
        '''if (self.train_incrementally and self.model_history is not None 
            and self.extra_X is not None and self.extra_y is not None):
            print("Using incremental training...")
            if self.extra_X.shape[0] != self.extra_y.size:
                raise ValueError("X and y don't have the same amount of datapoints.")

            # take all models from self.model_history and train them only with extra_X
            for dict_id in self.model_history:
                stored_dict = self.model_history[dict_id]
                stored_dict["model"] = self.train_model(stored_dict["model"], self.extra_X, self.extra_y)
                # reevaluate the performance of the additionally trained model
                _, test_X, _, test_y = train_test_split(self.X, self.y, test_size=0.3)
                pred = stored_dict["model"].predict(test_X)
                stored_dict["perf_score"] += self.prediction_performance_score(test_y, pred)
                
                # feed the evaluated parameter/performance pairs into the bayes optimizer
                params_arr = self.optimizer._space._as_array(stored_dict["params"])
                self.optimizer._space.register(params_arr, stored_dict["perf_score"])
                print(self.optimizer._space.__len__())'''
        
        # use the optimizer find best parameters
        self.optimizer.maximize(init_points = init_points, n_iter = n_iter)
        # print(self.optimizer.max["params"])
        self.optimal_params = self.transform_params(self.optimizer.max["params"])
    
    def instantiate_model(self, cor_params: dict):
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS
        '''
        pass

    def train_model(self, model, extra_X: np.ndarray, extra_y: np.ndarray):
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS
    
        Takes parameters, passes them to the individual ML model and returns the trained ML model.
        '''
        pass
    
    def transform_params(self, input_params: dict) -> dict:
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS
        
        Called by black_box_function
        Function that makes sure that the shape of params fed to the "to hyperparameter optimize ML Algorithm"
        is valid. Some arguments for example can only be given to the ML Algo as integers;
        the bayesian-optimization library only works on floats. (float -> int etc.)

        Has to modify the value stored under key "lag_to_add" to be of type int.
        '''
        pass

    def black_box_function_adapter(self):
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS
        
        A function that takes the exact parameter names as the to train ML algo.
        These are put into a dictionary and handed over to black_box_function().
        '''
        pass

    def train_new_model(self, params: dict, X: np.ndarray, y: np.ndarray):
        '''
        Instantiates a model using the given params dictionary.
        Then trains it on the given X, y data.
        '''
        #cor_params = self.transform_params(params)
        model = self.instantiate_model(params)
        return self.train_model(model, X, y)

    def black_box_function(self, cor_params):
        '''
        Train a model and evaluate its performance (called by optimize_hyperparameters()).
        
        Transform params to be accaptable hyperparameters for the Classifier.
        Since for pbounds it is not possible to specify that some parameters are integers.
        '''
        was_tested, perf_was = self.check_para_already_tested(cor_params)
        if was_tested:
            return perf_was

        # 0. modify the data to contain the lag specified in the dictionary
        if self.optimize_lag:
            lag_to_add = cor_params["lag_to_add"]
            self.num_lags = lag_to_add
        else:
            lag_to_add = self.num_lags
        # add lags to the data: new data will contain "lag_to_add"-many points less than data
        self.manage_lags(lag_to_add=lag_to_add) # -> set: self.X, self.y
        assert self.X.shape[1]/self.lagless_X.shape[1] == lag_to_add+1
        if self.verbose > 0:
            print(f"This iteration - lag: {lag_to_add} - dataset format: {self.X.shape}")
        #print(f"Modified lag to be {self.X.shape[1]/self.lagless_X.shape[1] -1}, since given {lag_to_add}")
        
        # We train the algorithm self.amt_train_per_params many times on the given params
        # with different train-test-splits to counteract overfitting.
        sum_perf_score = 0
        model_params = copy.deepcopy(cor_params)
        model_params.pop("lag_to_add")
        for _ in range(self.amt_train_per_params):
            # 1. draw a random test, train split from the given data
            train_X, test_X, train_y, test_y = train_test_split(self.X, self.y,
                                                                test_size = self.test_percentage,
                                                                random_state = self.random_state)
            # 2. train a model using the given params
            # print(f"Working on data of shape: {train_X.shape}")
            print(model_params)
            trained_model = self.train_new_model(params = model_params, X = train_X, y = train_y)

            # 3. make it predict unseen test data
            model_pred = trained_model.predict(test_X)

            # 4. evaluate the performance of the prediction
            perf_score = self.prediction_performance_score(test_y, model_pred)

            # since library only can maximize scores, in case we want to minimize we negate the performance metric
            if self.max_or_min == "min":
                perf_score = -perf_score
            
            sum_perf_score += perf_score
            # print("Performance for the "+ str(i) + " iteration: " + str(perf_score))
            self.random_state = self.random_state + 1
            
        # The performance of a set of hyperparameters for an ML algo is the average performance over multiple train-test splits
        ret_perf_score = sum_perf_score/self.amt_train_per_params    
        # TODO: careful, the handed over model is possibly only trained on parts of the data
        self.add_to_model_history(trained_model, cor_params, ret_perf_score)
        return ret_perf_score
    
    def prediction_performance_score(self, true_y, pred_y):
        '''
        This function evalutes the performance of a model that is used to evaluate which hyperparameters
        are the best.
        
        Called by self.black_box_function().

        Input:
            pred_y (np.ndarray): a "vector" of predictions
            true_y (np.ndarray): a "vector" of "true value" that should have been predicted
            metric (str): metric to use for performance evaluation -> "RMSE" or "MSE"

        Output:
            Chosen performance metric rounded to 8 digits
        '''

        # Possible stats to look at for the performance of the evaluated model
        if self.is_reg_task:
            if self.pred_perf_metric == "RMSE":
                perf_score = mean_squared_error(true_y, pred_y, squared=False)
            elif self.pred_perf_metric == "MSE":
                perf_score = mean_squared_error(true_y, pred_y, squared=True)
            elif self.pred_perf_metric == "overfit/underfit":
                # regression between the predictions and true values
                # look at distance of slope to 1
                raise ValueError("Not implemented yet!")
            else: 
                raise ValueError("Entered '"+self.pred_perf_metric+"' as performance metric. See Bayesian_Optimizer.prediction_performance_score() for available metrics")
        else:
            if self.pred_perf_metric == "accuracy":
                perf_score = accuracy_score(true_y, pred_y)
            else:
                raise ValueError("Entered "+self.pred_perf_metric+"as performance metric. See Bayesian_Optimizer.prediction_performance_score() for available metrics")
        return round(perf_score, 8)
    
    def train_optimal_model(self, train_features=None, train_labels=None):
        '''
        Can only be used after running self.optimize_hyperparameters().
        Then it takes the parameters specified in self.optimal_params and trains the ML model.
        
        Input:
            train_features: features on which the algorithm should be trained.
            train_labels: the labels corresponding to the features in train_features.
        
        Output:
            self.optimal_params == None:
                In this case we cant train the model, since no optimal parameters are specified.
            self.optimal_params != None:
                This method returns a model that is trained on the given data train_features, train_returns. 
                If no data is given as input we take the whole data set stored in self.features, self.returns.
        '''
        # check if already found some optimal parameters
        if self.optimal_params is None:
            print("There is no optimal set of parameters yet. Maybe you still have to run self.optimize_hyperparameters().")
            return None
        else:
            # check if train-data is given or not
            if train_features is None:
                optimal_params = copy.deepcopy(self.optimal_params)
                if self.optimize_lag:
                    used_lag = optimal_params["lag_to_add"]
                    self.manage_lags(used_lag)
                    self.num_lags = used_lag
                optimal_params.pop("lag_to_add")
                train_features = self.X
                train_labels = self.y
            trained_model = self.train_new_model(optimal_params, train_features, train_labels)
            self.trained_model = copy.deepcopy(trained_model)

            # compute and store some stats over the training data
            predictions_in_sample = trained_model.predict(train_features)
            in_sample_errors = list(predictions_in_sample - train_labels)
            self.in_sample_stats = self.compute_performance_stats(in_sample_errors)
            return trained_model


class CatBoost_HyperOpt(Bayesian_Optimizer):
    '''
    A class that inherits from the Bayesian_Optimizer class and implements the bayesian hyperparameter
    optimization of a CatBoost decision tree.
    '''

    def __init__(self, experiment_id: int, test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "RMSE", max_or_min: str = "min",
                 init_points: int = 2, n_iter: int = 20, device: str="CPU",
                 optimize_lag: bool=False, summary_file_path: str=f"trained_models/CatBoost_experiments_summary.csv"
                 ):
        self.is_reg_task = is_reg_task
        super().__init__(experiment_id=experiment_id, test_split_perc=test_split_perc,
                         search_space=search_space, is_reg_task=self.is_reg_task, 
                         pred_perf_metric=perf_metric, max_or_min=max_or_min, name="CatBoost_HyperOpt",
                         init_points=init_points, n_iter=n_iter, device=device, 
                         optimize_lag=optimize_lag, summary_file_path=summary_file_path,
                         incrementally_trainable=False, train_incrementally=False)

    def instantiate_model(self, cor_params: dict):
        if self.is_reg_task:
            model = cat.CatBoostRegressor(**cor_params, verbose=0, task_type=self.device, loss_function=self.pred_perf_metric)
        else:
            model = cat.CatBoostClassifier(**cor_params, verbose=0, task_type=self.device, loss_function=self.pred_perf_metric)
        return model

    def train_model(self, model, extra_X: np.ndarray, extra_y: np.ndarray):
        '''
        This trains the deployed ML algorithm, given a set of parameters and train data (features, labels).
        '''
        model.fit(extra_X, extra_y)
        return model
        
    def transform_params(self, input_params: dict) -> dict:
        '''
        Makes sure that the parameters passed to the actual ML algorithm are valid inputs.
        This includes for example turning floats into integers, and possible translating that integer
        as a categorical variable.
        '''
        ret_params = copy.deepcopy(input_params)
        # make sure the lag is set to an integer
        ret_params["lag_to_add"] = int(ret_params["lag_to_add"])

        # model params
        ret_params["iterations"] = int(ret_params["iterations"])
        ret_params["depth"] = int(ret_params["depth"])
        ret_params["border_count"] = int(ret_params["border_count"])
        ret_params["l2_leaf_reg"] = int(ret_params["l2_leaf_reg"])
        return ret_params
    
    def black_box_function_adapter(self, lag_to_add, iterations, depth, learning_rate, random_strength, bagging_temperature, border_count, l2_leaf_reg):
        '''
        Implement an adapter that translates the parameter names and hands them to the real black_box_function() that
        is inherited from the parent class (Bayesian_Optimizer).
        '''
        # put all given parameters into a dictionary
        wrong_params_dict = {'lag_to_add': lag_to_add,
                             'iterations': iterations,
                             'depth': depth,
                             'learning_rate': learning_rate,
                             'random_strength': random_strength,
                             'bagging_temperature': bagging_temperature,
                             'border_count': border_count,
                             'l2_leaf_reg': l2_leaf_reg
                             }
        cor_params = self.transform_params(wrong_params_dict)
        perf_score = self.black_box_function(cor_params)
        return perf_score
    

class LightGBM_HyperOpt(Bayesian_Optimizer):
    '''
    A class that inherits from the Bayesian_Optimizer class and implements the bayesian hyperparameter
    optimization of a LightGBM decision tree.
    '''    
    def __init__(self, experiment_id: int, test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = True, perf_metric: str = "RMSE", max_or_min: str = "min",
                 init_points: int = 2, n_iter:int = 20, device: str="CPU",
                 optimize_lag: bool=False, summary_file_path: str="trained_models/LightGBM_experiments_summary.csv"
                ):
        self.is_reg_task = is_reg_task
        super().__init__(experiment_id=experiment_id, test_split_perc=test_split_perc, search_space=search_space,
                         is_reg_task=self.is_reg_task, pred_perf_metric=perf_metric, max_or_min=max_or_min, name="LightGBM_HyperOpt",
                         init_points=init_points, n_iter=n_iter, device=device,
                         optimize_lag=optimize_lag, summary_file_path=summary_file_path,
                         incrementally_trainable=False, train_incrementally=False)

    def instantiate_model(self, cor_params: dict):
        print(cor_params)
        if self.is_reg_task:
            model = lgb.LGBMRegressor(**cor_params, verbose=0)
        else:
            model = lgb.LGBMClassifier(**cor_params, verbose=0)
        return model
    
    def train_model(self, model, extra_X: np.ndarray, extra_y: np.ndarray):
        '''
        This trains the deployed ML algorithm, given a set of parameters and train data (features, labels).
        '''
        model.fit(extra_X, extra_y)
        return model

    def transform_params(self, input_params: dict) -> dict:
        '''
        Makes sure that the parameters passed to the actual ML algorithm are valid inputs.
        This includes for example turning floats into integers, and possible translating that integer
        as a categorical variable.
        '''

        ret_params = copy.deepcopy(input_params)
        ret_params["boosting_type"] = "rf"

        # make sure the lag is set to an integer
        if "lag_to_add" in ret_params.keys():
            ret_params["lag_to_add"] = int(ret_params["lag_to_add"])
        if "n_estimators" in ret_params.keys():
            ret_params["n_estimators"] = int(ret_params["n_estimators"])
        if "max_depth" in ret_params.keys():
            ret_params["max_depth"] = int(ret_params["max_depth"])
        if "bagging_freq" in ret_params.keys():
            ret_params["bagging_freq"] = int(ret_params["bagging_freq"])
        # model params
        if "num_leaves" in ret_params.keys():
            ret_params["num_leaves"] = int(ret_params["num_leaves"])
        if "min_data_in_leaf" in ret_params.keys():
            ret_params["min_data_in_leaf"] = int(ret_params["min_data_in_leaf"])
        return ret_params

    def black_box_function_adapter(self, lag_to_add, n_estimators, learning_rate, max_depth, bagging_freq, bagging_fraction):
        '''
        Implement an adapter that translates the parameter names and hands them to the real black_box_function() that
        is inherited from the parent class (Bayesian_Optimizer).
        '''
        # put all given parameters into a dictionary
        wrong_params_dict = {'lag_to_add': lag_to_add,
                  'n_estimators': n_estimators,
                  'learning_rate': learning_rate,
                  'max_depth': max_depth,
                  "bagging_freq": bagging_freq,
                  "bagging_fraction": bagging_fraction,
                  #'num_leaves': num_leaves,
                  #'min_data_in_leaf': min_data_in_leaf,
                  #'lambda_l1': lambda_l1,
                  #'lambda_l2': lambda_l2,
                  #'min_gain_to_split': min_gain_to_split,
                  #'bagging_fraction': bagging_fraction,
                  #'feature_fraction': feature_fraction
                 }
        cor_params = self.transform_params(wrong_params_dict)
        perf_score = self.black_box_function(cor_params)
        return perf_score

    def black_box_function_adapter_large(self, lag_to_add, n_estimators, learning_rate, num_leaves, max_depth, min_data_in_leaf,
                                   lambda_l1, lambda_l2, min_gain_to_split, bagging_fraction, feature_fraction):
        '''
        Implement an adapter that translates the parameter names and hands them to the real black_box_function() that
        is inherited from the parent class (Bayesian_Optimizer).
        '''
        # put all given parameters into a dictionary
        wrong_params_dict = {'lag_to_add': lag_to_add,
                  'n_estimators': n_estimators,
                  'learning_rate': learning_rate,
                  'max_depth': max_depth,
                  'num_leaves': num_leaves,
                  'min_data_in_leaf': min_data_in_leaf,
                  'lambda_l1': lambda_l1,
                  'lambda_l2': lambda_l2,
                  'min_gain_to_split': min_gain_to_split,
                  'bagging_fraction': bagging_fraction,
                  'feature_fraction': feature_fraction
                 }
        cor_params = self.transform_params(wrong_params_dict)
        perf_score = self.black_box_function(cor_params)
        return perf_score


class XGBoost_HyperOpt(Bayesian_Optimizer):
    '''
    A class that inherits from the Bayesian_Optimizer class and implements the bayesian hyperparameter
    optimization of a XGBoost decision tree.
    
    GPU: conda install py-xgboost-gpu (package not available for conda in windows)
    '''
    def __init__(self,  experiment_id: int, test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = True, perf_metric: str = "RMSE", max_or_min: str = "min",
                 init_points: int = 2, n_iter: int = 20, device: str="CPU",
                 optimize_lag: bool=False, summary_file_path: str="trained_models/XGBoost_experiments_summary.csv"
                ):
        self.is_reg_task = is_reg_task
        super().__init__(experiment_id=experiment_id, test_split_perc=test_split_perc, search_space=search_space,
                         is_reg_task=self.is_reg_task, pred_perf_metric=perf_metric, max_or_min=max_or_min, name="XGBoost_HyperOpt",
                         init_points=init_points, n_iter=n_iter, device=device, summary_file_path=summary_file_path,
                         optimize_lag=optimize_lag, incrementally_trainable=False, train_incrementally=False)
  
    def instantiate_model(self, cor_params: dict):
        tree_method = "hist"
        if self.device == "GPU":
            tree_method = "gpu_hist"
        if self.is_reg_task:
            model = xgb.XGBRegressor(**cor_params, tree_method=tree_method)
        else:
            model = xgb.XGBClassifier(**cor_params, tree_method=tree_method)
        return model

    def train_model(self, model, extra_X: np.ndarray, extra_y: np.ndarray):
        '''
        This trains the deployed ML algorithm, given a set of parameters and train data (features, labels).
        '''
        model.fit(extra_X, extra_y)
        return model

    def transform_params(self, input_params: dict) -> dict:
        '''
        Makes sure that the parameters passed to the actual ML algorithm are valid inputs.
        This includes for example turning floats into integers, and possible translating that integer
        as a categorical variable.
        '''
        ret_params = copy.deepcopy(input_params)
        # make sure the lag is set to an integer
        ret_params["lag_to_add"] = int(ret_params["lag_to_add"])

        # model params        
        # TODO: "objective"???
        ret_params["eval_metric"] = "aucpr" # TODO
        ret_params["booster"] = "gbtree"    # TODO
        ret_params["max_depth"] = int(ret_params["max_depth"])
        if "lambda_" in ret_params.keys():
            ret_params["lambda"] = ret_params["lambda_"]
            ret_params.pop("lambda_")

        '''
        ret_params = {'lambda': input_params["lambda_"],
                  'alpha': input_params["alpha"],
                  'max_depth': input_params["max_depth"],
                  'eta': input_params["eta"],
                  'gamma': input_params["gamma"],
                  'max_depth': int(input_params["max_depth"]),
                  'booster': "gbtree",      # TODO
                  'eval_metric': "aucpr"    # TODO
                 }
        '''
        return ret_params

    def black_box_function_adapter(self, lag_to_add, lambda_, alpha, max_depth, eta, gamma):
        '''
        Implement an adapter that summarizes and transforms the parameters
        and hands them to the real black_box_function() that is inherited
        from the parent class (Bayesian_Optimizer).
        '''
        # put all given parameters into a dictionary
        wrong_params_dict = {'lag_to_add': lag_to_add,
                             'lambda_': lambda_,
                             'alpha': alpha,
                             'max_depth': max_depth,
                             'eta': eta,
                             'gamma': gamma,
                             }
        cor_params = self.transform_params(wrong_params_dict)
        perf_score = self.black_box_function(cor_params)
        return perf_score


class BaggedTree_HyperOpt(Bayesian_Optimizer):
    '''
    A class that inherits from the Bayesian_Optimizer class and implements the bayesian hyperparameter
    optimization of bagged decision trees.
    '''
    def __init__(self,  experiment_id: int, test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = True, perf_metric: str = "RMSE", max_or_min: str = "min",
                 init_points: int = 2, n_iter: int = 20, device: str="CPU",
                 optimize_lag: bool=False, summary_file_path: str="trained_models/BaggedTree_HyperOpt_experiments_summary.csv"
                ):
        self.is_reg_task = is_reg_task
        super().__init__(experiment_id=experiment_id, test_split_perc=test_split_perc, search_space=search_space,
                         is_reg_task=self.is_reg_task, pred_perf_metric=perf_metric, max_or_min=max_or_min, name="BaggedTree_HyperOpt",
                         init_points=init_points, n_iter=n_iter, device=device, summary_file_path=summary_file_path,
                         optimize_lag=optimize_lag, incrementally_trainable=False, train_incrementally=False)
  
    def instantiate_model(self, cor_params: dict):
        '''
        No hyperparameters that are necessary when creating BaggedTrees.
        Could extend to optimize hyperparameters of the "DecisionTreeRegressor/Classifier".

        Still need to formaly receive the cor_params dictionary due to function definition in parent class.
        '''
        if self.is_reg_task:
            model = BaggingRegressor(estimator=DecisionTreeRegressor(), **cor_params, random_state=0)
        else:
            model = BaggingClassifier(estimator=DecisionTreeClassifier(), **cor_params, random_state=0)
        return model

    def train_model(self, model, extra_X: np.ndarray, extra_y: np.ndarray):
        '''
        This trains the deployed ML algorithm, given a set of parameters and train data (features, labels).
        '''
        model.fit(extra_X, extra_y)
        return model

    def transform_params(self, input_params: dict) -> dict:
        '''
        Makes sure that the parameters passed to the actual ML algorithm are valid inputs.
        This includes for example turning floats into integers, and possible translating that integer
        as a categorical variable.

        In the case of BaggedTrees we might only optimize the used lag.
        TODO: if only optimizing over lag, might as well try all reasonable lags and not use HyperOpt.
        '''
        ret_params = copy.deepcopy(input_params)
        # make sure the lag is set to an integer
        ret_params["lag_to_add"] = int(ret_params["lag_to_add"])
        ret_params["n_estimators"] = int(ret_params["n_estimators"])
        return ret_params

    def black_box_function_adapter(self, lag_to_add, n_estimators, max_features, max_samples):
        '''
        Implement an adapter that summarizes and transforms the parameters
        and hands them to the real black_box_function() that is inherited
        from the parent class (Bayesian_Optimizer).
        '''
        # put all given parameters into a dictionary
        wrong_params_dict = {'lag_to_add': lag_to_add,
                             'n_estimators': n_estimators,
                             'max_features': max_features,
                             'max_samples': max_samples,
                             }
        cor_params = self.transform_params(wrong_params_dict)
        perf_score = self.black_box_function(cor_params)
        return perf_score
    
    def store_model_to_path(self, model, path: str):
        if not os.path.exists("/".join(path.split("/")[:-1])):
            os.makedirs("/".join(path.split("/")[:-1]))
        warnings.warn("Model saving for BaggedTree_HyperOpt has not been implemented yet.")
        #model.save_model(path)
    

class BaggedTree(trainable_model):
    '''
    A simple BaggedTree class that has no parameters that are optimized.
    Thus, less costly to run.
    '''
    def __init__(self, experiment_id: int, n_estimators: int = 500,
                 summary_file_path: str=f"trained_models/BaggedTree_experiments_summary.csv"):
        self.n_estimators = n_estimators
        self.num_lags = None
        super().__init__(experiment_id=experiment_id, device="CPU", model_name="BaggedTree",                         
                         incrementally_trainable=False, train_incrementally=False,
                         summary_file_path=summary_file_path)

    def reset_model_training(self):
        self.trained_model = None

    def train_final_model(self):
        if self.lagless_X.shape[0] != self.lagless_y.shape[0]:
            raise ValueError("X and y don't have the same amount of datapoints.")
        
        if self.train_incrementally:
            raise ValueError("NOT IMPLEMENTED")
            if not (self.extra_X is None or self.extra_y is None):
                if self.extra_X.shape[0] != self.extra_y.size:
                    raise ValueError("X and y don't have the same amount of datapoints.") 
        else:
            self.reset_model_training()
            self.add_extraXy_to_dataset()
            self.manage_lags(lag_to_add=self.num_lags) # -> set: self.X, self.y
            model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=self.n_estimators, random_state=0)
            model.fit(self.X, self.y)
            predictions_in_sample = model.predict(self.X)
            in_sample_errors = list(predictions_in_sample - self.y)
            self.in_sample_stats = self.compute_performance_stats(in_sample_errors)
        self.trained_model=model

    def predict(self, X: np.ndarray, model) -> np.ndarray:
        return model.predict(X)
    
    def store_model_to_path(self, model, path: str):
        if not os.path.exists("/".join(path.split("/")[:-1])):
            os.makedirs("/".join(path.split("/")[:-1]))
        warnings.warn("Model saving for BaggedTree has not been implemented yet.")
        #model.save_model(path)
        