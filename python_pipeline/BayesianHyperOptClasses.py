# This file implements a class that can be used as parent class when doing a bayesian hyperparameter optimization 
# for a machine learning algorithm. It makes use of the "bayesian-optimization" library.

import copy
import time
from typing import Tuple, List
import numpy as np
from utils import opt, add_lags
from tqdm import tqdm

# Models that are hyper-para optimized
import xgboost as xgb
import catboost as cat
import lightgbm as lgb
from sklearn.ensemble import BaggingRegressor

#pip install bayesian-optimization
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

class trainable_model:
    def __init__(self, device: str="CPU", incrementally_trainable: bool=False, train_incrementally: bool=False):
        self.X = None
        self.y = None
        self.extra_X = None
        self.extra_y = None
        self.num_lags = None

        self.verbose = 0
        self.device = device
        self.incrementally_trainable = incrementally_trainable
        self.train_incrementally = train_incrementally
        if not incrementally_trainable and train_incrementally:
            self.train_incrementally = False
            print("Incremental training is not available or not implemented for this model.")
        
        self.trained_model = None

    def add_extraXy_to_dataset(self):
        if not (self.extra_X is None or self.extra_y is None):
            self.lagless_X = np.concatenate([self.lagless_X, self.extra_X.reshape(1, self.extra_X.size)], axis=0)
            self.lagless_y = np.concatenate([self.lagless_y, self.extra_y], axis=None)
            self.extra_X = None
            self.extra_y = None
            assert self.lagless_y.shape[0] == self.lagless_X.shape[0]

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

        Takes features X and labels y and returns a trained model 
        that is saved to self.trained_model.
        '''
        pass

    def predict(self, X: np.ndarray, model) -> np.ndarray:
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS

        Takes the given model or the model in self.trained_model and
        makes a prediction on the data given in X.
        '''
        pass
    
    def predict_with_trained_model(self, X: np.ndarray) -> np.ndarray:
        model = self.trained_model
        if model is None:
            raise ValueError(f"There is no optimal_model stored that can be used to make a prediction.")
        return self.predict(X, model)
    
    def add_lags(self, dataset: np.ndarray, lags: int) -> np.ndarray:
        '''
        For a time-series of datapoints, adds "lags"-many lags to each datapoint.
        Since for the first "lags"-many ts points there is no data for the lags,
        the dataset that is returned contains "lags"-many datapoints less.
        '''
        # datapoints for which we are able to add the requested number of lags
        lagged_dataset = dataset[lags:, :]
        for this_lag in range(lags):
            # for lags = 3 we get this_lag = 0 -> 1 -> 2
            this_subset = dataset[this_lag:-lags+this_lag, :]
            lagged_dataset = np.concatenate([lagged_dataset, this_subset], axis=1)
        return lagged_dataset

    def manage_lags(self, lag_to_add):
        if lag_to_add is None:
            raise ValueError("There is no specification given regarding the lag size! Check if the search_space variable is correct.")
        self.X = self.add_lags(self.lagless_X, lag_to_add)
        self.y = self.lagless_y[lag_to_add:]
        assert self.X.shape[0] == self.y.shape[0]
        assert self.X.shape[1] == self.lagless_X.shape[1] * (lag_to_add+1)

    def expanding_window(self, lagless_data: np.ndarray, ind_f_vars: List[int], col_names: List[str],
                    num_factors: int, num_lags: int, opt: opt, min_window_size: int = 100, verbose: int = 0) -> np.ndarray:

        # make the num_lags available as class attribute
        self.num_lags = num_lags
        self.verbose = verbose

        # ind_f_vars - Indices of variables to forecast
        T = lagless_data.shape[0]   # T - number of points in dataset
        #TODO: m = opt.m   # m - insample window size, WHY???
        h = opt.h[0]   # h - forecast horizon
        min_last_window_idx = min_window_size+h
        max_last_window_idx = T-h-1  # all datapoints available up to last one that can be tested

        self.lagless_data = lagless_data
        self.X = None
        self.y = None
 
        error_list_per_var = []
        for pred_var_id in ind_f_vars:
            # window grows from min_window_size 
            # to whole dataset size (minus forecast horizon)
            error_list = []

            # first window
            self.lagless_X = self.lagless_data[:min_last_window_idx]
            self.lagless_y = self.lagless_data[h:min_last_window_idx+h, pred_var_id]
            # check if same amount of datapoints in X and y that are used by model training
            assert self.lagless_X.shape[0] == self.lagless_y.shape[0]

            # make sure model trains only on the newly defined lagless data (in self.lagless_X, self.lagless_y)
            assert self.X is None and self.y is None
            # train a model
            self.train_final_model()
            
            # evaluate its performance
            # next datapoint in timeframe has to be predicted
            new_testpoint_idx = min_last_window_idx

            # this is the individual datapoint that is added this iteration ("one row")
            single_test_X = self.lagless_data[new_testpoint_idx, :]
            single_test_y = self.lagless_data[new_testpoint_idx, pred_var_id]

            # create a lagged datapoint to ensure correct format when handing over to predict_with_trained_model()
            used_num_lags = self.num_lags
            assert used_num_lags >= 0 and used_num_lags+1 < new_testpoint_idx
            lagged_test_X = self.lagless_data[new_testpoint_idx-(used_num_lags+1):new_testpoint_idx, :]  
            lagged_test_X = self.add_lags(lagged_test_X, used_num_lags)
            lagged_test_X = np.reshape(lagged_test_X, (1, lagged_test_X.size))

            predictions = self.predict_with_trained_model(lagged_test_X)
            error_list.extend(predictions - single_test_y)

            for new_testpoint_idx in tqdm(range(min_last_window_idx+1, max_last_window_idx)):
                # make sure model trains only on the newly defined lagless data 
                self.X = None
                self.y = None

                # append extra_X and extra_y to the dataset (used to enable iterative training)
                self.add_extraXy_to_dataset()
                # the "extension of the window" for this iteration
                self.extra_X = single_test_X    # add old test point as extra data of this iteration 
                self.extra_y = np.array([single_test_y])
                
                # Current dataset shape that is used for the model training
                if self.verbose > 0:
                    print(f"Shape of lagless dataset available in this window: {self.lagless_X.shape}")

                # train the model using only the modified self.lagless_X, self.lagless_y
                assert self.X is None and self.y is None
                # CURRENTLY HERE
                assert self.extra_X is None or (self.extra_X.ndim == 1 and self.extra_y.size == 1)
                self.train_final_model()

                # set a new test point
                used_num_lags = self.num_lags
                assert used_num_lags >= 0 and used_num_lags+1 <= new_testpoint_idx
                # load new datapoint that will be given into self.extra_X, self.extra_y
                single_test_X = self.lagless_data[new_testpoint_idx, :]
                single_test_y = self.lagless_data[new_testpoint_idx, pred_var_id]
                # create a test datapoint with the lag used by the model
                lagged_test_X = self.lagless_data[new_testpoint_idx-(used_num_lags+1):new_testpoint_idx, :]  
                lagged_test_X = self.add_lags(lagged_test_X, used_num_lags)
                lagged_test_X = np.reshape(lagged_test_X, (1, lagged_test_X.size))

                # evaluate the model
                predictions = self.predict_with_trained_model(lagged_test_X)
                error_list.extend(predictions - single_test_y)

            error_list_per_var.append(error_list)
        return error_list_per_var


class Bayesian_Optimizer(trainable_model):
    '''
    The parent class for classes that want to implement bayesian hyperparameter optimization.
    To effectively implement a child of this class, we need to create the methods:
    black_box_function_adapter(), transform_params(), train_model() in the child class.
    For examples, see the classes: XGBoost_HyperOpt, CatBoost_HyperOpt, LightGBM_HyperOpt.
    '''
    def __init__(self, train_test_split_perc: float, search_space: dict,
                 is_reg_task: bool, pred_perf_metric: str, max_or_min: str, name: str,
                 init_points: int, n_iter: int, device: str, optimize_lag: bool, 
                 incrementally_trainable: bool=False, train_incrementally: bool=False):
        
        super().__init__(device=device, 
                         incrementally_trainable=incrementally_trainable, train_incrementally=train_incrementally)
        # True -> its a regression task, False -> its a classification task
        self.is_reg_task = is_reg_task

        # to decide how much of the given data is to be used for each training and
        # how much for the following performance evaluation -> better than single train set
        self.train_percentage = train_test_split_perc
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
        '''if self.optimize_lag and not lagless_X.shape[0] > 1:
            raise ValueError("Make sure to properly set the lagless_X and lagless_y variable with the available clean (non-lagged) data.")
        if lagless_X.shape[0] != lagless_y.shape[0]:
            raise ValueError("X and y don't have the same amount of datapoints.")
        self.lagless_X = lagless_X
        self.lagless_y = lagless_y'''

        # Track the performance at different choices of hyperparameters
        self.model_history = {}

        # The name of the algorithm that is optimized
        self.name = name

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
        Usese the optimal_model to make a prediction on given data.
        '''
        assert X.shape[1] == self.X.shape[1]
        return model.predict(X)

    def store_bayes_optimizer(self, file_path: str):
        # TODO
        pass

    def add_to_model_history(self, trained_model, para_dict: dict, perf_score: float):
        '''
        Keeps track of tried params and the resulting performances.
        '''
        self.model_history[len(self.model_history)] = {"model": trained_model, "params": para_dict, "perf": perf_score}

    def check_para_already_tested(self, para_dict: dict) -> Tuple[bool, float]:
        '''
        Tests if para dict was already tested.
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
        
    def black_box_function_adapter(self):
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS
        
        A function that takes the exact parameter names as the to train ML algo.
        These are put into a dictionary and handed over to black_box_function().
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

    def train_new_model(self, params: dict, X: np.ndarray, y: np.ndarray):
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
        for _ in range(self.amt_train_per_params):
            # 1. draw a random test, train split from the given data
            train_X, test_X, train_y, test_y = train_test_split(self.X, self.y,
                                                                test_size = self.train_percentage,
                                                                random_state = self.random_state)

            # 2. train a model using the given params
            model_params = copy.deepcopy(cor_params)
            model_params.pop("lag_to_add")
            # print(f"Working on data of shape: {train_X.shape}")
            trained_model = self.train_new_model(params = model_params, X = train_X, y = train_y)

            # 3. make that classifier predict unseen test data
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
            return trained_model

class CatBoost_HyperOpt(Bayesian_Optimizer):
    '''
    A class that inherits from the Bayesian_Optimizer class and implements the bayesian hyperparameter
    optimization of a CatBoost decision tree.
    '''

    def __init__(self, train_test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "RMSE", max_or_min: str = "min",
                 init_points: int = 2, n_iter: int = 20, device: str="CPU",
                 optimize_lag: bool=False
                 ):
        self.is_reg_task = is_reg_task
        super().__init__(train_test_split_perc=train_test_split_perc,
                         search_space=search_space, is_reg_task=self.is_reg_task, 
                         pred_perf_metric=perf_metric, max_or_min=max_or_min, name="CatBoost",
                         init_points=init_points, n_iter=n_iter, device=device, 
                         optimize_lag=optimize_lag,
                         incrementally_trainable=False, train_incrementally=False)
        
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
    def __init__(self, train_test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "RMSE", max_or_min: str = "min",
                 init_points: int = 2, n_iter:int = 20, device: str="CPU",
                 optimize_lag: bool=False
                ):
        self.is_reg_task = is_reg_task
        super().__init__(train_test_split_perc=train_test_split_perc, search_space=search_space,
                         is_reg_task=self.is_reg_task, pred_perf_metric=perf_metric, max_or_min=max_or_min, name="LightGBM",
                         init_points=init_points, n_iter=n_iter, device=device,
                         optimize_lag=optimize_lag,
                         incrementally_trainable=False, train_incrementally=False)

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
        ret_params["num_leaves"] = int(ret_params["num_leaves"])
        ret_params["max_depth"] = int(ret_params["max_depth"])
        ret_params["min_data_in_leaf"] = int(ret_params["min_data_in_leaf"])
        ret_params["lambda_l1"] = int(ret_params["lambda_l1"])
        ret_params["lambda_l2"] = int(ret_params["lambda_l2"])
        return ret_params

    def instantiate_model(self, cor_params: dict):
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

    def black_box_function_adapter(self, lag_to_add, learning_rate, num_leaves, max_depth, min_data_in_leaf,
                                   lambda_l1, lambda_l2, min_gain_to_split, bagging_fraction, feature_fraction):
        '''
        Implement an adapter that translates the parameter names and hands them to the real black_box_function() that
        is inherited from the parent class (Bayesian_Optimizer).
        '''
        # put all given parameters into a dictionary
        wrong_params_dict = {'lag_to_add': lag_to_add,
                  'learning_rate': learning_rate,
                  'num_leaves': num_leaves,
                  'max_depth': max_depth,
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
    def __init__(self, train_test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "RMSE", max_or_min: str = "min",
                 init_points: int = 2, n_iter: int = 20, device: str="CPU",
                 optimize_lag: bool=False,
                ):
        self.is_reg_task = is_reg_task
        super().__init__(train_test_split_perc=train_test_split_perc, search_space=search_space,
                         is_reg_task=self.is_reg_task, pred_perf_metric=perf_metric, max_or_min=max_or_min, name="XGBoost",
                         init_points=init_points, n_iter=n_iter, device=device,
                         optimize_lag=optimize_lag,
                         incrementally_trainable=False, train_incrementally=False)

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

from sklearn.ensemble import BaggingRegressor 
from sklearn.tree import DecisionTreeRegressor   
class BaggedTree(trainable_model):
    # TODO: make child-class of hyper opt so can optimize lag
    def __init__(self, n_estimators: int = 500):
        self.n_estimators = n_estimators
        super().__init__(device="CPU", incrementally_trainable=False, train_incrementally=False)

    def reset_model_training(self):
        self.trained_model = None

    def train_final_model(self):
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y don't have the same amount of datapoints.")
        
        if self.train_incrementally:
            raise ValueError("NOT IMPLEMENTED")
            if not (self.extra_X is None or self.extra_y is None):
                if self.extra_X.shape[0] != self.extra_y.size:
                    raise ValueError("X and y don't have the same amount of datapoints.") 
        else:
            self.reset_model_training()
            self.add_extraXy_to_dataset()
            model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=self.n_estimators, random_state=0)
            model.fit(self.X, self.y)
        self.trained_model=model

    def predict(self, X: np.ndarray, model) -> np.ndarray:
        return model.predict(X)