# This file implements a class that can be used as parent class when doing a bayesian hyperparameter optimization 
# for a machine learning algorithm. It makes use of the "bayesian-optimization" library.

import copy
import time

import catboost as cat
import lightgbm as lgb

#pip install bayesian-optimization
import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


class Bayesian_Optimizer:
    '''
    The parent class for classes that want to implement bayesian hyperparameter optimization.
    To effectively implement a child of this class, we need to create the methods:
    black_box_function_adapter(), transform_params(), train_model() in the child class.
    For examples, see the classes: XGBoost_HyperOpt, CatBoost_HyperOpt, LightGBM_HyperOpt.
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, train_test_split_perc: float, search_space: dict, is_reg_task: bool, pred_perf_metric: str, max_or_min: str, name: str):
        # Part of the data that the ML model is trained with
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y don't have the same amount of datapoints.")
        self.X = X
        self.y = y
        self.amount_datapoints = self.y.shape[0]
        
        # True -> its a regression task, False -> its a classification task
        self.is_reg_task = is_reg_task

        # to decide how much of the given data is to be used for each training and
        # how much for the following performance evaluation -> better than single train set
        self.train_percentage = train_test_split_perc
        # Over how many different train-test splits the performance of a set of params should be evaluated
        self.amt_train_per_params = 1
        # TODO: make this k-fold cross val instead of random draws
        # TODO: does this even make sense for time-series data? How is cross-val done in TS data?

        #for the "randomized" train-test split draws
        self.random_state = 0
        
        # Defining your search space (dictionary)
        self.search_space = search_space
        
        # The BayesianOptimization object and the optimal found parameters
        self.optimizer = None
        self.optimal_params = None

        # The name of the algorithm that is optimized
        self.name = name

        # The metric used to evaluate which model performed best
        self.pred_perf_metric = pred_perf_metric

        # minimize or maximize performance metric
        if not max_or_min in ["max", "min"]:
            raise ValueError(f"max_or_min has to be set to either 'max' or 'min', not: '{max_or_min}'.")
        self.max_or_min = max_or_min


    def optimize_hyperparameters(self, init_points: int, n_iter: int):
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
        self.optimizer = BayesianOptimization(f = self.black_box_function_adapter,
                                 pbounds = self.search_space,
                                 random_state = 1,
                                 verbose = 0)
        print("Now right before setting the optimal_params field")
        # TODO: change to minimize dependent on to optimize metric
        self.optimizer.maximize(init_points = init_points, n_iter = n_iter)
        print(self.optimizer.max["params"])
        self.optimal_params = self.transform_params(self.optimizer.max["params"])
        
        
    def black_box_function_adapter(self):
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS
        
        A function that takes the exact parameter names as the to train ML algo.
        These are put into a dictionary and handed over to black_box_function().
        '''
        pass
    
    
    def transform_params(self, input_params):
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS
        
        Called by black_box_function
        Function that makes sure that the shape of params fed to the "to hyperparameter optimize ML Algorithm"
        is valid. Some arguments for example can only be given to the ML Algo as integers;
        the bayesian-optimization library only works on floats. (float -> int etc.)
        '''
        pass
    
    
    def train_model(self, params, X, y):
        '''
        HAS TO BE IMPLEMENTED BY CHILD CLASS
    
        Takes parameters, passes them to the individual ML model and returns the trained ML model.
        '''
        pass
    
    
    def black_box_function(self, params):
        '''
        Train a model and evaluate its performance (called by optimize_hyperparameters()).
        
        Transform params to be accaptable hyperparameters for the Classifier.
        Since for pbounds it is not possible to specify that some parameters are integers.
        '''
        cor_params = self.transform_params(params)

        sum_perf_score = 0
        # We train the algorithm self.amt_train_per_params many times on the given params
        # with different train-test-splits to counteract overfitting.
        for i in range(self.amt_train_per_params):
            # 1. draw a random test, train split from the given data
            train_X, test_X, train_y, test_y = train_test_split(self.X, self.y,
                                                                test_size = self.train_percentage,
                                                                random_state = self.random_state)

            # 2. train a model using the given params
            print("Start Training of "+str(self.name))
            start = time.time()
            trained_model = self.train_model(params = cor_params, X = train_X, y = train_y)
            end = time.time()
            time_taken = end-start
            print("Training of "+str(self.name)+" took: "+str(time_taken)+" sec.")

            # 3. make that classifier predict unseen test data
            model_pred = trained_model.predict(test_X)

            # 4. evaluate the performance of the trading bots prediction (trade fee: 0.3%)
            perf_score = self.prediction_performance_score(test_y, model_pred)

            # since library only can maximize scores, in case we want to minimize we invert the performance metric
            if self.max_or_min == "min":
                perf_score = -perf_score
            
            sum_perf_score += perf_score
            print("Performance for the "+ str(i) + " iteration: " + str(perf_score))
            self.random_state = self.random_state + 1
            
        # The performance of a set of hyperparameters for an ML algo is the average performance over multiple train-test splits
        ret_perf_score = sum_perf_score/self.amt_train_per_params    
        print("The average performance is "+str(ret_perf_score))
        return ret_perf_score
    
    
    def prediction_performance_score(self, true_y, pred_y):
        '''
        This function evalutes the performance of a model that is used to evaluate which hyperparameters
        are the best.
        
        Called by self.black_box_function().

        Input:
            pred_y (np.ndarray): a "vector" of predictions
            true_y (np.ndarray): a "vector" of "true value" that should have been predicted
            metric (str): metric to use for performance evaluation -> "rmse" or "mse"

        Output:
            Chosen performance metric rounded to 8 digits
        '''

        # Possible stats to look at for the performance of the evaluated model
        if self.is_reg_task:
            if self.pred_perf_metric == "rmse":
                perf_score = mean_squared_error(true_y, pred_y, squared=False)
            elif self.pred_perf_metric == "mse":
                perf_score = mean_squared_error(true_y, pred_y, squared=True)
            else: 
                raise ValueError("Entered "+self.pred_perf_metric+"as performance metric. See Bayesian_Optimizer.prediction_performance_score() for available metrics")
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
        # Check if self.optimal_params is set
        if self.optimal_params is None:
            print("There is no optimal set of parameters yet. Maybe you still have to run self.optimize_hyperparameters().")
            return None
        else:
            # Check if train-data is given or not
            if train_features is None:
                train_features = self.X
                train_labels = self.y
            trained_model = self.train_model(self.optimal_params, train_features, train_labels)
            return trained_model


class CatBoost_HyperOpt(Bayesian_Optimizer):
    '''
    A class that inherits from the Bayesian_Optimizer class and implements the bayesian hyperparameter
    optimization of a trading bot that uses a CatBoost decision tree.
    '''

    def __init__(self, X: str, y: str, train_test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "rmse", max_or_min: str = "min"):
        self.is_reg_task = is_reg_task
        super().__init__(X, y, train_test_split_perc, search_space, self.is_reg_task, perf_metric, max_or_min, "CatBoost")

    
    def black_box_function_adapter(self, iterations, depth, learning_rate, random_strength, bagging_temperature, border_count, l2_leaf_reg):
        '''
        Implement an adapter that translates the parameter names and hands them to the real black_box_function() that
        is inherited from the parent class (Bayesian_Optimizer).
        '''
        params = {'iterations': iterations,
                  'depth': depth,
                  'learning_rate': learning_rate,
                  'random_strength': random_strength,
                  'bagging_temperature': bagging_temperature,
                  'border_count': border_count,
                  'l2_leaf_reg': l2_leaf_reg
                 }
        perf_score = self.black_box_function(params)
        return perf_score
        
    def transform_params(self, input_params):
        '''
        Makes sure that the parameters passed to the actual ML algorithm are valid inputs.
        This includes for example turning floats into integers, and possible translating that integer
        as a categorical variable.
        '''
        ret_params = copy.deepcopy(input_params)
        ret_params["iterations"] = int(ret_params["iterations"])
        ret_params["depth"] = int(ret_params["depth"])
        ret_params["border_count"] = int(ret_params["border_count"])
        ret_params["l2_leaf_reg"] = int(ret_params["l2_leaf_reg"])
        return ret_params
    
    def train_model(self, params, X, y):
        '''
        This trains the deployed ML algorithm, given a set of parameters and train data (features, labels).
        '''
        cor_params = self.transform_params(params)
        if self.is_reg_task:
            model = cat.CatBoostRegressor(**cor_params)
        else:
            model = cat.CatBoostClassifier(**cor_params)
        model.fit(X, y)
        return model


class LightGBM_HyperOpt(Bayesian_Optimizer):
    '''
    A class that inherits from the Bayesian_Optimizer class and implements the bayesian hyperparameter
    optimization of a trading bot that uses a LightGBM decision tree.
    '''    
    def __init__(self, X: str, y: str, train_test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "rmse", max_or_min: str = "min"):
        self.is_reg_task = is_reg_task
        super().__init__(X, y, train_test_split_perc, search_space, self.is_reg_task, perf_metric, max_or_min, "XGBoost")


    def black_box_function_adapter(self, learning_rate, num_leaves, max_depth, min_data_in_leaf,
                                   lambda_l1, lambda_l2, min_gain_to_split, bagging_fraction, feature_fraction):
        '''
        Implement an adapter that translates the parameter names and hands them to the real black_box_function() that
        is inherited from the parent class (Bayesian_Optimizer).
        '''
        params = {'learning_rate': learning_rate,
                  'num_leaves': num_leaves,
                  'max_depth': max_depth,
                  'min_data_in_leaf': min_data_in_leaf,
                  'lambda_l1': lambda_l1,
                  'lambda_l2': lambda_l2,
                  'min_gain_to_split': min_gain_to_split,
                  'bagging_fraction': bagging_fraction,
                  'feature_fraction': feature_fraction
                 }
        perf_score = self.black_box_function(params)
        return perf_score
        
        
    def transform_params(self, input_params):
        '''
        Makes sure that the parameters passed to the actual ML algorithm are valid inputs.
        This includes for example turning floats into integers, and possible translating that integer
        as a categorical variable.
        '''
        ret_params = copy.deepcopy(input_params)
        ret_params["num_leaves"] = int(ret_params["num_leaves"])
        ret_params["max_depth"] = int(ret_params["max_depth"])
        ret_params["min_data_in_leaf"] = int(ret_params["min_data_in_leaf"])
        ret_params["lambda_l1"] = int(ret_params["lambda_l1"])
        ret_params["lambda_l2"] = int(ret_params["lambda_l2"])
        return ret_params
    
    
    def train_model(self, params, X, y):
        '''
        This trains the deployed ML algorithm, given a set of parameters and train data (features, labels).
        '''
        cor_params = self.transform_params(params)
        if self.is_reg_task:
            model = lgb.LGBMRegressor(**cor_params)
        else:
            model = lgb.LGBMClassifier(**cor_params)
        model.fit(X, y)
        return model


class XGBoost_HyperOpt(Bayesian_Optimizer):
    '''
    A class that inherits from the Bayesian_Optimizer class and implements the bayesian hyperparameter
    optimization of a trading bot that uses a XGBoost decision tree.
    '''
    def __init__(self, X: str, y: str, train_test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "rmse", max_or_min: str = "min"):
        self.is_reg_task = is_reg_task
        super().__init__(X, y, train_test_split_perc, search_space, self.is_reg_task, perf_metric, max_or_min, "XGBoost")


    def black_box_function_adapter(self, lambda_1, alpha, max_depth, eta, gamma):
        '''
        Implement an adapter that translates the parameter names and hands them to the real black_box_function() that
        is inherited from the parent class (Bayesian_Optimizer).
        '''
        params = {'lambda': lambda_1,
                  'alpha': alpha,
                  'max_depth': max_depth,
                  'eta': eta,
                  'gamma': gamma,
                 }
        perf_score = self.black_box_function(params)
        return perf_score
    
    
    def transform_params(self, input_params):
        '''
        Makes sure that the parameters passed to the actual ML algorithm are valid inputs.
        This includes for example turning floats into integers, and possible translating that integer
        as a categorical variable.
        '''
        ret_params = copy.deepcopy(input_params)
        # "objective"
        ret_params["eval_metric"] = "aucpr"
        ret_params["booster"] = "gbtree"
        ret_params["max_depth"] = int(ret_params["max_depth"])        
        #ret_params["lambda"] = ret_params["lambda_1"]      
        return ret_params
  

    def train_model(self, params, X, y):
        '''
        This trains the deployed ML algorithm, given a set of parameters and train data (features, labels).
        '''
        cor_params = self.transform_params(params)
        if self.is_reg_task:
            model = xgb.XGBRegressor(**cor_params)
        else:
            model = xgb.XGBClassifier(**cor_params)
        model.fit(X, y)
        return model