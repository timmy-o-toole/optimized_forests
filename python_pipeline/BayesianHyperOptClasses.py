# This file implements a class that can be used as parent class when doing a bayesian hyperparameter optimization 
# for a machine learning algorithm. It makes use of the "bayesian-optimization" library.

import copy
import time
from typing import Tuple
import numpy as np

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
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str="CPU", incrementally_trainable: bool=False, train_incrementally: bool=False):
        if X.shape[0] != y.size:
            raise ValueError("X and y don't have the same amount of datapoints.")
        self.X = X
        self.y = y
        self.extra_X = None
        self.extra_y = None

        self.device = device
        self.incrementally_trainable = incrementally_trainable
        self.train_incrementally = train_incrementally
        if not incrementally_trainable and train_incrementally:
            self.train_incrementally = False
            print("Incremental training is not available or not implemented for this model.")
        self.trained_model = None

    def add_extraXy_to_dataset(self):
        if not (self.extra_X is None or self.extra_y is None):
            self.X = np.concatenate([self.X, self.extra_X], axis=0)
            self.y = np.concatenate([self.y, self.extra_y], axis=None)
            assert self.y.shape[0] == self.X.shape[0]

            self.extra_X = None
            self.extra_y = None

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

    def predict_trained_model(self, X: np.ndarray) -> np.ndarray:
        model = self.trained_model
        if model is None:
            raise ValueError(f"There is no optimal_model stored that can be used to make a prediction.")
        return self.predict(X, model)

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
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, train_test_split_perc: float, search_space: dict,
                 is_reg_task: bool, pred_perf_metric: str, max_or_min: str, name: str,
                 init_points: int, n_iter: int, device: str,
                 incrementally_trainable: bool, train_incrementally: bool):
        # Part of the data that the ML model is trained with
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y don't have the same amount of datapoints.")
        super().__init__(X, y, device, incrementally_trainable, train_incrementally)
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
        
    def train_final_model(self):
        '''
        Uses the bayesian hyperpara optimization to find good parameters
        and then returns a model trained on the whole dataset using those parameters.
        '''
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y don't have the same amount of datapoints.")
        
        if self.train_incrementally:
            if not (self.extra_X is None or self.extra_y is None):
                if self.extra_X.shape[0] != self.extra_y.size:
                    raise ValueError("X and y don't have the same amount of datapoints.")
            self.optimize_hyperparameters(init_points=0, n_iter=self.incremental_train_n_iter)
        else:
            self.reset_model_training()
            self.add_extraXy_to_dataset()
            self.optimize_hyperparameters(init_points=self.init_points, n_iter=self.n_iter)
        trained_model = self.train_optimal_model()
        return trained_model

    def predict(self, X: np.ndarray, model=None) -> np.ndarray:
        '''
        Usese the optimal_model to make a prediction on given data.
        '''
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
        if (self.train_incrementally and self.model_history is not None 
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
                print(self.optimizer._space.__len__())        
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

        sum_perf_score = 0
        # We train the algorithm self.amt_train_per_params many times on the given params
        # with different train-test-splits to counteract overfitting.
        for _ in range(self.amt_train_per_params):

            # 1. draw a random test, train split from the given data
            train_X, test_X, train_y, test_y = train_test_split(self.X, self.y,
                                                                test_size = self.train_percentage,
                                                                random_state = self.random_state)

            # 2. train a model using the given params
            # print("Start Training of "+str(self.name))
            start = time.time()
            trained_model = self.train_new_model(params = cor_params, X = train_X, y = train_y)
            end = time.time()
            time_taken = end-start
            # print("Training of "+str(self.name)+" took: "+str(time_taken)+" sec.")

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
                train_features = self.X
                train_labels = self.y
            trained_model = self.train_new_model(self.optimal_params, train_features, train_labels)
            self.trained_model = copy.deepcopy(trained_model)
            return trained_model


class CatBoost_HyperOpt(Bayesian_Optimizer):
    '''
    A class that inherits from the Bayesian_Optimizer class and implements the bayesian hyperparameter
    optimization of a CatBoost decision tree.
    '''

    def __init__(self, X: str, y: str, train_test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "RMSE", max_or_min: str = "min", init_points: int = 2, n_iter: int = 20, device: str="CPU"):
        self.is_reg_task = is_reg_task
        super().__init__(X, y, train_test_split_perc, search_space, self.is_reg_task, perf_metric, max_or_min, "CatBoost", init_points, n_iter, device, False, False)
        
    def transform_params(self, input_params: dict) -> dict:
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
    
    def black_box_function_adapter(self, iterations, depth, learning_rate, random_strength, bagging_temperature, border_count, l2_leaf_reg):
        '''
        Implement an adapter that translates the parameter names and hands them to the real black_box_function() that
        is inherited from the parent class (Bayesian_Optimizer).
        '''
        # put all given parameters into a dictionary
        wrong_params_dict = {'iterations': iterations,
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
    def __init__(self, X: str, y: str, train_test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "RMSE", max_or_min: str = "min", init_points: int = 2, n_iter:int = 20, device: str="CPU"):
        self.is_reg_task = is_reg_task
        super().__init__(X, y, train_test_split_perc, search_space, self.is_reg_task, perf_metric, max_or_min, "LightGBM", init_points, n_iter, device, False, False)

    def transform_params(self, input_params: dict) -> dict:
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

    def black_box_function_adapter(self, learning_rate, num_leaves, max_depth, min_data_in_leaf,
                                   lambda_l1, lambda_l2, min_gain_to_split, bagging_fraction, feature_fraction):
        '''
        Implement an adapter that translates the parameter names and hands them to the real black_box_function() that
        is inherited from the parent class (Bayesian_Optimizer).
        '''
        # put all given parameters into a dictionary
        wrong_params_dict = {'learning_rate': learning_rate,
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
    def __init__(self, X: str, y: str, train_test_split_perc: float, search_space: dict, 
                 is_reg_task: bool = "True", perf_metric: str = "RMSE", max_or_min: str = "min", init_points: int = 2, n_iter: int = 20, device: str="CPU"):
        self.is_reg_task = is_reg_task
        super().__init__(X, y, train_test_split_perc, search_space, self.is_reg_task, perf_metric, max_or_min, "XGBoost", init_points, n_iter, device, False, False)

    def transform_params(self, input_params: dict) -> dict:
        '''
        Makes sure that the parameters passed to the actual ML algorithm are valid inputs.
        This includes for example turning floats into integers, and possible translating that integer
        as a categorical variable.
        '''
        ret_params = copy.deepcopy(input_params)
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

    def black_box_function_adapter(self, lambda_, alpha, max_depth, eta, gamma):
        '''
        Implement an adapter that summarizes and transforms the parameters
        and hands them to the real black_box_function() that is inherited
        from the parent class (Bayesian_Optimizer).
        '''
        # put all given parameters into a dictionary
        wrong_params_dict = {'lambda_': lambda_,
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
    def __init__(self, X:np.ndarray, y:np.ndarray, n_estimators:int=500):
        self.n_estimators = n_estimators
        super().__init__(X, y)

    def reset_model_training(self):
        pass

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
    
