
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

class opt:
    def __init__(self):

        self.c = 1            # include constant if set to one
        self.m = 192          # out-of-sample periods
        self.max_AR = 6       # maximum AR lags in AR(p) process
        self.max_F = 6        # maximum number of lags of factors in diffusion model
        self.nf_static = 10   # maximum possible number of static factors
        self.h = [1,3,6,12]   # forecast horizons [1, 6, 12]
        self.ic = 'bic'     # information criterion, either 'aic', 'bic', 'hq'
        self.ic_VS = "CV"   # Optimal IC for Lasso/EN Forecast ["CV", "AIC", "BIC"] and forecast_models_selected.m
        self.direct = 1     # compute direct forecast for ar if 1, otherwise iterative.

        self.run_pretransformation = 0          # [0,1] If one, recreate the dictionary. This is done when new data is added
        self.stationarity_test = 'adf & pp'     #['adf & pp', 'pp', 'adf'] # perform stationarity test according to the most restrictive of the two tests, the pp or the adf test
        self.interpolating_method = 'spline'    # ['none', 'spline', 'factor']
        self.transformation_method = 'fred' # ['fred', 'fred_all', 'all']
        self.preselection = 0   # [0,1] If one do subset if zeor do not subset
        
        self.start_date = '01.01.1975'  # Define start date
        self.end_date = '01.12.2022'    # Define end date

        self.vs_alpha = [0.5, 1]    # For Lasso and Elastic-Net
        self.min_train_ratio = 0.7  # Training sample size (Ratio from total)
        self.test_size = 5          # Test size in cross-validation
        self.LambdaCVn = 200    	# Number of grids for Lambda in Lasso/EN - norm 200

        self.vn = ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "DPCERA3M086SBEA"]

        # opt.vn = ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "DPCERA3M086SBEA", ...
        #           "RETAILx", "HOUST", "M2SL", "CONSPI", "WPSFD49207", ...
        #           "CMRMTSPLx", "RPI", "FEDFUNDS", "IPFUELS", "IPMANSICS", ...
        #           "CLAIMSx", "CPIULFSL", "CUSR0000SAS", "PCEPI", "PPICMM"]; #"all" , "CPIAUCSL"
        self.vnt =[]    # Includes variable names transformations_name and "all"
        self.vntt = []  # Includes variable names transformations for all if "all" include


def add_lags(dataset: np.ndarray, lags: int) -> np.ndarray:
    '''
    For a time-series of datapoints, adds "lags"-many lags to each datapoint.
    Since for the first "lags"-many ts points there is no data for the lags,
    the dataset that is returned contains "lags"-many datapoints less.
    '''
    # datapoints for which we are able to add the requested number of lags
    lagged_dataset = dataset[lags:, :]
    for this_lag in range(lags):
        print(this_lag)
        # for lags = 3 we get this_lag = 0 -> 1 -> 2
        this_subset = dataset[this_lag:-lags+this_lag, :]
        lagged_dataset = np.concatenate([lagged_dataset, this_subset], axis=1)
        print(lagged_dataset.shape)
    return lagged_dataset


def expanding_window(data: np.ndarray, model, ind_f_vars: List[int], col_names: List[str],
                    num_factors: int, num_lags: int, opt: opt, min_window_size: int = 100) -> np.ndarray:
    # ind_f_vars - Indices of variables to forecast
    T = data.shape[0]   # T - number of points in dataset
    #TODO: m = opt.m   # m - insample window size, WHY???
    h = opt.h[0]   # h - forecast horizon
    min_last_window_idx = min_window_size+1 # +1 for "test" datapoint
    max_last_window_idx = T-h-1   # all datapoints available up to last one that can be tested

    error_list_per_var = []
    for pred_var_id in ind_f_vars:
        # window grows from min_window_size 
        # to whole dataset size (minus forecast horizon)
        error_list = []

        # add lags: new data will contain "num_lags"-many points less than data
        data_with_lags = add_lags(data, num_lags)
        
        # first window
        X = data_with_lags[:min_last_window_idx]
        # labels are h many timesteps in the future of "available" data X
        y = data[h:min_last_window_idx+h, pred_var_id]
        
        # all but last point are used fore training
        model.X = X[:-1, :]
        model.y = y[:-1]
        # last datapoint in timeframe has to be predicted
        test_X = X[-1, :]
        test_X = np.reshape(test_X, (1, test_X.size))
        test_y = y[-1]

        # train a model
        model.train_final_model()
        # evaluate its performance
        predictions = model.predict_with_trained_model(test_X)
        error_list.extend(predictions - test_y)

        for new_testpoint_idx in tqdm(range(min_last_window_idx+1, max_last_window_idx)):
            # append extra_X and extra_y to the dataset
            model.add_extraXy_to_dataset()
            # add old test point as extra data of this iteration 
            # (the "extension of the window" for this iteration)
            model.extra_X = test_X
            model.extra_y = np.array([test_y])
            
            # set a new test point
            test_X = data_with_lags[new_testpoint_idx]  
            test_X = np.reshape(test_X,(1, test_X.size))
            test_y = data_with_lags[new_testpoint_idx, pred_var_id]

            # train the model
            model.train_final_model()

            # evaluate the model
            predictions = model.predict_with_trained_model(test_X)
            error_list.extend(predictions - test_y)

        error_list_per_var.append(error_list)
    return error_list_per_var