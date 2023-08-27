
import numpy as np
from typing import List, Tuple

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
        # for lags = 3 we get this_lag = 0 -> 1 -> 2
        this_subset = dataset[this_lag:-lags+this_lag, :]
        lagged_dataset = np.concatenate([lagged_dataset, this_subset], axis=1)
    return lagged_dataset


def expanding_window(data: np.ndarray, ind_f_vars: List[int], col_names: List[str],
                  num_factors: int, num_lags: int, opt: opt, min_window_size: int = 100) -> np.ndarray:
    # ind_f_vars - Indices of variables to forecast
    T = data.shape[0]   # T - number of points in dataset
    print(T)
    m = opt.m   # m - insample window size
    h = opt.h[0]   # h - forecast horizon
    print(h)
    min_window_idx = min_window_size+num_lags+1 # +1 for "test" datapoint
    max_window_idx = T-h   # all datapoints available up to last one that can be tested

    for pred_var_id in ind_f_vars:
        # window grows from min_window_size 
        # to whole dataset size (minus forecast horizon)
        for last_index_of_window in range(min_window_idx, max_window_idx):
            # growing window of data
            X = data[:last_index_of_window, :]
            # add lags: new X will contain "num_lags"-many points less than X
            X = add_lags(X, num_lags)
            # labels are h many timesteps in the future of "available" data X
            # "+num_lags" since first datapoints lost due to "not enough lags"
            Y = data[num_lags+h:last_index_of_window+h, pred_var_id]

            # all but last point are used fore training
            train_X = X[:-1, :]
            train_Y = Y[:-1]
            # last datapoint in timeframe has to be predicted
            test_X = X[-1, :]
            test_Y = Y[-1]

            # TODO: train a model on (train_X, train_Y)
            # TODO: test the trained model on (test_X, test_Y)
            

            return
            pass
    return