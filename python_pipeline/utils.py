
import numpy as np

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