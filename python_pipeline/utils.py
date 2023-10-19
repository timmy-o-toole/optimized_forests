
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


def load_csv_as_np(file_path: str):
    return np.loadtxt(file_path, delimiter=";", encoding='utf-8-sig')


import matplotlib.pyplot as plt
def mse_from_true_and_pred(Y_true: List[float], Y_pred: List[float], plot: bool = False, verbose: bool = False):
    error_vec = np.subtract(Y_true,Y_pred)
    if plot:
        plt.hist(error_vec, bins=80, edgecolor='k')
        plt.show()
    
    mse = np.square(error_vec).mean() 
    if verbose:
        print("MSE: ", round(mse, 8))
    return mse

def mse_from_error_vec(error_vec: List[float], plot: bool = False, verbose: bool = False):
    if plot:
        plt.hist(error_vec, bins=80, edgecolor='k')
        plt.show()
    
    mse = np.square(error_vec).mean() 
    if verbose:
        print("MSE: ", round(mse, 8))
    return mse

import copy
def enrich_dataset(dataset: np.ndarray, 
                   dataset_col_names: List[str],
                   nr_pca_factors: int=0,
                   nr_umap_dim: int=0):
    assert dataset.ndim == 2
    print(dataset.shape)

    modified_data = copy.deepcopy(dataset)
    transformers = []
    if nr_pca_factors > 0:
        # do pca of data 
        dataset_pca = compute_pca(dataset, nr_pca_factors)
        modified_data, transformer_pca = np.concatenate([modified_data, dataset_pca], axis=1)
        dataset_col_names.extend([f"PCA_{i}" for i in range(nr_pca_factors)])
        transformers.append(transformer_pca)
    if nr_umap_dim > 0:
        dataset_umap = compute_umap(dataset, nr_umap_dim)
        modified_data, transformer_umap = np.concatenate([modified_data, dataset_umap], axis=1)
        dataset_col_names.extend([f"UMAP_{i}" for i in range(nr_umap_dim)])
        transformers.append(transformer_umap)

    assert modified_data.shape[1] == len(dataset_col_names)
    return modified_data, dataset_col_names, transformers


from sklearn.decomposition import PCA
def compute_pca(dataset: np.ndarray, nr_pca_factors: int):
    # Assuming data is a numpy array with shape (n_samples, n_features)
    pca = PCA(n_components=nr_pca_factors)
    pca.fit(dataset)
    dataset_reduced = pca.transform(dataset)
    return dataset_reduced, pca.transform

import umap
def compute_umap(dataset: np.ndarray, nr_umap_dim: int):
    # Assuming data is a numpy array with shape (n_samples, n_features)
    reducer = umap.UMAP(n_components=nr_umap_dim)
    reducer.fit(dataset)
    dataset_reduced = reducer.transform(dataset)
    return dataset_reduced, reducer.transform

def apply_transformers_to_data(dataset: np.ndarray, transformers: List[callable]):
    modified_data = copy.deepcopy(dataset)
    for transform in transformers:
        extra_data = transform(dataset)
        modified_data = np.concatenate([modified_data, extra_data], axis=1)
    return modified_data