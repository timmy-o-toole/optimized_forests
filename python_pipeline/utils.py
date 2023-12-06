
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
from scipy.fft import fft
from statsmodels.tsa.seasonal import STL
import pandas as pd
def enrich_dataset(dataset: np.ndarray, 
                   dataset_col_names: List[str],
                   nr_pca_factors: int=0,
                   nr_umap_dim: int=0,
                   min_mean_max_idx: int=-1,
                   fft_idx: int=-1,
                   ema_idx: int=-1,
                   stl_idx: int=-1):
    
    assert dataset.ndim == 2
    modified_data = copy.deepcopy(dataset)
    dataset_col_names_enriched = copy.deepcopy(dataset_col_names)
    #dataset_col_names_enriched = ["CPI"]

    if nr_pca_factors > 0:
        # do pca of data 
        dataset_pca, transformer_pca = compute_pca(dataset, nr_pca_factors)
        modified_data = np.concatenate([modified_data, dataset_pca], axis=1) #modified_data[:, 103:104]
        dataset_col_names_enriched.extend([f"PCA_{i}" for i in range(nr_pca_factors)])

    if nr_umap_dim > 0:
        # do umap of data 
        dataset_umap, transformer_umap = compute_umap(dataset, nr_umap_dim)
        modified_data = np.concatenate([modified_data, dataset_umap], axis=1)
        dataset_col_names_enriched.extend([f"UMAP_{i}" for i in range(nr_umap_dim)])

    if ema_idx > -1:
        # do exponential moving average for idx ema_idx
        ema_df = pd.DataFrame(dataset[:, ema_idx], columns=["idx_data"])
        ema_df["EMA"] = ema_df['idx_data'].ewm(alpha=0.2, adjust=False).mean()
        ema_for_idx = ema_df["EMA"].to_numpy()
        modified_data = np.concatenate([modified_data, ema_for_idx.reshape(-1, 1)], axis=1)
        dataset_col_names_enriched.extend(["EMA_alpha0.2"])

    results = []
    for last_idx in range(5, dataset.shape[0]):
        subset = dataset[:last_idx, :]
        this_step = []
        if min_mean_max_idx > -1:
            ts_min = np.min(subset[:, min_mean_max_idx])
            ts_mean = np.mean(subset)
            ts_max = np.max(subset)
            this_step.extend([ts_min, ts_mean, ts_max])
        if fft_idx > -1:
            fourier_factors = fft(subset[:, fft_idx])
            this_step.extend(list(np.abs(fourier_factors[:5])))
        if stl_idx > -1:
            stl = STL(subset[:, stl_idx], period=31)
            stl_results = stl.fit()
            this_step.extend([stl_results.trend[-1], stl_results.seasonal[-1], stl_results.resid[-1]])
        results.append(this_step)
    
    if min_mean_max_idx > -1:
        dataset_col_names_enriched.extend(["min", "mean", "max"])
    if fft_idx > -1:
        dataset_col_names_enriched.extend([f"FF_idx{fft_idx}_fac{i}" for i in range(5)])
    if stl_idx > -1:
        dataset_col_names_enriched.extend(["STL_trend", "STL_seasonal", "STL_resid"])

    results = np.array(results)
    #first_rows = np.full((1, results.shape[1]), np.nan)
    first_rows = np.zeros((1, results.shape[1]))
    for _ in range(5):
        results = np.concatenate([first_rows, results], axis=0)
    modified_data = np.concatenate([modified_data, np.array(results)], axis=1)

    assert modified_data.shape[1] == len(dataset_col_names_enriched)
    return modified_data, dataset_col_names_enriched, None


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

import shap
def show_shap_importance(model, dataset, overall_plot:bool=True, datapoint_idx:int=-1):
    explainer = shap.Explainer(model)
    if datapoint_idx > -1:
        sample_data = dataset[datapoint_idx, :]
        shap_values_idx = explainer.shap_values(sample_data)
        shap.plots.force(explainer.expected_value[0], shap_values_idx, sample_data, matplotlib=True)
        #shap.initjs()
        #shap.force_plot(explainer.expected_value[0], shap_values_idx, sample_data)
        plt.show()
    
    if overall_plot:
        shap_values = explainer.shap_values(dataset)
        shap.summary_plot(shap_values, dataset, plot_type="dot")
        plt.show()


def transition_to_tim_csvformat(prediction_errors, model_name, model_label, target_var, horizon, filename, start=75, date_crisis_list=None):
    '''
    Make sure the values in prediction_errors are the ERRORS of the models predictions!
    '''
    if date_crisis_list is None:
        date_crisis_list = get_date_crisis_list()
    
    column_names = ["filename", "model_name", "start", "horizon", "target_var", "model_label", "date", "crisis", "values"]
    df = pd.DataFrame(columns=column_names)

    for i in range(len(prediction_errors)):
        new_data_point = [filename, model_name, start, horizon, target_var, model_label, date_crisis_list[i, 0], date_crisis_list[i, 1], prediction_errors[i]]
        df.loc[i] = new_data_point

    return df


import os
def get_date_crisis_list(reference_path=None):

    if reference_path is None:
        model_preds_file = "sdfm_fred_192.csv"
        model_preds_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\model_prediction_errors\\", model_preds_file)
    else:
        model_preds_path = reference_path
    df = pd.read_csv(model_preds_path, delimiter=",")
    date_crisis_list = df[["date", "crisis"]].sort_values(ascending=True, by="date").drop_duplicates().to_numpy()
    return date_crisis_list


def store_array_as_csv(prediction_errors, file_name, csv_path=None, verbose=True):
    if csv_path is None:
        csv_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\model_prediction_errors\\", file_name)

    if os.path.exists(csv_path):
        raise ValueError(f"File in path {csv_path} already exists! Choose a different file name or path.")
    
    np.savetxt(csv_path, prediction_errors, delimiter=",")
    if verbose:
        print(f"Stored the given model prediction errors in {csv_path}.")


def update_model_pred_summary(additional_predictions_df, path_model_pred_summary=None, verbose=True):
    if path_model_pred_summary is None:
        file_model_pred_summary = "sdfm_fred_192_modified.csv"
        path_model_pred_summary = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\model_prediction_errors\\", file_model_pred_summary)

    if not os.path.exists(path_model_pred_summary):
        original_file =  os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\model_prediction_errors\\sdfm_fred_192.csv")
        predictions_so_far_df = pd.read_csv(original_file, delimiter=",")
    else:
        predictions_so_far_df = pd.read_csv(path_model_pred_summary, delimiter=",")

    assert set(predictions_so_far_df.columns) == set(additional_predictions_df.columns)
    predictions_extended_df = pd.concat([predictions_so_far_df, additional_predictions_df], sort=True, ignore_index=True)

    predictions_extended_df.to_csv(path_model_pred_summary, index=False)
    if verbose:
        print(f"Updated the file in path {path_model_pred_summary} with the provided predictions.")


def store_model_errors(error_list, model_errors_file_name, model_name, model_label, target_var, horizon, store_indiv=True, add_to_collection=True, verbose=True):
    if not isinstance(error_list, list):
        error_list = list(error_list)

    if store_indiv:
        store_array_as_csv(error_list, model_errors_file_name, verbose=verbose)
    if add_to_collection:
        tim_formatted_df = transition_to_tim_csvformat(error_list, model_name, model_label, target_var, horizon, model_errors_file_name)
        update_model_pred_summary(tim_formatted_df, path_model_pred_summary=None, verbose=verbose)


def create_model_preds_dataset(path_errors_per_model, truth, target_var="CPIAUCSL", horizon="h1", model_label="standard data", store_path=None):
    '''
    dataset_df = create_model_preds_dataset("optimized_forests/python_pipeline/model_prediction_errors/sdfm_fred_192.csv", [1 for _ in range(192)])
    '''
    
    # "model_label" -> "standard data", "fixedset30", "fixedset60"
    # "target_var" -> "INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "DPCERA3M086SBEA"
    # "horizon" -> "h1", "h3", "h6", "h12
    # "start" -> "75"
    # "crisis" -> "No Crisis", "Great Recession", "COVID-19"
    # "values" -> errors of predictions of the model listed in "model_name"
    
    dates = get_date_crisis_list()[:, 0]

    if isinstance(truth, int):
        # if truth is an int, it gives the index of the target variable in the data_out.csv file
        train_dataset_df = pd.read_csv(os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv"), delimiter=";")
        truth = list(train_dataset_df.to_numpy()[:, truth])[-len(dates):]

    # getting model_names for which the requested specifics are available
    model_preds_df = pd.read_csv(path_errors_per_model)
    model_names = model_preds_df
    cond1 = model_preds_df['target_var'] == target_var
    cond2 = model_preds_df['horizon'] == horizon
    cond3 = model_preds_df['model_label'] == model_label
    cond4 = model_preds_df['start'] == 75
    filtered_df = model_preds_df[cond1 & cond2 & cond3 & cond4].drop(columns=["target_var", "horizon", "model_label", "start", "filename", "crisis"])
    model_names = filtered_df["model_name"].drop_duplicates().to_numpy()

    dataset_df = pd.DataFrame({"dates": dates, "truth": truth})
    for model_name in model_names:
        model_predictions = []
        for idx, date in enumerate(dates):
            set_date = filtered_df["date"] == date
            set_model_name = filtered_df["model_name"] == model_name
            
            this_error = filtered_df[set_date & set_model_name]["values"]
            if this_error.size > 1:
                raise ValueError(f"There exist more than two results for the chosen settings. It appears that multiple results have been stored for: {[target_var, horizon, model_label, 75, model_name, date]}")
            this_prediction = truth[idx] - this_error.values[0]
            model_predictions.append(this_prediction)
        dataset_df[model_name] = model_predictions

    if not store_path is None:
        dataset_df.to_csv(store_path)

    return dataset_df