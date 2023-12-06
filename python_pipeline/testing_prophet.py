from prophet import Prophet
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List


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


benchmark_rf_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\fred__75__rf__h_1__4.csv")
benchmark_rf_errors = np.loadtxt(benchmark_rf_path, encoding='utf-8-sig')

data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv")  # Inflation idx: 103
#data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_trans_outlier_and_na_removed.csv")    # Inflation idx: 104
dataset = np.loadtxt(data_file_path, delimiter=";", encoding='utf-8-sig')
col_names = [str(i) for i in range(dataset.shape[1])]
dataset = pd.DataFrame(dataset, columns=col_names)

start_date = datetime(1999, 1, 1)
num_data_points = len(dataset)
timestamps = [start_date + relativedelta(months=i) for i in range(num_data_points)]
dataset['ds'] = timestamps

predict_col = 103
#dataset = dataset.iloc[-len(benchmark_rf_errors):]
#dataset["y"] = benchmark_rf_errors
#print(dataset)
dataset = dataset.rename(columns={str(predict_col): 'y'})

# Loop through your expanding window data
true_values = []
pred_values = []
rf_pred = []
alternative = []

prophet_errors = []
alternative_error = []
prophet_errors_lower=[]
prophet_errors_upper=[]
first_it = True

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plt.ion()
for i in range(383, len(dataset)-1):
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)

    # univariate forecasts
    train_data = dataset.loc[:i+1, ["y", "ds"]]  # Expanding window

    # multivariate forecasts
    '''for k in range(len(dataset.columns)-1):
        if k != predict_col:
            model.add_regressor(str(k))
    train_data = dataset.iloc[:i+1]'''

    # train a prophet model on the available data
    model.fit(train_data)

    # predict a future value with the trained model
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    #model.plot(forecast)

    # Update true and predicted values list
    truth = dataset.loc[i+1]["y"]
    true_values.append(truth)
    pred_upper = forecast.iloc[-1]['yhat_upper']
    prediction = forecast.iloc[-1]['yhat']
    pred_lower = forecast.iloc[-1]['yhat_lower']

    pred_values.append(prediction)


    # simple model
    #alt_pred = train_data[-6:]["y"].mean()
    #alternative.append(alt_pred)
    #äalternative_error.append(truth - alt_pred)


    #rf_pred.append(truth + benchmark_rf_errors[i+1-383])

    prophet_errors_upper.append(truth - pred_upper)
    prophet_errors.append(truth - prediction)
    prophet_errors_lower.append(truth - pred_lower)


    if i%10 == 0:
        print("Benchmark RF:")
        mse_from_error_vec(benchmark_rf_errors[:len(prophet_errors)], plot=False, verbose=True)

        print("FB Prophet:")
        mse_from_error_vec(prophet_errors, plot=False, verbose=True)

        avg_RF_prophet_error = (benchmark_rf_errors[:len(prophet_errors)] + prophet_errors)/2
        print("Combined RF+Prophet:")
        mse_from_error_vec(avg_RF_prophet_error, plot=False, verbose=True)

        #print("Moving Average: 10")
        #mse_from_error_vec(alternative_error, verbose=True)

    ax1.clear()
    ax1.plot(true_values, label="Truth")
    ax1.plot(pred_values, label="My Prediction")
    ax1.plot(alternative, label="Moving Average: 10")
    #ax1.plot(rf_pred, label="Benchmark RF")
    ax1.set_title("Prediction")
    
    ax1.legend()
    

    ax2.clear()
    
    ax2.plot(benchmark_rf_errors, label="Benchmark RF - Error")
    ax2.plot(prophet_errors, label="My Prediction - Error")
    #ax2.plot(prophet_errors_lower, label="Prophet Lower")
    #ax2.plot(prophet_errors_upper, label="Prophet Upper")
    ax2.set_title("Errors")
    ax2.plot([0 for _ in range(len(prophet_errors))])
    ax2.legend()
    plt.grid(True)
    plt.pause(0.04)
plt.ioff()
plt.show()