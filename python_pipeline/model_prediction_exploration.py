import pandas as pd
import os
import utils

model_preds_file = "sdfm_fred_192.csv"
model_preds_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\model_prediction_errors\\", model_preds_file)

#df = pd.read_csv(model_preds_path)

#print(df["crisis"].drop_duplicates())
# "model_label" -> "standard data", "fixedset30", "fixedset60"
# "target_var" -> "INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "DPCERA3M086SBEA"
# "horizon" -> "h1", "h3", "h6", "h12
# "start" -> "75"
# "crisis" -> "No Crisis", "Great Recession", "COVID-19"
# "values" -> errors of predictions of the model listed in "model_name"

error_list = [102 for i in range(192)]
model_errors_file_name = "some102.csv"
model_name = "rf____new2"
model_label = "standard data"
target_var = "CPIAUCSL"
horizon = "h1"
#utils.store_model_errors(error_list, model_errors_file_name, model_name, model_label, target_var, horizon,
#                         store_indiv=False, add_to_collection=True, verbose=True)

#dataset_df = utils.create_model_preds_dataset("optimized_forests/python_pipeline/model_prediction_errors/sdfm_fred_192.csv", truth=103)
#print(dataset_df)

store_path = "optimized_forests/python_pipeline/model_prediction_errors/sdfm_fred_192_colwise.csv"
#dataset_df.to_csv(store_path, index=False)


dataset_df = pd.read_csv(store_path)
dataset_df = dataset_df.drop(columns=["dates"])
#dataset_df_noTruth = dataset_df.drop(columns=["truth"])

best_pred_val_list = []
best_pred_error_list = []
best_pred_model_list = []
best_pred_index_list = []
for index, row in dataset_df.iterrows():
    this_truth = row["truth"]
    best_pred_val = 99999
    best_pred_error = 99999
    best_pred_name = ""
    for column_name, value in row.items():#
        if column_name == "truth":
            continue
        this_model_error = abs(value - this_truth)
        if this_model_error < best_pred_error:
            best_pred_val = value
            best_pred_error = this_model_error
            best_pred_name = column_name
            best_pred_index = dataset_df.columns.get_loc(column_name)
            
    best_pred_val_list.append(best_pred_val)
    best_pred_error_list.append(best_pred_error)
    best_pred_model_list.append(best_pred_name)
    best_pred_index_list.append(best_pred_index)

print(best_pred_model_list)
print(best_pred_index_list)
print(best_pred_val_list)
print(dataset_df.columns)

import numpy as np
print("Always best - MSE: ", np.square(np.array(best_pred_error_list)).mean())
print("###########################")

truth_col = dataset_df["truth"]
for column_name, column_values in dataset_df.items():
    print(f"{column_name} - MSE: {np.square(np.array(column_values-truth_col)).mean()}")


import matplotlib.pyplot as plt
from collections import Counter
# plotting how often model is the best
counter = Counter(best_pred_model_list)
labels, counts = zip(*counter.items())
plt.bar(labels, counts, edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.show()

# Create dataset for classification
# full dataset from before
# remove truth column
# add column index of best model to use
# add regular input data for model predictions and PCA
import copy
dataset_for_clf = copy.deepcopy(dataset_df)
dataset_for_clf["best_model_index"] = best_pred_index_list

# get regular data that predictions are done over
data_file_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out.csv")  # Inflation idx: 103
dataset = np.loadtxt(data_file_path, delimiter=";", encoding='utf-8-sig')
dataset_last192 = dataset[-192:, :]
dataset_last192_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out_last192_X.csv")
np.savetxt(dataset_last192_path, dataset_last192, delimiter=',')

best_predIndex_last192_path = os.path.join(os.getcwd(), "optimized_forests\\python_pipeline\\data_out_last192_optimalModelIndex_y.csv")
np.savetxt(best_predIndex_last192_path, np.array(best_pred_index_list, dtype=int)-1, delimiter=',', fmt='%d')  # since "truth" class would be at index 0 -> shift everything so "0" class is actually meaningfull



# try classification models: (could use softmax for weighted averaging)
# rf
# svm
# simple nn

plot = True
if plot:
    plt.plot(dataset_df)
    plt.plot(dataset_df["truth"], label="TRUTH")
    plt.plot(np.array(best_pred_val_list), label="ALWAYS BEST")
    plt.legend()
    plt.show()


'''
from sklearn.linear_model import LinearRegression
print(dataset_df.columns)
y = dataset_df['truth']
X = dataset_df.drop(columns=["truth"])

print(X.isna().any().any())

model = LinearRegression()
model.fit(X, y)

from sklearn.metrics import mean_squared_error
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print("MSE: ", mse)
'''