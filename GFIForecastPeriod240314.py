# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:16:03 2022

@author: 70K9734
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import cvxpy as cp
import os
import warnings
import shap
import time


from FeatureSelection import fs_scaled, fs_sanding
from ConvexOptimization import CustomizedOptimizationTrainingError, CustomizedOptimization
# from OptimizationModule import *
warnings.filterwarnings("ignore")

timestr = time.strftime("%Y%m%d")

nbpt = 6
# n_horizon= 13

target_variable= 'Order'
# commented out to carryout the analysis on NMFon 21st
# filename= 'Data/IndSumFinal230821.csv'
filename= 'Data/industrial_bertopicTM240314.csv'




n_test = 6
n_validation = 6
#n_labels= 35
n_check = 6
n_shift = 0
n_float= 0

filter_percentile = 100

alpha_range= np.array([-5*10 ** (-3)])
# alpha_range= np.array([-5*10 ** (-3), -1*10**(-2)])
gamma_range= [10]
gamma_max_range= [10]


gamma_var= 10
gamma_max= 10 
alpha2= -5*10 ** (-3)
delta1= 0.05 
delta2= 0.05

# alpha_train= 0.05

alpha_train_range= [0.05, 0.1]

delta= np.array([0.005, 0.005])
n_val_fixed= 6


model_dict_feat = {'gradient_boost': GradientBoostingRegressor()}

cv_list = np.arange(2, 5)
step_list= np.arange(0, 0.05, 0.01)
loss_list= ['squared_error']


df= pd.read_csv(filename)

y= np.array(df['Order'])

# n_horizon = 56
n_num= df.shape[0] - n_validation

def drop_col(df):
    df= df.drop(['Date', 'Order', 'Actual'], axis= 1)
    df = df.fillna(0)
    for col in df.columns:
        if np.var(np.array(df[col]))<= 10:
            df.drop(col, axis=1)

    return df

   

X= drop_col(df)


def drop_peak_mean(X):
    for i in range(X.shape[1]):
        if max(X.iloc[:, i])/np.mean(X.iloc[:, i])>= 50:
            pass
        


rng = pd.date_range('30/04/2018', periods= df.shape[0], freq='M')

# set_n_estimators= [10, 20, 25, 50, 70, 100, 200, 300, 500]
set_n_estimators = [500]
set_min_samples_leaf= [5]
# set_min_samples_leaf = [2, 3, 4, 5, 10]
# set_max_depth = [2, 3, 4, 5, 10]
set_max_depth = [5]
# set_min_samples_split = [2, 3, 4, 5, 20]
set_min_samples_split = [2]
set_loss = ['huber', 'absolute_error']
# set_loss= ['ls', 'huber']
set_learning_rate = [0.01]
set_criterion = ['squared_error']

# set_n_estimators_rf = [100, 200, 500, 700, 1000]
set_n_estimators_rf = [100]
# set_min_samples_leaf_rf = [2, 3, 4, 5, 7, 10]
set_min_samples_leaf_rf = [2, 3]
# set_max_features_rf = ['sqrt', 'auto']
set_max_features_rf = ['sqrt']
# set_min_samples_split_rf = [2, 3, 4, 5, 20]
set_min_samples_split_rf = [4]



def mean(x):
    return cp.sum(x) / x.size


def variance(x, mode='unbiased'):
    if mode == 'unbiased':
        scale = x.size - 1
    elif mode == 'mle':
        scale = x.size
    else:
        raise ValueError('unknown mode: ' + str(mode))
    return cp.sum_squares(x - mean(x)) / scale

def feature_selection_local(X, y, n_crease= 6):
    n_labels = X.shape[0] - n_float
    rfc_rf = GradientBoostingRegressor(random_state=101)
    rfecv_rf = RFECV(estimator=rfc_rf, step=0.1, cv=4, scoring='neg_mean_absolute_error')

    X_reduced = X.iloc[0: n_labels - n_crease, :]
    y_reduced = y[0: n_labels - n_crease]

    rfecv_rf.fit(X_reduced, y_reduced)

    X.drop(X_reduced.columns[np.where(rfecv_rf.support_ == False)[0]], axis=1, inplace=True)
    print('Selected features:', X.columns)
    return X

# X_after_feat_sel= feature_selection_local(X, y, n_crease=6)


def forecast_module(set_n_estimators, set_min_samples_leaf, set_max_depth, set_min_samples_split, set_loss,
                    set_learning_rate, set_criterion,
                    set_n_estimators_rf, et_min_samples_leaf_rf, set_max_features_rf, set_min_samples_split_rf, X, y,
                    alpha_range, gamma_range, delta):
    # n_labels = X.shape[0] - 11
    X= X.copy(deep= True)
    n_labels = X.shape[0] - n_float
    n_train = n_labels - n_test - n_check
    n_train_final_forecast = n_labels



    X = np.array(X)
    #    y= np.array(labels)

    X_train = X[0: n_train, :]
    y_train = y[0: n_train]

    X_train_final_forecast = X[0: n_train_final_forecast, :]
    y_train_final_forecast = y[0: n_train_final_forecast]

    print('*' * 100)
    print('Computing forecast')
    y_validation_gb = np.empty(
        [X.shape[0], len(set_n_estimators), len(set_min_samples_leaf), len(set_max_depth), len(set_min_samples_split),
         len(set_loss), len(set_learning_rate), len(set_criterion)])
    for i in range(len(set_n_estimators)):
        for j in range(len(set_min_samples_leaf)):
            for k in range(len(set_max_depth)):
                for l in range(len(set_min_samples_split)):
                    for m in range(len(set_loss)):
                        for n in range(len(set_learning_rate)):
                            for p in range(len(set_criterion)):
                                y_validation_gb[:, i, j, k, l, m, n, p] = GradientBoostingRegressor(random_state=0,
                                                                                                    n_estimators=
                                                                                                    set_n_estimators[i],
                                                                                                    min_samples_leaf=
                                                                                                    set_min_samples_leaf[
                                                                                                        j], max_depth=
                                                                                                    set_max_depth[k],
                                                                                                    min_samples_split=
                                                                                                    set_min_samples_split[
                                                                                                        l],
                                                                                                    loss=set_loss[m],
                                                                                                    learning_rate=
                                                                                                    set_learning_rate[
                                                                                                        n], criterion=
                                                                                                    set_criterion[
                                                                                                        p]).fit(X_train,
                                                                                                                y_train).predict(
                                    X)

    y_validation_gb = y_validation_gb.reshape(X.shape[0], len(set_n_estimators) * len(set_min_samples_leaf) * len(
        set_max_depth) * len(set_min_samples_split) * len(set_loss) * len(set_learning_rate) * len(set_criterion))

    y_validation_rf = np.empty(
        [X.shape[0], len(set_n_estimators_rf), len(set_min_samples_leaf_rf), len(set_max_features_rf),
         len(set_min_samples_split_rf)])
    for i in range(len(set_n_estimators_rf)):
        for j in range(len(set_min_samples_leaf_rf)):
            for k in range(len(set_max_features_rf)):
                for l in range(len(set_min_samples_split_rf)):
                    y_validation_rf[:, i, j, k, l] = RandomForestRegressor(random_state=0,
                                                                           n_estimators=set_n_estimators_rf[i],
                                                                           min_samples_leaf=set_min_samples_leaf_rf[j],
                                                                           max_features=set_max_features_rf[k],
                                                                           min_samples_split=set_min_samples_split_rf[
                                                                               l]).fit(X_train, y_train).predict(X)

    y_validation_rf = y_validation_rf.reshape(X.shape[0], len(set_n_estimators_rf) * len(set_min_samples_leaf_rf) * len(
        set_max_features_rf) * len(set_min_samples_split_rf))

    y_validation = np.concatenate([y_validation_gb, y_validation_rf], axis=1)

    error_all = np.empty([y_validation.shape[1], n_test])
    y_extracted = y_validation[n_train: n_train + n_test, :]
    y_extracted = y_extracted.T
    y_augmented = y[n_train: n_train + n_test] * np.ones([y_validation.shape[1], 1], dtype=None)

    for i in range(y_validation.shape[1]):
        for j in range(n_test):
            error_all[i, j] = (y_augmented[i, j] - y_extracted[i, j])

    error_all = np.array(error_all)

    error_all = error_all.T

    error_all_absolute = abs(error_all)

    mape_final = np.mean(error_all_absolute, axis=0)

    df_error = pd.DataFrame(data=mape_final)

    is_lesser = df_error[0] <= np.percentile(mape_final, filter_percentile)
    df_error = df_error[is_lesser]
    error_hist = np.array(df_error)

    df_validation = pd.DataFrame(data=y_validation.T)
    df_validation = df_validation[is_lesser]
    y_validation_filtered = df_validation.to_numpy()
    y_validation_filtered = y_validation_filtered.T

    y_forecast_final_gb = np.empty(
        [X.shape[0], len(set_n_estimators), len(set_min_samples_leaf), len(set_max_depth), len(set_min_samples_split),
         len(set_loss), len(set_learning_rate), len(set_criterion)])
    for i in range(len(set_n_estimators)):
        for j in range(len(set_min_samples_leaf)):
            for k in range(len(set_max_depth)):
                for l in range(len(set_min_samples_split)):
                    for m in range(len(set_loss)):
                        for n in range(len(set_learning_rate)):
                            for p in range(len(set_criterion)):
                                y_forecast_final_gb[:, i, j, k, l, m, n, p] = GradientBoostingRegressor(random_state=0,
                                                                                                        n_estimators=
                                                                                                        set_n_estimators[
                                                                                                            i],
                                                                                                        min_samples_leaf=
                                                                                                        set_min_samples_leaf[
                                                                                                            j],
                                                                                                        max_depth=
                                                                                                        set_max_depth[
                                                                                                            k],
                                                                                                        min_samples_split=
                                                                                                        set_min_samples_split[
                                                                                                            l],
                                                                                                        loss=set_loss[
                                                                                                            m],
                                                                                                        learning_rate=
                                                                                                        set_learning_rate[
                                                                                                            n],
                                                                                                        criterion=
                                                                                                        set_criterion[
                                                                                                            p]).fit(
                                    X_train_final_forecast, y_train_final_forecast).predict(X)

    y_forecast_final_gb = y_forecast_final_gb.reshape(X.shape[0],
                                                      len(set_n_estimators) * len(set_min_samples_leaf) * len(
                                                          set_max_depth) * len(set_min_samples_split) * len(
                                                          set_loss) * len(set_learning_rate) * len(set_criterion))

    y_forecast_final_rf = np.empty(
        [X.shape[0], len(set_n_estimators_rf), len(set_min_samples_leaf_rf), len(set_max_features_rf),
         len(set_min_samples_split_rf)])
    for i in range(len(set_n_estimators_rf)):
        for j in range(len(set_min_samples_leaf_rf)):
            for k in range(len(set_max_features_rf)):
                for l in range(len(set_min_samples_split_rf)):
                    y_forecast_final_rf[:, i, j, k, l] = RandomForestRegressor(random_state=0,
                                                                               n_estimators=set_n_estimators_rf[i],
                                                                               min_samples_leaf=set_min_samples_leaf_rf[
                                                                                   j],
                                                                               max_features=set_max_features_rf[k],
                                                                               min_samples_split=
                                                                               set_min_samples_split_rf[l]).fit(
                        X_train_final_forecast, y_train_final_forecast).predict(X)

    y_forecast_final_rf = y_forecast_final_rf.reshape(X.shape[0],
                                                      len(set_n_estimators_rf) * len(set_min_samples_leaf_rf) * len(
                                                          set_max_features_rf) * len(set_min_samples_split_rf))

    y_forecast_final = np.concatenate([y_forecast_final_gb, y_forecast_final_rf], axis=1)

    df_forecast = pd.DataFrame(data=y_forecast_final.T)
    df_forecast = df_forecast[is_lesser]
    y_forecast_filtered = df_forecast.to_numpy()
    y_forecast_filtered = y_forecast_filtered.T

    error_filtered = np.empty([y_validation_filtered.shape[1], n_test + n_check])
    y_extracted_flltered = y_validation_filtered[n_train: n_train + n_test + n_check, :]
    y_extracted_flltered = y_extracted_flltered.T
    y_augmented_filtered = y[n_train: n_train + n_test + n_check] * np.ones([y_validation_filtered.shape[1], 1],
                                                                            dtype=None)

    for i in range(y_validation_filtered.shape[1]):
        for j in range(n_test + n_check):
            error_filtered[i, j] = (y_augmented_filtered[i, j] - y_extracted_flltered[i, j])

    error_filtered = np.array(error_filtered)

    error_filtered = error_filtered.T

    error_all_absolute_flltered = abs(error_filtered)

    mape_final_filtered = np.mean(error_all_absolute_flltered, axis=0)

    # Training error
    error_training = np.empty([y_forecast_filtered.shape[1], (n_train_final_forecast - n_test - n_check)])
    # for i in range(y_forecast_filtered.shape[1]):
    #    for j in range(n_train_final_forecast- n_test):
    #        error_training[i, j]= (y[]- y_forecast_filtered[i, j])

    y_extracted_train = y_forecast_filtered[0: n_train_final_forecast - n_test - n_check, :]
    y_extracted_train = y_extracted_train.T
    y_augmented_train = y[0: (n_train_final_forecast - n_test - n_check)] * np.ones([y_forecast_filtered.shape[1], 1],
                                                                                    dtype=None)
    print('*' * 100)
    print('Performing customized optimization')
    for i in range(y_forecast_filtered.shape[1]):
        for j in range(n_train_final_forecast - n_test - n_check):
            error_training[i, j] = (y_augmented_train[i, j] - y_extracted_train[i, j])

    error_training = np.array(error_training)
    error_training = error_training.T

    error_final = np.concatenate((error_training, error_filtered), axis=0)

    A = error_filtered

    A_train = error_training

    n_validation, n_model = A.shape
    
    
    weights= []
    for alpha in alpha_range:
        for gamma in gamma_range:
            for alpha_train in alpha_train_range:
                weights_= CustomizedOptimizationTrainingError(alpha, gamma, delta, alpha_train, n_val_fixed).customized_optimization(A_train, A)
            # weights= optimization_num_forecast
            weights.append(weights_)


    X_validation = y_validation_filtered

    n_order_points = X_validation.shape[0]

    validation_matrix = np.empty([n_order_points, n_model])

    validation_forecast_list= []
    final_forecast_list= []
    y_forecast_final_list= []
    for x_optimal in weights:
        for i in range(n_order_points):
            for j in range(n_model):
                validation_matrix[i, j] = X_validation[i, j] * x_optimal[j]

        validation_forecast = np.sum(validation_matrix, axis=1)
        validation_forecast_list.append(validation_forecast)

        X_forecast = y_forecast_filtered
   
        forecast_matrix = np.empty([n_order_points, n_model])

        for i in range(n_order_points):
            for j in range(n_model):
                forecast_matrix[i, j] = X_forecast[i, j] * x_optimal[j]

        final_forecast = np.sum(forecast_matrix, axis=1)
        final_forecast_list.append(final_forecast)

    return final_forecast_list, validation_forecast_list, y_forecast_final, weights


def simple_support(forecast):
    for i in range(forecast.shape[0]):
        if forecast[i]<= 0.25*np.mean(forecast):
        
            forecast[i]= np.mean(forecast)*(1+ 0.01*np.random.randn(1, 1))
    return forecast


def forecast_simple(X, y, alpha_range, gamma_range, delta):
    final_forecast_list, validation_forecast_list, y_forecast_final, weights = forecast_module(set_n_estimators,
                                                                                       set_min_samples_leaf,
                                                                                       set_max_depth,
                                                                                       set_min_samples_split, set_loss,
                                                                                       set_learning_rate, set_criterion,
                                                                                       set_n_estimators_rf,
                                                                                       set_min_samples_leaf_rf,
                                                                                       set_max_features_rf,
                                                                                       set_min_samples_split_rf, X, y, alpha_range, gamma_range, delta)
  


    return final_forecast_list, validation_forecast_list, y_forecast_final, weights



def performance_mat(X, y, alpha_range, gamma_range, delta):
    
    final_forecast_list, validation_forecast_list, y_forecast_final, weights = forecast_simple(X, y, alpha_range, gamma_range, delta)

    error_list= []
    # validation_forecast= simple_support(validation_forecast)
    for validation_forecast in validation_forecast_list:
  
        
        y = np.array(y)[-n_check:]
        y_bar = np.array(validation_forecast)[-(n_check + 6):][:n_check]
    
        error = np.empty([y.shape[0]])
        for i in range(y.shape[0]):
            error[i] = abs(y[i] - y_bar[i]) / (y[i] + y_bar[i])
    
        s_mape = 200 * np.mean(error)
    
        error_r = np.empty([y.shape[0]])
        for i in range(y.shape[0]):
            error_r[i] = 100 * abs(y[i] - y_bar[i]) / (y[i])
        err_rmse = np.mean(error_r)
        error_list.append(err_rmse)
        error_final= np.array(error_list)
        
        validation_forecast_flat= np.array(validation_forecast_list)
        final_forecast_flat= np.array(final_forecast_list)

    return error_final, validation_forecast_flat, final_forecast_flat, weights



def rationalize_columns(X, y):
    X= X.iloc[:y.shape[0]]
    return X
df= pd.read_csv(filename)
# X= df.drop(['Date', 'Order'], axis= 1)

X= drop_col(df)

y= np.array(df[target_variable].dropna())

def train_for_fs(df):
    n_train= df.shape[0] - (n_test + n_check)
    return n_train
#
n_train_fs= train_for_fs(df)

X_train= X[:n_train_fs]
y_train= y[:n_train_fs]


selected_features = []

# X_feat, col_list= fs_scaled(X_train, y_train, loss_list, cv_list, step_list, is_scaling= 1)

X_feat, col_list= fs_sanding(X_train, y_train, loss_list, cv_list, step_list, is_scaling= 1)





# print(col_list)
def create_set(col_list):
    col_set= []
    for l in col_list:
        col_set.append(tuple(l))
    col_set= set(col_set)
    return col_set

col_list= create_set(col_list)

def to_str(col_list):
    col_list_str= []
    for l in col_list:
        for feat_name in l:
            feat_name= str(feat_name)
        col_list_str.append(list(l))
    return col_list_str

col_list= to_str(col_list)

X_final_df= []
for col_name in col_list:
    X_ind_ = X[col_name]
    X_final_df.append(X_ind_)

rmse_list= []
validation_list= []
val_list= []
for_list= []
weight_list= []

for X_df_one in X_final_df:
    try:
        error_final, validation_forecast_flat, final_forecast_flat, weights= performance_mat(X_df_one, y, alpha_range, gamma_range, delta)
        # print('Percentage error:', round(error_final, 2))
        rmse_list.append(error_final)
        val_list.append(validation_forecast_flat)
        for_list.append(final_forecast_flat)
        weight_list.append(weights)
        # validation_list.append(validation_forecast_)

    except:
        continue

plt.plot(np.array(rmse_list).T, ':+')
plt.show()


# extract the configuration with lowest error

rmse_all = np.array(rmse_list).reshape(-1)



# find the features where the average rmse is smallest

def rmse_array(rmse_list):
    rmse_arr= np.array(rmse_list)
    rmse_mean= np.mean(rmse_arr, axis=1)
    min_index= np.argmin(rmse_mean)
    return min_index
    
    
def process_weight_list(weight_list):
    weight_list= np.array(weight_list)
    weight_list= weight_list.reshape(weight_list.shape[0]* weight_list.shape[1], -1)
    return weight_list


best_feature_arg= rmse_array(rmse_list)


X_best= X_final_df[best_feature_arg]

weight_list= process_weight_list(weight_list)



def ind_best_hyperparameter(rmse):
    n_opt= np.argmin(rmse)
    return n_opt

n_opt= ind_best_hyperparameter(rmse_all)
Opt_weight= weight_list[n_opt, :]

arg_weight= Opt_weight.argsort()[-3:]

# print('The respective weight is:', Opt_weight[arg_weight])



model_list= []
for n_estimators in set_n_estimators:
    for min_samples_leaf in set_min_samples_leaf:
        for max_depth in set_max_depth:
            for min_samples_split in set_min_samples_split:
                for loss in set_loss:
                    for learning_rate in set_learning_rate:
                        for criterion in set_criterion:
                            model_list.append(GradientBoostingRegressor(random_state=0,
                                                      n_estimators= n_estimators,
                                                      min_samples_leaf= min_samples_leaf,
                                                      max_depth= max_depth,
                                                      min_samples_split= min_samples_split,
                                                      loss= loss,
                                                      learning_rate= learning_rate,
                                                      criterion= criterion))

for n_estimators in set_n_estimators_rf:
    for min_samples_leaf in set_min_samples_leaf_rf:
        for max_features in set_max_features_rf:
            for min_samples_split in set_min_samples_split_rf:
                model_list.append(RandomForestRegressor(random_state=0,
                                                         n_estimators= n_estimators,
                                                         min_samples_leaf= min_samples_leaf,
                                                         min_samples_split= min_samples_split))





def weight_num_forecast(X_best, n_num, model_list, df):
    X_train= X_best.iloc[:n_num, :]
    y= np.array(df[target_variable])
    y_train= y[:n_num]
    y_pred= []
    for model in model_list:
        model.fit(X_train, y_train)
        y_pred_= model.predict(X_best)
        y_pred.append(y_pred_)
    
    return np.array(y_pred), y

y_pred, y_act= weight_num_forecast(X_best, n_num, model_list, df)


def error_cal(y_pred, y_act, n_num):
    y_pred_trans= y_pred.T
    err_size= df.shape[0] - n_num
    err= np.empty([err_size, y_pred.shape[0]])
    for i in range(err_size):
        for j in range(y_pred.shape[0]):
            err[i, j]= 100*(y_act[i] - y_pred_trans[i,j])/y_act[i]
            
    return err

err= error_cal(y_pred, y_act, n_num)



# def optimization_num_forecast_dummy(err): 
#     n_validation, n_model = err.shape
#     x = cp.Variable(n_model)
#     A= err
#     objective_value1= cp.norm( A[0, :]@x, 1)
#     objective_value2= cp.norm( A[1, :]@x, 1)
#     objective_value3= cp.norm( A[2, :]@x, 1)
#     objective_value4= cp.norm( A[3, :]@x, 1)
#     objective_value5= cp.norm( A[4, :]@x, 1)
#     objective_value6= cp.norm( A[5, :]@x, 1)
#     # objective_value4= cp.norm( A[3, :]@x, 1)
#     variance1= cp.sum_squares(A[0, :]*x - mean(A[0, :]*x))
#     variance2= cp.sum_squares(A[1, :]*x - mean(A[1, :]*x))
#     variance3= cp.sum_squares(A[2, :]*x - mean(A[2, :]*x))
#     variance4= cp.sum_squares(A[3, :]*x - mean(A[3, :]*x))
#     variance5= cp.sum_squares(A[4, :]*x - mean(A[4, :]*x))
#     variance6= cp.sum_squares(A[5, :]*x - mean(A[5, :]*x))
#     # variance_sum= variance1+ variance2+ variance3+ variance4
#     variance_sum= variance1+ variance2+ variance3 +  variance4 +  variance5 +  variance6
#     max_error1= cp.norm( A[0, :]*x, "inf")
#     max_error2= cp.norm( A[1, :]*x, "inf")
#     max_error3= cp.norm( A[2, :]*x, "inf")
#     max_error4= cp.norm( A[3, :]*x, "inf")
#     max_error5= cp.norm( A[4, :]*x, "inf")
#     max_error6= cp.norm( A[5, :]*x, "inf")
#     # max_error4= cp.norm( A[3, :]*x, "inf")
#     # max_error_sum= max_error1+ max_error2+ max_error3+ max_error4
#     max_error_sum= max_error1+ max_error2+ max_error3 + max_error4 + max_error5 + max_error6
#     testing_error_co= objective_value1 + objective_value2 + objective_value3 + objective_value4 + objective_value5 + objective_value6 + gamma_var*variance_sum+ gamma_max*max_error_sum
#     objective = cp.Minimize(testing_error_co)
#     constraints = [alpha2 <= x, 1- delta1<=sum(x), 1+ delta2>= sum(x)] 
#     prob = cp.Problem(objective, constraints)
#     # The optimal objective value is returned by `prob.solve()`.
#     result = prob.solve(solver=cp.ECOS)
#     x_optimal= x.value
#     return x_optimal

# x_optimal= optimization_num_forecast(err)

# y_pred= weight_num_forecast(X_best, n_num, model_list, df)



weight_for_final= Opt_weight[arg_weight]
explainer= []
shap_values= []

for model in model_list:
    model.fit(X_best.iloc[:y.shape[0], :], y)

for model in model_list:
    explainer_= shap.TreeExplainer(model)
    explainer.append(explainer_)                            


for expla in explainer:
    shap_values.append(expla.shap_values(X_best))
    
# for i in x_optimal.shape[0]:
#     wt_shap_values[i]= x_optimal*shap_values
                                                      
global_importance= np.average(shap_values, axis= 0, weights= Opt_weight)
global_score= np.average(global_importance, axis= 0)
# global_score= np.average(global_importance, axis= 0)
# print(global_score.shape)

simple_global_importance= np.average(shap_values, axis= 0)
simple_global_score= np.average(simple_global_importance, axis= 0)



def plot_bar(X, y):
    x_values= X.columns
    y_values= y

    plt.figure(figsize= (20, 6))
    plt.bar(x_values, y_values)
    plt.xlabel('Features')
    plt.xticks(rotation= 45, fontsize= 10)
    plt.ylabel('weighted SHAP value')


    plt.show()

def bar_both(vec1, vec2):
    fig, ax = plt.subplots()
    bar_width = 0.35
# Positioning the bars
    r1 = range(len(vec1))
    r2 = [x + bar_width for x in r1]
# Creating the bars
    ax.bar(r1, vec1, color='blue', width=bar_width, edgecolor='grey', label='Vector 1')
    ax.bar(r2, vec2, color='orange', width=bar_width, edgecolor='grey', label='Vector 2')
# Adding labels and title
    ax.set_xlabel('Category')
    ax.set_ylabel('Values')
    ax.set_title('Barplot of Two Vectors')
    ax.set_xticks([r + bar_width / 2 for r in range(len(vec1))])
    # ax.set_xticklabels(x)
# Adding legend
    ax.legend()
# Display the plot
    plt.show()



# plot_bar(X_best, global_score1)
plot_bar(X_best, global_score)

bar_both(global_score, simple_global_score)

plot_df= pd.read_csv(filename)

actual= np.array(plot_df['Actual'])
num_forecast= np.array(plot_df['Order'])


def visualize_forecast(actual, forecast, num):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(rng[-forecast.shape[0]:], forecast, 'r')
    ax.plot(rng[0: actual.shape[0]], actual, ':k')
    ax.plot(rng[0: num_forecast.shape[0]], num_forecast, 'g')
    # ax.plot(rng[-best_forecast.shape[0]:][:-6], y[-rng[-best_forecast.shape[0]:][:-6].shape[0]:])
    
    ax.legend(['Textual forecast', 'Actual', 'Numerical forecast'])
    plt.show()



def find_best_forecast(list_rmse,
                       list_forecast,
                       y):
    rmse= np.array(list_rmse)
    forecast= np.array(list_forecast)

    idx_min_rmse= np.argmin(rmse)
    best_forecast_= forecast.reshape(forecast.shape[0]*forecast.shape[1], forecast.shape[2])
    best_forecast= best_forecast_[idx_min_rmse, :]
    
    

# The second smallest value's index is in partitioned_indices[1]
    try:
        partitioned_indices = np.argpartition(rmse, 1)[:2]
        second_min_index = partitioned_indices[1]
        second_best_forecast_= forecast.reshape(forecast.shape[0]*forecast.shape[1], forecast.shape[2])
        second_best_forecast= best_forecast_[second_min_index, :]

    except:
        second_best_forecast= best_forecast_[0, :]
    
    # print('the best forecast is:', best_forecast)
    
    visualize_forecast(actual, best_forecast, num_forecast)
    # plt.plot(y)

    return best_forecast, second_best_forecast

best_forecast, second_best_forecast= find_best_forecast(rmse_list,
                    for_list, y)

best_importance= dict(zip(X_best.columns, global_score))

# for feat, val in best_importance.items():
#     print(feat)
with open('features_{}.txt'.format(timestr), 'w') as f:
    for feat, val in best_importance.items():
        f.write(str(feat + ' ---- '))
        f.write(str(val))
        f.write('\n')