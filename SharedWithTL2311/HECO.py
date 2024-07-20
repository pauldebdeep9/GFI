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
from FeatureSelection import *
from ConvexOptimization import *
warnings.filterwarnings("ignore")


nbpt = 6
# n_horizon= 13
n_horizon = 56



n_test = 6
n_validation = 6
#n_labels= 35
n_check = 6
n_shift = 6


### classification n
n_val_class = 3
n_for_class = 3
thr = 0.5
filter_percentile = 100

# alpha_range= np.array([-7*10 ** (-3), -5*10 ** (-3), -1*10 ** (-3), -1*10**(-2), -2*10**(-2)])
alpha_range= np.array([-5*10 ** (-3), -1*10**(-2)])
gamma_range= [10]
gamma_max_range= [10]
delta= np.array([0.05, 0.05])



model_dict_feat = {'gradient_boost': GradientBoostingRegressor()}


# df= pd.read_csv('Data/2301/IndJp230117.csv')
df= pd.read_csv('Data/IndSum230125.csv')
# df= pd.read_csv('IndSumFS230228.csv')
# df= pd.read_csv('Data/2301/IndSumNum230118.csv')


n_train_final_forecast= df.shape[0] - 11
# df['time']= (df.loc['time']).to_datetime
# df= df.sort_values('Date')
# df= df.drop('time', axis=1)

# print(df.head())
# print(df.columns)
y= df['Order']
y= np.array(y)[0: y.shape[0]-11]


def drop_col(df):
    df= df.drop('Date', axis= 1)
    df = df.dropna(axis=1)
    for col in df.columns:
        if np.var(np.array(df[col]))<= 10:
            df.drop(col, axis=1)

    return df


    

X= drop_col(df)
# print(df.columns)
# df.to_csv('sample.csv')

def drop_peak_mean(X):
    for i in range(X.shape[1]):
        if max(X.iloc[:, i])/np.mean(X.iloc[:, i])>= 50:
            pass
        


rng = pd.date_range('31/10/2017', periods= df.shape[0], freq='M')

# set_n_estimators= [10, 20, 25, 50, 70, 100, 200, 300, 500]
set_n_estimators = [100, 700]
# set_n_estimators= [10, 20, 50, 100, 200]
set_min_samples_leaf= [2, 10, 20]
# set_min_samples_leaf = [2, 3, 4, 5, 10]
# set_max_depth = [2, 3, 4, 5, 10]
set_max_depth = [3]
# set_min_samples_split = [2, 3, 4, 5, 20]
set_min_samples_split = [2]
set_loss = ['huber', 'absolute_error']
# set_loss= ['ls', 'huber']
set_learning_rate = [0.01]
set_criterion = ['squared_error']

# set_n_estimators_rf = [10, 20, 30, 40, 50, 100, 200, 500, 750]
set_n_estimators_rf = [100]
# set_min_samples_leaf_rf = [2, 3, 4, 5, 7, 10]
set_min_samples_leaf_rf = [2]
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
    n_labels = X.shape[0] - 11
    rfc_rf = GradientBoostingRegressor(random_state=101)
    rfecv_rf = RFECV(estimator=rfc_rf, step=0.1, cv=3, scoring='neg_mean_absolute_error')

    X_reduced = X.iloc[0: n_labels - n_crease, :]
    y_reduced = y[0: n_labels - n_crease]

    rfecv_rf.fit(X_reduced, y_reduced)

    X.drop(X_reduced.columns[np.where(rfecv_rf.support_ == False)[0]], axis=1, inplace=True)
    print('Selected features:', X.columns)
    return X

X_after_feat_sel= feature_selection_local(X, y, n_crease=6)


def forecast_module(set_n_estimators, set_min_samples_leaf, set_max_depth, set_min_samples_split, set_loss,
                    set_learning_rate, set_criterion,
                    set_n_estimators_rf, et_min_samples_leaf_rf, set_max_features_rf, set_min_samples_split_rf, X_after_feat_sel, y,
                    alpha_range, gamma_range, delta):
    # n_labels = X.shape[0] - 11
    X= X_after_feat_sel.copy(deep= True)
    n_labels = X.shape[0] - 11
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

    n_validation, n_model = A.shape
    
       
    weights= []
    for alpha in alpha_range:
        for gamma in gamma_range:
            weights_= CustomizedOptimization(alpha, gamma, delta).customized_optimization(A)
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


# final_forecast1, validation_forecast1= forecast_module(set_n_estimators, set_min_samples_leaf, set_max_depth, set_min_samples_split, set_loss, set_learning_rate, set_criterion, X1, y1, n_train, n_labels)

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
  
    
    # plt.plot(y) 
    # plt.plot(final_forecast, 'r')
    # plt.plot(validation_forecast, 'g')
    # plt.show()

    return final_forecast_list, validation_forecast_list, y_forecast_final, weights


# final_forecast, validation_forecast, y_forecast_final, x_optimal = forecast_simple(X= X_after_feat_sel, y)

# validation_forecast= simple_support(validation_forecast)

def performance_mat(X_after_feat_sel, y, alpha_range, gamma_range, delta):
    
    final_forecast_list, validation_forecast_list, y_forecast_final, weights = forecast_simple(X_after_feat_sel, y, alpha_range, gamma_range, delta)

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


# error_final= performance_mat(X_after_feat_sel, y, alpha_range, gamma_range, gamma_max_range, delta)

# s_mape, err_rmse = performance_mat(X, y)

def rationalize_columns(X, y):
    X= X.iloc[:y.shape[0]]
    return X
df= pd.read_csv('Data\IndSum230125.csv')
X= df.drop(['Date', 'Order'], axis= 1)
y= np.array(df['Order'].dropna())

def train_for_fs(df):
    n_train= df.shape[0] - (n_test + n_check)
    return n_train
#
n_train_fs= train_for_fs(df)

X_train= X[:n_train_fs]
y_train= y[:n_train_fs]


# X_feat= drop_col(df)
# X_feat_rat= X_feat[:n_train_feat]
# y_feat_rat= y[:n_train_feat]


# cv_list = np.arange(2, 7)
# step_list= [0.2, 0.3, 0.5, 0.7, 1]
selected_features = []
# input_list= []
cv_list = np.arange(2, 3)
step_list= [0.2]

# X_feat, col_list= feature_selection_function(X_train, y_train)
# col_list= feature_selection(model_dict= model_dict, cv_list = cv_list, step_list = step_list).fit_rfecv(X_train, y_train)


X_feat, col_list= fs_scaled(X_train, y_train, is_scaling= 1)

# X_sel_functions, col_list= fs_sanding(X_train, y_train, is_scaling= 1)



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

def process_weight_list(weight_list):
    weight_list= np.array(weight_list)
    weight_list= weight_list.reshape(weight_list.shape[0]* weight_list.shape[1], -1)
    return weight_list

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







weight_for_final= Opt_weight[arg_weight]
explainer= []
shap_values= []

for model in model_list:
    model.fit(X_after_feat_sel.iloc[:y.shape[0], :], y)

for model in model_list:
    explainer_= shap.TreeExplainer(model)
    explainer.append(explainer_)                            


for expla in explainer:
    shap_values.append(expla.shap_values(X_after_feat_sel))
                                                      
global_importance= np.average(shap_values, axis= 0, weights= Opt_weight)
#
# # print the global importance of each feature
# print("Global Importance:", global_importance.shape)
global_score= np.average(global_importance, axis= 0)
# print(global_score.shape)


def plot_bar(X, y):
    x_values= X.columns
    y_values= y

    plt.figure(figsize= (20, 6))
    plt.bar(x_values, y_values)
    plt.xlabel('Features')
    plt.xticks(rotation= 45, fontsize= 6)
    plt.ylabel('weighted SHAP value')


    plt.show()


plot_bar(X_after_feat_sel, global_score)




'''
with open('features221102.txt', 'w') as f:
    for feat in X.columns:
        f.write(feat)
        f.write('\n')
'''



