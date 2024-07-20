

# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math

import warnings
warnings.filterwarnings("ignore")

# dataset= load_boston()




model_dict={'gradient_boost': GradientBoostingRegressor()}



# # def transform_to_df(X):
#     X_df=pd.DataFrame(data=X, columns= feature_names)
#     return X_df 




class feature_selection:
    def __init__(self,
                 model_dict,
                 cv_list,
                 step_list, 
                 loss_list,
                 is_lasso= 1):
        self.model_dict= model_dict
        self.cv_list= cv_list
        self.step_list= step_list
        self.loss_list= loss_list

    def fit_rfecv(self,
                  X,
                  y):
        selected_features = []
        X_sel= []
        for model in self.model_dict.values():
            for cv in self.cv_list:
                for step in self.step_list:
                    for loss in self.loss_list:
                        try:
                            rfecv= RFECV(estimator= model, loss= loss, cv= cv, step= step)
                            _ = rfecv.fit(X, y)
                            X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
                            if len(X.columns) < 1:
                                X_sel.append(X)
                                print('# of features selected:', len(X.columns))
                                selected_features.append(X.columns)
                                
            # selected_features.append((X.columns[np.where(rfecv.support_)]))
                        # print(X.columns)
                                
                        except:
                            continue
                            
                       
                                    
        # print('This is the list:', X_sel) 
            # print(selected_features)
        # return selected_features, X_sel
        return selected_features
    
    



def feature_selection_function(X, y, loss_list, cv_list, step_list):
    X_sel_functions= []
    col_name= []
    for model in model_dict.values():
        for loss in loss_list:
            for cv in cv_list:
                for step in step_list:
                    try:
                        rfecv = RFECV(estimator=model, cv=cv, step=step)
                        rfecv.fit(X, y)
                        X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
                        X.dropna(axis=1, inplace=True)
                # selected_features.append((X.columns[np.where(rfecv.support_)]))
                    # print(X.columns)
                        X_sel_functions.append(X)
                        col_name.append(X.columns)
                    except:
                        continue

#         if len(X_sel_functions)<= 1:
#                     # Split data into training and testing sets
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#                     # Standardize features
#                 scaler = StandardScaler()
#                 X_train_scaled = scaler.fit_transform(X_train)
#                 X_test_scaled = scaler.transform(X_test)

# # Initialize Lasso regression model
#                 lasso = Lasso(alpha=0.1)  # You can adjust the regularization strength (alpha) as needed
#                 lasso.fit(X_train_scaled, y_train)
#                 selected_features_indices = np.where(lasso.coef_ != 0)[0]
#                 X_sel_functions = X.iloc[:, selected_features_indices]
#                 col_name= X_sel_functions.columns
#                 print('Selected features are:', X_sel_functions.columns)
#                 print('fraction of selected features=', len(X_sel_functions.columns)/len(X.columns))
    return X_sel_functions, col_name


def fs_scaled(X, y, loss_list, cv_list, step_list, is_scaling= 1):
    if is_scaling == 1:
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        X_scaled= pd.DataFrame(data= X_scaled, columns=(X.columns))
        X_sel_functions, col_name= feature_selection_function(X_scaled, y, loss_list, cv_list, step_list)
    else:
        X_sel_functions, col_name= feature_selection_function(X, y, loss_list, cv_list, step_list)
    return X_sel_functions, col_name


def sand_paper(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X.iloc[i, j] = ((X.iloc[i, j] + 10**(-2))/(10**(-2)))

    return X

def fs_sanding(X, y, loss_list, cv_list, step_list, is_scaling= 1):
    if is_scaling== 1:
        X_sand= sand_paper(X)
        X_sel_functions, col_name = feature_selection_function(X_sand, y, loss_list, cv_list, step_list)
    else:
        X_sel_functions, col_name = feature_selection_function(X, y, loss_list, cv_list, step_list)
    return X_sel_functions, col_name


        

# cv_list = np.arange(1, 5)
# step_list= np.arange(0, 0.1, 0.05)
# # loss_list= ['squared_error', 'absolute_error', 'huber', 'quantile']
# loss_list= ['squared_error']
# target_variable= 'Order'
# # commented out to carryout the analysis on NMFon 21st
# # filename= 'Data/IndSumFinal230821.csv'
# filename= 'Data/FeatureEncoded240221.csv'

# df= pd.read_csv(filename)

# y= np.array(df['Order'])
# y= y[~np.isnan(y)]
# # n_horizon = 56
# # n_num= df.shape[0] - n_validation

# def drop_col(df):
#     df= df.drop(['Date', 'Order'], axis= 1)
#     df = df.dropna(axis=1)
#     for col in df.columns:
#         if np.var(np.array(df[col]))<= 10:
#             df.drop(col, axis=1)

#     return df

# X= drop_col(df)
# X_feat, col_list= fs_scaled(X.iloc[:y.shape[0]], y, loss_list, cv_list, step_list, is_scaling= 0)




