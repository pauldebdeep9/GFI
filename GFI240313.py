# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:28:48 2024

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


from FeatureSelection import fs_scaled
from ConvexOptimization import CustomizedOptimizationTrainingError, CustomizedOptimization
from OptimizationModule import *
warnings.filterwarnings("ignore")

timestr = time.strftime("%Y%m%d")



target_variable= 'Order'
# commented out to carryout the analysis on NMFon 21st
# filename= 'Data/IndSumFinal230821.csv'
filename= 'Data/IndSumNMF230921.csv'





def drop_col(df):
    df= df.drop(['Date', 'Order'], axis= 1)
    df = df.dropna(axis=1)
    for col in df.columns:
        if np.var(np.array(df[col]))<= 10:
            df.drop(col, axis=1)

    return df

   
df= pd.read_csv(filename)
y= np.array(df['Order'])


X= drop_col(df)



