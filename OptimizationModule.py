# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:44:02 2023

@author: 70K9734
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


delta1= delta2= 0.05
alpha2= -(10)**(-3)
gamma_var=1
gamma_max= 10


err= np.random.randn(10, 100)




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




def optimization_num_forecast(err):
    
    n_validation, n_model = err.shape
    
    x = cp.Variable(n_model)
    A= err
    
    l1_list= []
    
    for i in range(A.shape[0]):
        objective_value= cp.norm( A[i, :]@x, 1)
        l1_list.append(objective_value)


    l2_list= []
    for i in range(A.shape[0]):
        variance= cp.sum_squares(A[i, :]*x - mean(A[0, :]*x))
        l2_list.append(variance)
    
    l_infinity_list= []
    for i in range(A.shape[0]):
        variance= cp.norm( A[0, :]*x, "inf")
        l_infinity_list.append(variance)
        
    l1_residual= sum(np.array(l1_list))
    variance_sum= sum(np.array(l2_list))
    max_error= sum(np.array(l_infinity_list))

    # testing_error_co= objective_value1 + objective_value2 + objective_value3 + objective_value4 + objective_value5 + objective_value6 + gamma_var*variance_sum+ gamma_max*max_error_sum

    testing_error_co= l1_residual+ gamma_var*variance_sum+ gamma_max*max_error

   
    objective = cp.Minimize(testing_error_co)

    constraints = [alpha2 <= x, 1- delta1<=sum(x), 1+ delta2>= sum(x)] 

    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(solver=cp.ECOS, verbose= True)

    x_optimal= x.value

    plt.plot(x_optimal)
    plt.show()

    return x_optimal



x_optimal= optimization_num_forecast(err)






