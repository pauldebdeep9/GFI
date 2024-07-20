

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# A = error_filtered

# n_val_fixed= 6


def mean(x):
    return cp.sum(x) / x.size



class CustomizedOptimization:
    def __init__(self,
                 alpha,
                 gamma,
                 delta,
                 n_val_fixed):
        self.alpha= alpha
        self.gamma= gamma
    
        self.delta= delta
        self.n_val_fixed= n_val_fixed
        # self.delta2 = delta2
        # self.A= A






    def customized_optimization(self, A):

        n_validation, n_model = A.shape
        self.delta1= self.delta[0]
        self.delta2= self.delta[1]
        x = cp.Variable(n_model)

        l1_list = []

        for i in range(self.n_val_fixed):
            objective_value = cp.norm(A[i, :] @ x, 1)
            l1_list.append(objective_value)

        l2_list = []
        for i in range(self.n_val_fixed):
            variance = cp.sum_squares(A[i, :] * x - mean(A[0, :] * x))
            l2_list.append(variance)

        l_infinity_list = []
        for i in range(self.n_val_fixed):
            variance = cp.norm(A[0, :] * x, "inf")
            l_infinity_list.append(variance)

        l1_residual = sum(np.array(l1_list))
        variance_sum = sum(np.array(l2_list))
        max_error = sum(np.array(l_infinity_list))

      
        testing_error_co = l1_residual + self.gamma * variance_sum 

        
        # testing_error_co = objective_value1 + objective_value2 + objective_value3 + objective_value4 + objective_value5 + objective_value6 + self.gamma * variance_sum

        objective = cp.Minimize(testing_error_co)

    # constraints = [alpha2 <= x, 1 - delta1 <= sum(x), 1 + delta2 >= sum(x)]

        constraints = [self.alpha <= x]
        constraints += [1 - self.delta1 <= sum(x)]
        constraints += [1 + self.delta2 >= sum(x)]


        prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve(solver=cp.ECOS)

        x_optimal = x.value

        return x_optimal
    






class CustomizedOptimizationTrainingError:
    def __init__(self,
                 alpha,
                 gamma,
                 delta,
                 alpha_train,
                 n_val_fixed):
        
        self.alpha= alpha
        self.gamma= gamma
    
        self.delta= delta
        self.alpha_train= alpha_train
        self.n_val_fixed= n_val_fixed
        
    def customized_optimization(self, A_train, A_val):

        n_validation, n_model = A_val.shape
        n_train, n_model = A_train.shape

        self.delta1= self.delta[0]
        self.delta2= self.delta[1]
        x = cp.Variable(n_model)

        
        l1_list = []

        for i in range(self.n_val_fixed):
            objective_value = cp.norm(A_val[i, :] @ x, 1)
            l1_list.append(objective_value)

        l2_list = []
        for i in range(self.n_val_fixed):
            variance = cp.sum_squares(A_val[i, :] * x - mean(A_val[0, :] * x))
            l2_list.append(variance)

        l_infinity_list = []
        for i in range(self.n_val_fixed):
            variance = cp.norm(A_val[0, :] * x, "inf")
            l_infinity_list.append(variance)

        l1_residual = sum(np.array(l1_list))
        variance_sum = sum(np.array(l2_list))
        max_error = sum(np.array(l_infinity_list))

      
        testing_error = l1_residual + self.gamma * variance_sum 



        l1_list_train = []
        for i in range(n_train):
            objective_value = cp.norm(A_train[i, :] @ x, 1)
            l1_list_train.append(objective_value)

        l2_list_train = []
        for i in range(n_train):
            variance = cp.sum_squares(A_train[i, :] * x - mean(A_train[0, :] * x))
            l2_list_train.append(variance)

        l_infinity_list_train = []
        for i in range(n_train):
            max = cp.norm(A_train[0, :] * x, "inf")
            l_infinity_list_train.append(max)

        l1_residual_train = sum(np.array(l1_list_train))
        variance_sum_train = sum(np.array(l2_list_train))
        max_error_train = sum(np.array(l_infinity_list_train))

        training_error= l1_residual_train + self.gamma*variance_sum_train
   
        total_error= testing_error + self.alpha_train* training_error

 
        objective = cp.Minimize(total_error)

    # constraints = [alpha2 <= x, 1 - delta1 <= sum(x), 1 + delta2 >= sum(x)]

        constraints = [self.alpha <= x]
        constraints += [1 - self.delta1 <= sum(x)]
        constraints += [1 + self.delta2 >= sum(x)]


        prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve(solver=cp.ECOS)

        x_optimal = x.value

        return x_optimal











