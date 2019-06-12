#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 00:15:14 2019

@author: Shrinidhi KR and Sneha V
"""
# Installing Required Libraries
import pandas as pd
import numpy  as np
import pickle
from sklearn.model_selection import train_test_split
import GWO

# Function: Global Centering
def global_centering(Xdata_matrix, user_in_columns = True):
    ''' Calculates the global mean of the matrix 
        For each column, Sum of column / Count in column
    '''
    Xdata = Xdata_matrix.copy(deep = True)
    ''' Mean of the entire dataset is calculated 
        Missing values are discarded when calculating the mean
    '''
    if (user_in_columns == False):
        Xdata = Xdata.T
        
    global_mean = sum(Xdata.sum())/sum(Xdata.count()) 
    # Missing values are discarded when calculating the mean
    ''' Mean is subtracted from each value in the matrix
        Nan is filled with zero
    '''
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (pd.isnull(Xdata.iloc[i, j])):
                Xdata.iloc[i, j] = 0.0
            elif (pd.isnull(Xdata.iloc[i, j]) == False):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - global_mean      
    return Xdata, global_mean


# Function: Weigth Matrix
def weigth_matrix_calc(Xdata, w_list):
    ''' Create a users * users matrix
        Assign weights to each index excluding diagonal (self to self)
    '''
    k = 0
    weigth_matrix = pd.DataFrame(np.zeros((Xdata.shape[1], Xdata.shape[1])))
    for i in range(0, weigth_matrix.shape[0]):
        for j in range(0, weigth_matrix.shape[1]):           
            if (i == j):
                weigth_matrix.iloc[i, j] = 0.0
            else:
                weigth_matrix.iloc[i, j] = w_list[k]
                k = k + 1            
    return weigth_matrix


# Function: Ratings Prediction
def ratings_prediction(original, weigth_matrix, global_mean, p_user_list, bias_user_list, bias_item_list):
    bias = original.copy(deep = True)
    prediction = original.copy(deep = True)
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            bias.iloc[i, j] = bias_user_list[j] + bias_item_list[i]
            prediction.iloc[i, j] = 0
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            for k in range(0,  weigth_matrix.shape[1]):
                if ((not (pd.isnull(original.iloc[i, k]))) and (not (pd.isnull(original.iloc[i, j]))) and k != j):
                    prediction.iloc[i, j] = prediction.iloc[i, j] + weigth_matrix.iloc[k,j]*(original.iloc[i, k]+ (-bias_user_list[k] - bias_item_list[i]))
            prediction.iloc[i, j] = prediction.iloc[i, j]/p_user_list[i]**(1/2)
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            prediction.iloc[i, j] = prediction.iloc[i, j] + bias.iloc[i, j] + global_mean
    return prediction


# Function: RSME calculator
def rmse_calculator(original, prediction):   
    mse = prediction.copy(deep = True)   
    for i in range (0, original.shape[0]):
        for j in range (0, original.shape[1]):
            if (not (pd.isnull(original.iloc[i, j]))):
                mse.iloc[i][j] = (original.iloc[i][j] - prediction.iloc[i][j])**2 
            else:
                mse.iloc[i][j] = 0
    rmse  = sum(mse.sum())/sum(mse.count())
    rmse = (rmse)**(1/2)    
    return rmse


# Function: Separate Lists
def separate_lists(Xdata, variable_list):
    ''' Separating the variable list which contains random positions of the wolf as weights and bias
        for each user and each item
    '''
    w_list = [0]*(Xdata.shape[1]**2 - Xdata.shape[1])
    bias_user_list = [0]*Xdata.shape[1]
    bias_item_list = [0]*Xdata.shape[0]
    r = len(w_list)
    s = r + len(bias_user_list)
    t = s + len(bias_item_list)
    
    # Weights set for users
    w_list = list(variable_list.iloc[0:r])

    # Bias set for users
    bias_user_list = list(variable_list.iloc[r:s])

    # Bias set for items
    bias_item_list = list(variable_list.iloc[s:t]) 

    return w_list, bias_user_list, bias_item_list


# Function: Loss Function
def loss_function(original, variable_list, user_in_columns = True):
    Xdata, global_mean = global_centering(original, user_in_columns = user_in_columns)
    
    # Number of users who watched a movie
    p_user_list = original.count(axis = 1)
    
    # Obtaining lists that contain weights for each movie and bias for each item and each movie
    w_list, bias_user_list, bias_item_list = separate_lists(Xdata, variable_list)
    
    # Assign values from ww_list_user into a matrix 
    weigth_matrix = weigth_matrix_calc(Xdata, w_list)
    
    # Predict ratings 
    prediction = ratings_prediction(original, weigth_matrix, global_mean, p_user_list, bias_user_list, bias_item_list)
    
    # Calculation of rmse
    rmse = rmse_calculator(original, prediction)
    
    return prediction, rmse, weigth_matrix, global_mean,  p_user_list, bias_user_list, bias_item_list


# Function: User Based Model
def user_based_model(Xdata, user_in_columns = True, pack_size = 25, iterations = 75):
    n = Xdata.shape[1]**2 + Xdata.shape[0]
    
    # GWO Objective Function
    def solver_min (variables_values = [0]):
        prediction, rmse, weigth_matrix, global_mean,  p_user_list, bias_user_list, bias_item_list = loss_function(Xdata, variable_list = variables_values, user_in_columns = user_in_columns)
        return rmse
    
    gwo = GWO.grey_wolf_optimizer(target_function = solver_min, pack_size = pack_size, min_values = [-1.5]*n, max_values = [1.5]*n, iterations = iterations)
    ubm = loss_function(Xdata, variable_list = gwo[:-1], user_in_columns = user_in_columns)
    
    return ubm, gwo
   
    
if __name__ == '__main__':
    
    # Read the dataset
    df = pd.read_csv('200x30movie_ratings.csv')
    
    '''X_train, X_test = train_test_split(df,test_size=0.25, random_state=42)'''
    
    '''# Discard first column as it contains movie names
    xtrain = X_train.iloc[:,1:]
    xtest = X_test.iloc[:,1:]'''
    
    # Set the first column as index of row names
    #X = X.set_index(df.iloc[:,0]) 
    
    df_new = df.iloc[:,1:]
    df_train = df_new[0:150]
    df_test = df_new[151:]
    
    file = open("ps_user_edited_all_dump1_20.pickle","wb")
    
    # Calling the model function
    u_i_bm = user_based_model(df_train, pack_size = 20, iterations = 100)
    
    #Testing prediction
    test_predictions = ratings_prediction(df_test, u_i_bm[0][2], u_i_bm[0][3], u_i_bm[0][4], u_i_bm[0][5], u_i_bm[0][6])
    
    print("Predictions of test", test_predictions)
    
    pickle.dump(u_i_bm, file)
    
    test_rmse = rmse_calculator(df_test,test_predictions)
    print("Rmse of test data", test_rmse)
    
    pickle_u_i_bm = pickle.load(open("ps_user_edited_all_dump1_20.pickle","rb"))
    file.close()
    print("Predictions of pickle", pickle_u_i_bm[0][0])
    #print("Predictions", u_i_bm[0][0])
    
    
