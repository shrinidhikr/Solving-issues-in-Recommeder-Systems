#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:44:32 2019

@author: sneha
"""

import pandas as pd
import pickle

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


def tester(df_test, u_i_bm) :
    #Testing prediction
    test_predictions = ratings_prediction(df_test, u_i_bm[0][2], u_i_bm[0][3], u_i_bm[0][4], u_i_bm[0][5], u_i_bm[0][6])
    
    #print("Predictions of test", test_predictions)
    
    rmse = rmse_calculator(df_test1, test_predictions)
    
    return rmse
  

if __name__ == '__main__':
    
    # Read the dataset
    df = pd.read_csv('/home/sneha/Documents/RSGWO/322_30movie_ratings.csv')

    pickle_u_i_bm = pickle.load(open("/home/sneha/Documents/RSGWO/RESULTS/ps_user_edited_all_dump1_5.pickle","rb"))

    df_new = df.iloc[:,1:]
    df_test1 = df_new[151:200]
    df_test2 = df_new[200:250]
    df_test3 = df_new[250:300]
    
    # Calling the model function            
    r1 = tester(df_test1, pickle_u_i_bm)
    r2 = tester(df_test2, pickle_u_i_bm)
    r3 = tester(df_test3, pickle_u_i_bm)
    
    r = (r1 + r2 + r3)/3
    print(r)