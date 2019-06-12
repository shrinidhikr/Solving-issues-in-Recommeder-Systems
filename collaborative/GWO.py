# Required Libraries
import pandas as pd
import numpy  as np
import random
import datetime

# Function: Initialize Variables
def initial_position(target_function, pack_size = 5, min_values = [-5,-5], max_values = [5,5]):
    position = pd.DataFrame(np.zeros((pack_size, len(min_values))))
    position['Fitness'] = 0.0
    for i in range(0, pack_size):
        for j in range(0, len(min_values)):
             position.iloc[i,j] = random.uniform(min_values[j], max_values[j])
             print("P",i,j,position.iloc[i,j])
        position.iloc[i,-1] = target_function(position.iloc[i,0:position.shape[1]-1])
    return position

# Function: Initialize Alpha
def alpha_position(target_function, dimension = 2):
    alpha = pd.DataFrame(np.zeros((1, dimension)))
    alpha['Fitness'] = 0.0
    for j in range(0, dimension): #redundants
        alpha.iloc[0,j] = 0.0
    alpha.iloc[0,-1] = target_function(alpha.iloc[0,0:alpha.shape[1]-1])
    return alpha

# Function: Initialize Beta
def beta_position(target_function, dimension = 2):
    beta = pd.DataFrame(np.zeros((1, dimension)))
    beta['Fitness'] = 0.0
    for j in range(0, dimension):
        beta.iloc[0,j] = 0.0
    beta.iloc[0,-1] = target_function(beta.iloc[0,0:beta.shape[1]-1])
    return beta

# Function: Initialize Delta
def delta_position(target_function, dimension = 2):
    delta = pd.DataFrame(np.zeros((1, dimension)))
    delta['Fitness'] = 0.0
    for j in range(0, dimension):
        delta.iloc[0,j] = 0.0
    delta.iloc[0,-1] = target_function(delta.iloc[0,0:delta.shape[1]-1])
    return delta

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = position.copy(deep = True)
    for i in range(0, position.shape[0]):
        if (updated_position.iloc[i,-1] < alpha.iloc[0,-1]):
            for j in range(0, updated_position.shape[1]):
                alpha.iloc[0,j] = updated_position.iloc[i,j]
        if (updated_position.iloc[i,-1] > alpha.iloc[0,-1] and updated_position.iloc[i,-1] < beta.iloc[0,-1]):
            for j in range(0, updated_position.shape[1]):
                beta.iloc[0,j] = updated_position.iloc[i,j]
        if (updated_position.iloc[i,-1] > alpha.iloc[0,-1] and updated_position.iloc[i,-1] > beta.iloc[0,-1]  and updated_position.iloc[i,-1] < delta.iloc[0,-1]):
            for j in range(0, updated_position.shape[1]):
                delta.iloc[0,j] = updated_position.iloc[i,j] 
    return alpha, beta, delta

# Function: Updtade Position
def update_position(target_function, position, alpha, beta, delta, a_linear_component = 2, min_values = [-5,-5], max_values = [5,5]):
    updated_position = position.copy(deep = True)   
    for i in range(0, updated_position.shape[0]):
        for j in range (0, len(min_values)):   
            r1_alpha = random.uniform(0,1)
            r2_alpha = random.uniform(0,1)
            a_alpha = 2*a_linear_component*r1_alpha - a_linear_component
            c_alpha = 2*r2_alpha           
            distance_alpha = abs(c_alpha*alpha.iloc[0,j] - position.iloc[i,j]) 
            x1 = alpha.iloc[0,j] - a_alpha*distance_alpha      
            r1_beta = random.uniform(0,1)
            r2_beta = random.uniform(0,1)
            a_beta = 2*a_linear_component*r1_beta - a_linear_component
            c_beta = 2*r2_beta               
            distance_beta = abs(c_beta*beta.iloc[0,j] - position.iloc[i,j]) 
            x2 = beta.iloc[0,j] - a_beta*distance_beta                           
            r1_delta = random.uniform(0,1)
            r2_delta = random.uniform(0,1)
            a_delta = 2*a_linear_component*r1_delta - a_linear_component
            c_delta = 2*r2_delta               
            distance_delta = abs(c_delta*delta.iloc[0,j] - position.iloc[i,j]) 
            x3 = delta.iloc[0,j] - a_delta*distance_delta                                 
            updated_position.iloc[i,j] = (x1 + x2 + x3)/3
            if (updated_position.iloc[i,j] > max_values[j]):
                updated_position.iloc[i,j] = max_values[j]
            elif (updated_position.iloc[i,j] < min_values[j]):
                updated_position.iloc[i,j] = min_values[j]        
        updated_position.iloc[i,-1] = target_function(updated_position.iloc[i,0:updated_position.shape[1]-1])
    return updated_position

# GWO Function
def grey_wolf_optimizer(target_function, pack_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50):    
    count = 0
    position = initial_position(target_function = target_function, pack_size = pack_size, min_values = min_values, max_values = max_values)
    print(" pos ",position)
    alpha = alpha_position(target_function = target_function, dimension = len(min_values))
    print("a", alpha)
    beta = beta_position(target_function = target_function, dimension = len(min_values))
    print("b", beta)
    delta = delta_position(target_function = target_function, dimension = len(min_values))
    print("d", delta)
    
    while (count <= iterations):    
        print("Iteration = ", count, alpha.iloc[0,-1])      
        print(datetime.datetime.now()) 
        a_linear_component = 2 - count*(2/iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position = update_position(target_function, position, alpha, beta, delta, a_linear_component = a_linear_component, min_values = min_values, max_values = max_values)
        count = count + 1    
   
    print(alpha.iloc[0,-1])
    return alpha.iloc[alpha['Fitness'].idxmin(),:].copy(deep = True)
