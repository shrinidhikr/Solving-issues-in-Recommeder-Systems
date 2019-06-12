
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import operator
import numpy as np


# In[2]:


pickle_u_i_bm = pickle.load(open("/home/shrinidhikr/Downloads/Recommender-Systems-Collaborative_Filtering-Regression_User_Based-master/ps_user_edited_all_dump1_20.pickle","rb"))
print("Root mean square error =", pickle_u_i_bm[0][1])
df = pd.read_csv('/home/shrinidhikr/Documents/Collaborative_Filtering_Recommender_Systems-master/ml-latest-small/322*30movie_ratings.csv')


# In[3]:


id_mappings = pd.read_csv('/home/shrinidhikr/Documents/Collaborative_Filtering_Recommender_Systems-master/ml-latest-small/movies.csv')
id_mappings.head(10)
id_mappings.set_index('movieId',inplace=True)


# In[4]:


result_data = pd.DataFrame(pickle_u_i_bm[0][0])
result_data['moviesId'] = df['moviesId']
result_data.set_index('moviesId',inplace=True)
df.set_index('moviesId',inplace=True)


# In[5]:


result_data.head()


# In[6]:


user_str = 'user'
user_ids = []
for i in range(1,result_data.shape[1]+1):
    user_ids.append(user_str+str(i))

result_data.columns=user_ids
df.columns=user_ids


# In[7]:


file = open("finalprocessed_predicted_matrix.pickle","wb")
pickle.dump(result_data,file)
file.close()

file_ = open("processed_original_data.pickle","wb")
pickle.dump(df,file_)
file_.close()


# In[8]:


def recommender(usr_id):

    result_data = pickle.load(open("finalprocessed_predicted_matrix.pickle","rb"))
    df = pickle.load(open("processed_original_data.pickle","rb"))
    
    ordf = pd.DataFrame()
    ordf[usr_id] = df[usr_id]
    ordf.dropna(axis=0,inplace=True)
    rlist = ordf.index.values.tolist()
    
    nrdf = pd.DataFrame()
    nrdf[usr_id] = result_data[usr_id]
    try:
        nrdf.drop(rlist,inplace=True)
    except:
        None
    
    r_dict = nrdf.T.to_dict('index')
    r_dict = r_dict[usr_id]
    sorted_dict = sorted(r_dict.items(), key=operator.itemgetter(1),reverse=True)
    
    top_n = sorted_dict[0:4]
    name_list = []
    for each_rec in top_n:
        name_list.append(id_mappings.loc[each_rec[0],:].tolist()[0])
    
    return name_list


# In[9]:


print(recommender('user12'))


# In[10]:


def rated_before(usr_id):
    df = pickle.load(open("processed_original_data.pickle","rb"))
    rated_userdf = pd.DataFrame()
    rated_userdf[usr_id] = df[usr_id]
    
    rated_userdf.dropna(axis=0,inplace=True)
    r_rated = rated_userdf.T.to_dict('index')
    r_rated = r_rated[usr_id]
    sorted_rated_dict = sorted(r_rated.items(), key=operator.itemgetter(1),reverse=True)
    
    top_n_rated = sorted_rated_dict[0:4]
    name_list_rated = []
    for each_rec in top_n_rated:
        name_list_rated.append(id_mappings.loc[each_rec[0],:].tolist()[0])
    
    return name_list_rated


# In[11]:


print(rated_before('user12'))

