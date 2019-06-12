# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:20:25 2019

@author: sahil
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

data = pd.read_csv('books.csv', encoding = "ISO-8859-1")
data = data.dropna()
data = data[data.Author!='unknown']
data = data.drop_duplicates(subset=None, keep='first', inplace=False)
data.to_csv('books1.csv',index=False)

data1 = pd.read_csv('books1.csv',encoding = "ISO-8859-1")
for i in range(len(data1['title'])):
    data1['title'][i] = re.sub(r"\([^)]*\)","",data1['title'][i])
 
data1.to_csv('books2.csv',index=False)    
