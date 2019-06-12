# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:26:22 2019

@author: sahil
"""
import pickle
import flask
import numpy as np
import pandas as pd
import operator
from flask import Flask, flash, redirect, render_template, request, session, abort,url_for
import os
from scipy import spatial


APP = flask.Flask(__name__)

@APP.route('/')
def index():
     return render_template('login.html')

@APP.route('/register')
def register():
    return render_template('register.html')

@APP.route('/dashboard')
def dashboard():
    print("hello")
    return render_template('dashboard.html')   


@APP.route('/dashboard',methods=['POST'])
def booksdisp():
        pickle_in = open("doc2vec1.pickle","rb")
        doc2vec_matrix = pickle.load(pickle_in)

        pickle_in = open("titles1.pickle","rb")
        titles = pickle.load(pickle_in)

        print("hello1")
        x = request.form["b"]
        print(x)

        Title = x
        index = titles.index(Title)
        scores = {}
        for i in range(349):
            cosine_sim = 1-spatial.distance.cosine(doc2vec_matrix[i],doc2vec_matrix[index])
            scores[i] = cosine_sim
        scores = sorted(scores.items(),key=lambda kv: kv[1],reverse=True)
        
        recommd = []
        recomdscore = []
        top_recomm = 10;
        for i in range(top_recomm):
            if(scores[i+1][1]!=1.0):
                recommd.append(titles[scores[i+1][0]])
                recomdscore.append(scores[i+1][1]*100)               

        #recommd = np.array(recommd)
        #recomdscore = np.array(recomdscore)       
        
        return render_template('dashboard.html' ,recommd=recommd,Hidden=True)   


@APP.route('/dashboard1',methods=['POST'])
def dashboard1():
    user = request.form["username"]
    print(user)
    def rated_before(usr_id):
        id_mappings = pd.read_csv('movies.csv')
        id_mappings.head(10)
        id_mappings.set_index('movieId',inplace=True)
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
       
  
    def recommender(usr_id):
        
        result_data = pickle.load(open("finalprocessed_predicted_matrix.pickle","rb"))
        df = pickle.load(open("processed_original_data.pickle","rb"))
        id_mappings = pd.read_csv('movies.csv')
        id_mappings.head(10)
        id_mappings.set_index('movieId',inplace=True)
        
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

    name_list_rated = rated_before(user)
    name_list = recommender(user)
     

    return render_template('dashboard1.html',user=user,nlr=name_list_rated,nl=name_list,Hidden=True) 




if __name__ == '__main__':
    APP.debug=True
    APP.run()