# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:26:22 2019

@author: sahil
"""
import pickle
import flask
from scipy import spatial
pickle_in = open("doc2vec.pickle","rb")
doc2vec_matrix = pickle.load(pickle_in)

pickle_in = open("titles.pickle","rb")
titles = pickle.load(pickle_in)


APP = flask.Flask(__name__)

@APP.route('/')
def index():

    Title = "THE LOVERS - A Novel"
    index = titles.index(Title)
    scores = {}
    for i in range(532):
        cosine_sim = 1-spatial.distance.cosine(doc2vec_matrix[i],doc2vec_matrix[index])
        scores[i] = cosine_sim
    scores = sorted(scores.items(),key=lambda kv: kv[1],reverse=True)
    print(scores)
    recommd = []
    top_recomm = 10;
    for i in range(top_recomm):
        if(scores[i+1][1]!=1.0):
            recommd.append(titles[scores[i+1][0]])
        
    return("<p>" + "</p><p>".join(recommd) + "</p>")

if __name__ == '__main__':
    APP.debug=True
    APP.run()