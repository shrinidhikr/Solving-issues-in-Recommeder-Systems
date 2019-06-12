# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:42:49 2019

@author: sahil
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk,re
from numpy.linalg import norm
from sklearn import linear_model
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from gensim.models import Word2Vec,Phrases
lemmatizer = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords  
#nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

data = pd.read_csv('bookdata.csv', encoding = "ISO-8859-1")
#d = pd.DataFrame({"title":[Title],"Genre":[Genre],"Author":[Author],"Description":[Description]})
#data = data.append(d,ignore_index = True)

for i in range(len(data['Author'])):
    data['Author'][i] = data['Author'][i].replace(" ","")
    
title = data['title']
#add all data to col as Alldata
Data = data['Gener']+data['Author']+data['Description']
data['Alldata'] = Data

normalize_data =[]
clean_pos=[]
#def normalize():
for i in data['Alldata']:
    clean_data =  re.sub('[^a-zA-Z]',' ',i)
    clean_data = re.sub(r'\b[uU]\b','you',clean_data)
    clean_data=clean_data.lower()#
    #ps = PorterStemmer()
    #clean_data = [ps.stem(word) for word in clean_data if not word in stopwords.words('english')] 
    
    clean_data = [lemmatizer.lemmatize(word, pos='v') for word in clean_data.split() if not word in stopwords.words('english')]
    ''' for word,tag in nltk.pos_tag(clean_data):
        if tag.startswith("NN"):
             clean_pos.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
             clean_pos.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
             clean_pos.append(lemmatizer.lemmatize(word, pos='a'))
        else:
             clean_pos.append(word)
     '''    
    clean_data =' '.join( clean_data )
    normalize_data.append(clean_data)
    clean_pos = []
    


doc = []
for s in normalize_data:
    doc.append(s.split())        
        
#need pass bigrams in word2vec

#phrases = Phrases(pos_tag_nor_data, min_count=1, threshold=2)
#bigram = Phrases(phrases)
#def word2vect_and_doc2vec_matrix():
model2 = Word2Vec(doc, min_count = 1, size = 100,window = 6, sg = 1)         
print(model2)
print((model2.wv.vocab))

word2vec_dict = dict({})
for idx, key in enumerate(model2.wv.vocab):
    word2vec_dict[key] = model2.wv[key]
print(word2vec_dict)    
#sim_words = model2.wv.most_similar('marvel_nn')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(norm=None,ngram_range=(1,1))    
new_term_freq_matrix = tfidf.fit_transform(normalize_data)
print(new_term_freq_matrix)

# create dictionary to find a tfidf word each word
word2tfidf = dict(zip(tfidf.get_feature_names(),tfidf.idf_))
for word, score in word2tfidf.items():
    print(word, score)
 
doc2vec_matrix = []
           
for sent in normalize_data:
    c =0 
    sum_word_vectors = 0
    for word in sent.split():
        if(len(word)!=1):
            multipywordtfidf = np.multiply(word2tfidf[word],word2vec_dict[word])
            sum_word_vectors = np.add(sum_word_vectors, multipywordtfidf)
            c = c+1
    print(sum_word_vectors)
    print(norm(sum_word_vectors))    
    doc2vec_matrix.append(sum_word_vectors/c)
       

doc2vec_matrix = np.array(doc2vec_matrix)
#return doc2vec_matrix     
pickle_out = open("doc2vec1.pickle","wb")
pickle.dump(doc2vec_matrix,pickle_out)
pickle_out.close()
#similarity
#def cosinemat():        
#return cosine_sim
a = -0.01
Title ="Death Note, Vol. 5" 
titles = list(title)
t = []
for i in titles:
     i = re.sub('[(\xa0)(\xa0\xa0)(Ã\x83Â\x82Ã\x82ÂÃ\x83Â\x82Ã\x82Â)]','',i)
     t.append(i)
avg=-5    
print(t)
pickle_out = open("titles1.pickle","wb")
pickle.dump(t,pickle_out)
pickle_out.close()

index = t.index(Title)
scores = {}
for i in range(len(Data)):
    cosine_sim = 1-spatial.distance.cosine(doc2vec_matrix[i],doc2vec_matrix[index])
    scores[i] = cosine_sim
scores = sorted(scores.items(),key=lambda kv: kv[1],reverse=True)
print(scores)

recommd = []
top_recomm =8;
for i in range(top_recomm):
    recommd.append(titles[scores[i+1][0]])

print(recommd)    


for i in range(349):
    cosine_sim = 1-spatial.distance.cosine(doc2vec_matrix[i+1],doc2vec_matrix[0])
    print(cosine_sim)
    
#test
indexes = []
for i in recommd:
    indexes.append(t.index(i))
avg = 0
c = 0
for i in indexes:
    for j in indexes:
        if i != j:
            cosine_sim = 1-spatial.distance.cosine(doc2vec_matrix[i],doc2vec_matrix[j])+a
            avg = np.add(avg,cosine_sim)
            c = c+1
            print("inter similarity "+str(t[i])+" "+str(t[j])+" "+str(cosine_sim))
print("Average of inetr-similarity is"+str(avg/c))            
print()
print("diversity of recommendation system"+str(1-(avg/c)))
        