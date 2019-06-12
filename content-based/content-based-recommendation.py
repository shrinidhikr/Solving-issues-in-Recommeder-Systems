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
from sklearn import linear_model
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from gensim.models import Word2Vec,Phrases
lemmatizer = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords  
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

data = pd.read_csv('books2.csv', encoding = "ISO-8859-1")
#d = pd.DataFrame({"title":[Title],"Genre":[Genre],"Author":[Author],"Description":[Description]})
#data = data.append(d,ignore_index = True)

for i in range(len(data['Author'])):
    data['Author'][i] = data['Author'][i].replace(" ","")
    
title = data['title']
#add all data to col as Alldata
Data = data['Gener']+data['Author']+data['Description']
data['Alldata'] = Data

normalize_data =[]
#def normalize():
for i in data['Alldata']:
    clean_data =  re.sub('[^a-zA-Z]',' ',i)
    clean_data = re.sub(r'\b[uU]\b','you',clean_data)
    clean_data=clean_data.lower()
    clean_data=clean_data.split()#
    #ps = PorterStemmer()
    #clean_data = [ps.stem(word) for word in clean_data if not word in stopwords.words('english')] 
    clean_data = [lemmatizer.lemmatize(word,pos="a") for word in clean_data if not word in stopwords.words('english')]
    clean_data =' '.join( clean_data )
    normalize_data.append(clean_data)
    


doc = []
for s in normalize_data:
    doc.append(s.split())        
        
#need pass bigrams in word2vec

#phrases = Phrases(pos_tag_nor_data, min_count=1, threshold=2)
#bigram = Phrases(phrases)
#def word2vect_and_doc2vec_matrix():
model2 = Word2Vec(doc, min_count = 1, size = 300,window = 8, sg = 1)         
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
c=0           
i=0
for sent in normalize_data:
    j=0
    sum_word_vectors = 0
    for word in sent.split():
        if(len(word)!=1):
            multipywordtfidf = np.multiply(word2tfidf[word],word2vec_dict[word])
            sum_word_vectors = np.add(sum_word_vectors, multipywordtfidf)
            c = c+1
        j = j + 1
    print(sum_word_vectors)    
    doc2vec_matrix.append(sum_word_vectors/c)
    c =0    
    i = i + 1

doc2vec_matrix = np.array(doc2vec_matrix)
#return doc2vec_matrix     
pickle_out = open("doc2vec.pickle","wb")
pickle.dump(doc2vec_matrix,pickle_out)
pickle_out.close()
#similarity
#def cosinemat():        
for i in range(5):
    cosine_sim = 1-spatial.distance.cosine(doc2vec_matrix[i+1],doc2vec_matrix[0])
    print(cosine_sim)
#return cosine_sim
Title ="Death Note, Vol. 10" 
titles = list(title)
t = []
for i in titles:
     i = re.sub('[(\xa0)(\xa0\xa0)]','',i)
     t.append(i)
    
print(t)
pickle_out = open("titles.pickle","wb")
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
top_recomm =12;
for i in range(top_recomm):
    recommd.append(titles[scores[i+1][0]])

print(recommd)    
        