# Solving data sparsity and cold start in Recommender Systems - Mini Project

  *Collaborative Filtering using Regression based approach with GWO for handling data sparsity
  
  *Content based approach with Natural Language Processing technique for solving cold start of new user and new item


# Directions for running the project  
  // After cloning the project from this GitHub Repository, follow the steps given below
  
  *Setup the environment :
      
      sudo apt-get install python-pip
      sudo pip install numpy
      sudo pip install pandas 
  
  *Installing scikit-learn on Ubuntu is easy and straightforward. You can install it either using apt-get install or pip.
    Here is the command to install scikit-learn using apt-get install:

    $ sudo apt-get install python-sklearn

  *We can install it using pip using the following command:

    $ sudo pip install scikit-learn

    After installing scikit-learn, we can test the installation by doing following commands in Python Terminal.

>>> import sklearn 
>>> sklearn.__version__ 
'0.17' 

Congratulations, you have successfully set up scikit-learn!
------------------------------------------------------------
 *Open an editor, for example spyder.
 *Later install python pickle module for storage.

*In collaborative part, 
           ---> Change the location address of the dataset to it's location as mentioned in your device.         
                Now run the python program test_file.py by using the command
                                >>python3 test_file.py
                The pack_size can be varied .
            
                
*In content part,
           --->Import the following libraries:-
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
             
             If errors occur,install the following from thee terminal :
             
             sudo pip install -U nltk
             sudo apt-get install libgtk2.0-dev
             tar xvf geany-1.25.tar.gz
             cd geany-1.25
             ./configure
             sudo make
             sudo make install
             pip3 install --upgrade gensim
             pip install nltk
             
             sudo apt-get update -y
             sudo apt-get install -y intltool
            
   *Now run the python program content-based-recommendation.py by using the command
                                >>python3 test_file.py
           
   --------------------------------------------------------------------------------------------
   
#To run the UI
    
    *Use the command 
          >>python3 Recommend-books1.py
 ----------------------------------------------------------------------------------------------         
   
 Team members - [Sneha V](https://github.com/snehavishwanatha), [Sahil MH](https://github.com/mohedsahil), [Shubhang P](https://github.com/ShubhangPK)
