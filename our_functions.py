# Import required libraries
import pandas as pd
import numpy as np
import time
import missingno as msno 
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
import ast
import re
import itertools
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import random
from scipy.spatial import distance as dist
nltk.download('averaged_perceptron_tagger')

# Function to obtain (randomly) the initial clusters for the k_means algorithm
def get_initial_centroids(seed, k, data):
    # seed: integer  value to define a seed to get k-means reproducible
    # k: integer number of clusters to be used
    # data: Array dataset to be used
    
    # Set the random seed
    random.seed(seed)
    
    # Get k random initial centroids from the data
    centroids = pd.DataFrame(data).loc[random.sample(range(0, len(data)), k)]
        
    return(centroids)

# Function to calculate the k-means algorithm
def our_kmeans(data, k, max_iter = 200, conv = 0.000001, our_seed = 0):
    # data: Array dataset to be used
    # k: integer number of clusters to be used
    # max_iter: maximum number of iterations allowed
    # conv: minimum value (epsilon) for convergence criteria
    # our_seed: integer  value to define a seed to get k-means reproducible
   
    # Define the initial value of i to start iterations
    i = 0
    
    # Define the initial diference between step n and n+1 to start iterating
    dif_clusters = 1
    
    # Calculate the intial random centroids
    centroids = get_initial_centroids(our_seed, k, data)
    
    # Find the closest random cluster that the data belongs to (each example of the data)
    cluster = np.argmin(dist.cdist(data, centroids, 'euclidean'),axis=1)
    
    # Loop until get convergence
    while i <= max_iter and dif_clusters > conv:
        # Calculate the centroids of each cluster with the data provided
        centroids = np.vstack([data[cluster == i, :].mean(axis = 0) for i in range(k)])
        
        # Find the closest cluster that the data belongs to (each example of the data)
        clusters_aux = np.argmin(dist.cdist(data, centroids, 'euclidean'), axis = 1)
        
        # Calculate the convergence regarding the previous clusters with the recalculated clusters
        dif_clusters = np.linalg.norm(np.subtract(clusters_aux, cluster))
        
        # Increment by 1 the counter of iterations
        i += 1
        
        # Replace the old cluster with the new cluster
        cluster = clusters_aux
    
    # Calculate the inertia after the algorithm converged
    # Concatenate to the data the clusters as a column
    aux = pd.concat([pd.DataFrame(data), pd.DataFrame(cluster, columns=['Cluster'])], axis=1)
    
    inertia = 0
    
    #loop over different values of clusters
    for i in range(k):
        # Filter the data corresponding to each cluster and remove the final one (that is the cluster)
        data_cluster = aux[aux['Cluster'] == i][aux.columns[:-1]].to_numpy()
        
        # Calculate the inertia for the final clusters
        inertia += sum(dist.cdist(data_cluster, centroids[i].reshape((1, data.shape[1])), 'euclidean')**2)
        
    return cluster, inertia


# Define the stopwords
stop_words = set(stopwords.words('english'))

# Select stemmer
stemming = PorterStemmer()

# Tokenization
def tokenize_row(row):
    tokens = nltk.word_tokenize(row)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words


# Stemming of irregular verbs
irregulars = pd.read_csv("data/irregular_verbs.csv", sep=";", encoding='utf8')

def irregular_verbs(row):
    new_row = []
    for word in row:
        if word not in irregulars['sp'].to_list() and word not in irregulars['pp'].to_list():
            new_row.append(word)
            continue
        else:
            for i in range(irregulars.shape[0]):
                if word == irregulars['sp'][i]:
                    new_row.append(irregulars['inf'][i])
                    continue
                if word == irregulars['pp'][i]:
                    new_row.append(irregulars['inf'][i])
                    continue
    return new_row

# Before stemming remove capitalization and retain only terms that are more than 2 letters
def stem_row(row):
    stemmed_list = [word.lower() for word in row if len(word.lower())>2]
    return stemmed_list

# Remove stopwords
def rm_stopwords_row(row):
    meaningful_words = [w.lower() for w in row if not w.lower() in stop_words]
    return meaningful_words


# Before stemming remove capitalization and retain only terms that are more than 2 letters
def avoid_words(row):
    avoid_types = ['JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ']
    tagged = nltk.pos_tag(row)
#     result = [i[0] for i in tagged if i[1] != 'JJ' and i[1] !='JJR' and i[1] !='JJS' and i[1] !='VB' and i[1] !='VBD' and i[1] !='VBG' and i[1] !='VBN' and i[1] !='VBP' and i[1] !='VBZ']
    result = [word for (word,tag) in tagged if tag != 'JJ' and tag !='JJR' and tag !='JJS' and tag !='VB' and tag !='VBD' and tag !='VBG' and tag !='VBN' and tag !='VBP' and tag !='VBZ']
    return result