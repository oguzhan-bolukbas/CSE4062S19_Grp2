#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from scipy.misc import comb
from itertools import combinations


dff = pd.read_excel('proteinDataSet.xlsx')  # Reading our excel dataset
dfColumns = dff.columns  # We keep column names for future use
df = dff.values
X = df[:, 0:7597]  # Values of the features w/o labels
y = df[:, -1]  # Values of last column, which are labels

# Function to calculate silhouette values of each cluster
def silhouette_value(range_n_clusters):
    #range_n_clusters = [3, 4, 5, 6]
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        preds = clusterer.fit_predict(X)
        centers = clusterer.cluster_centers_

        score = silhouette_score(X, preds, metric='euclidean')
        print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
        
        
        
        
def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)    
    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred

def rand_score (labels_true, labels_pred):
    check_clusterings(labels_true, labels_pred)
    my_pair = list(combinations(range(len(labels_true)), 2)) #create list of all combinations with the length of labels.
    def is_equal(x):
        return (x[0]==x[1])
    my_a = 0
    my_b = 0
    for i in range(len(my_pair)):
            if(is_equal((labels_true[my_pair[i][0]],labels_true[my_pair[i][1]])) == is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) 
               and is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) == True):
                my_a += 1
            if(is_equal((labels_true[my_pair[i][0]],labels_true[my_pair[i][1]])) == is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) 
               and is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) == False):
                my_b += 1
    my_denom = comb(len(labels_true),2)
    ri = (my_a + my_b) / my_denom
    return ri


# Function to calculate number of instances in each cluster
def instance_counts(range_n_clusters):
    print('\n')
    print("Cluster Lengths With k=2: ")
    kMeans = KMeans(n_clusters=2)
    kMeans.fit(X)
    predicted_Y = list(kMeans.labels_)
    print(Counter(kMeans.labels_))
    print("STD: ", np.std(predicted_Y))
    print("NMI: ", normalized_mutual_info_score(y,predicted_Y))
    print("Rand Index: ", rand_score(y,predicted_Y))


    
    print('\n')
    print('\n')
    print("Cluster Lengths With k=3: ")
    kMeans = KMeans(n_clusters=3)
    kMeans.fit(X)
    predicted_Y = list(kMeans.labels_)
    print(Counter(kMeans.labels_))
    print("STD: ", np.std(predicted_Y))
    print("NMI: ", normalized_mutual_info_score(y,predicted_Y))
    print("Rand Index: ", rand_score(y,predicted_Y))

    print('\n')
    print('\n')
    print("Cluster Lengths With k=4: ")
    kMeans = KMeans(n_clusters=4)
    kMeans.fit(X)
    predicted_Y = list(kMeans.labels_)
    print(Counter(kMeans.labels_))
    print("STD: ", np.std(predicted_Y))
    print("NMI: ", normalized_mutual_info_score(y,predicted_Y))
    print("Rand Index: ", rand_score(y,predicted_Y))


    print('\n')
    print('\n')
    print("Cluster Lengths With k=5: ")
    kMeans = KMeans(n_clusters=5)
    kMeans.fit(X)
    predicted_Y = list(kMeans.labels_)
    print(Counter(kMeans.labels_))
    print("STD: ", np.std(predicted_Y))
    print("NMI: ", normalized_mutual_info_score(y,predicted_Y))
    print("Rand Index: ", rand_score(y,predicted_Y))


    print('\n')
    print('\n')
    print("Cluster Lengths With k=6: ")
    kMeans = KMeans(n_clusters=6)
    kMeans.fit(X)
    predicted_Y = list(kMeans.labels_)
    print(Counter(kMeans.labels_))
    print("STD: ", np.std(predicted_Y))
    print("NMI: ", normalized_mutual_info_score(y,predicted_Y))
    print("Rand Index: ", rand_score(y,predicted_Y))




instance_counts(clusterRange)
silhouette_value(clusterRange)


# In[ ]:




