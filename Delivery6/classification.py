#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.feature_selection import mutual_info_classif     # First Selection Method
from sklearn.feature_selection import f_classif               # Second Selection Method
from sklearn.feature_selection import SelectKBest             # Third Selection Method
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split          # Import train_test_split function
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


dff = pd.read_excel('Desktop/proteinDataSet.xlsx')                 # Reading our excel dataset
dfColumns = dff.columns                                        # We keep column names for future use
df = dff.values     
X = df[:,0:7597]                                              # Values of the features w/o labels
y = df[:, -1]                                                 # Values of last column, which are labels

                                                              # Splitting our dataset: 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

selectionWithMutualInfo = []
selectionWithFStats = []

# First, we'll see the selected features with mutual information selection method
  #print("FEATURES WITH MUTUAL INFO CLASSIF: ")    
feature_scores = mutual_info_classif(X_train, y_train)
for score, fname in sorted(zip(feature_scores, dfColumns), reverse=True)[:20]:
    #print(fname, score)
    selectionWithMutualInfo.append(fname)

# Second, we'll see the selected features with F-Statistics method
  #print("FEATURES WITH F_CLASSIF: ")    
feature_scores = f_classif(X_train, y_train)[0]
for score, fname in sorted(zip(feature_scores, dfColumns), reverse=True)[:20]:
    #print(fname, score)
    selectionWithFStats.append(fname)
    
#Third, we will select the features using chi square
test = SelectKBest(score_func=chi2, k=20)
#Fit the function for ranking the features by score
fit = test.fit(X, y)
#Summarize scores numpy.set_printoptions(precision=3) print(fit.scores_)
#Apply the transformation on to dataset
feature_scores = fit.transform(X)




# Now, we'll extract the selected features from the original dataset, and create discrete two dataset with them
col1 = dff[selectionWithMutualInfo[0]]
col2 = dff[selectionWithMutualInfo[1]]
col3 = dff[selectionWithMutualInfo[2]]
col4 = dff[selectionWithMutualInfo[3]]
col5 = dff[selectionWithMutualInfo[4]]
col6 = dff[selectionWithMutualInfo[5]]
col7 = dff[selectionWithMutualInfo[6]]
col8 = dff[selectionWithMutualInfo[7]]
col9 = dff[selectionWithMutualInfo[8]]
col10 = dff[selectionWithMutualInfo[9]]
col11 = dff[selectionWithMutualInfo[10]]
col12 = dff[selectionWithMutualInfo[11]]
col13 = dff[selectionWithMutualInfo[12]]
col14 = dff[selectionWithMutualInfo[13]]
col15 = dff[selectionWithMutualInfo[14]]
col16 = dff[selectionWithMutualInfo[15]]
col17 = dff[selectionWithMutualInfo[16]]
col18 = dff[selectionWithMutualInfo[17]]
col19 = dff[selectionWithMutualInfo[18]]
col20 = dff[selectionWithMutualInfo[19]]
col21 = dff['label']

dfWithMutualInfo = pd.DataFrame({'I1': col1,
                   'I2': col2,
                   'I3': col3,
                   'I4': col4,
                   'I5': col5,
                   'I6': col6,
                   'I7': col7,
                   'I8': col8,
                   'I9': col9,
                   'I10': col10,
                   'I11': col11,
                   'I12': col12,
                   'I13': col13,
                   'I14': col14,
                   'I15': col15,
                   'I16': col16,
                   'I17': col17,
                   'I18': col18,
                   'I19': col19,
                   'I20': col20,
                   'label': col21})
dfWithMutualInfo.to_excel('/home/onur/Desktop/mutualInfoSelected20.xlsx')

col1 = dff[selectionWithFStats[0]]
col2 = dff[selectionWithFStats[1]]
col3 = dff[selectionWithFStats[2]]
col4 = dff[selectionWithFStats[3]]
col5 = dff[selectionWithFStats[4]]
col6 = dff[selectionWithFStats[5]]
col7 = dff[selectionWithFStats[6]]
col8 = dff[selectionWithFStats[7]]
col9 = dff[selectionWithFStats[8]]
col10 = dff[selectionWithFStats[9]]
col11 = dff[selectionWithFStats[10]]
col12 = dff[selectionWithFStats[11]]
col13 = dff[selectionWithFStats[12]]
col14 = dff[selectionWithFStats[13]]
col15 = dff[selectionWithFStats[14]]
col16 = dff[selectionWithFStats[15]]
col17 = dff[selectionWithFStats[16]]
col18 = dff[selectionWithFStats[17]]
col19 = dff[selectionWithFStats[18]]
col20 = dff[selectionWithFStats[19]]
col21 = dff['label']

dfWithFStats = pd.DataFrame({'I1': col1,
                   'I2': col2,
                   'I3': col3,
                   'I4': col4,
                   'I5': col5,
                   'I6': col6,
                   'I7': col7,
                   'I8': col8,
                   'I9': col9,
                   'I10': col10,
                   'I11': col11,
                   'I12': col12,
                   'I13': col13,
                   'I14': col14,
                   'I15': col15,
                   'I16': col16,
                   'I17': col17,
                   'I18': col18,
                   'I19': col19,
                   'I20': col20,
                   'label': col21})
dfWithFStats.to_excel('/home/onur/Desktop/fStatsSelected.xlsx')


col1 = dff['ABHD12B']
col2 = dff['ABLIM1']
col3 = dff['ADAP2']
col4 = dff['AFF4']
col5 = dff['AGO1']
col6 = dff['ALDH3A1']
col7 = dff['ALDH3A2']
col8 = dff['ALOX15']
col9 = dff['AMOT']
col10 = dff['ANAPC16']
col11 = dff['ANKRD45']
col12 = dff['APEX2']
col13 = dff['AQP5']
col14 = dff['ATP2B3']
col15 = dff['AURKB']
col16 = dff['BARX1']
col17 = dff['C22orf46']
col18 = dff['CA2']
col19 = dff['CCDC134']
col20 = dff['CCDC186']
col21 = dff['label']

dfWithChiSquare = pd.DataFrame({'I1': col1,
                   'I2': col2,
                   'I3': col3,
                   'I4': col4,
                   'I5': col5,
                   'I6': col6,
                   'I7': col7,
                   'I8': col8,
                   'I9': col9,
                   'I10': col10,
                   'I11': col11,
                   'I12': col12,
                   'I13': col13,
                   'I14': col14,
                   'I15': col15,
                   'I16': col16,
                   'I17': col17,
                   'I18': col18,
                   'I19': col19,
                   'I20': col20,
                   'label': col21})
dfWithChiSquare.to_excel('/home/onur/Desktop/chiSquareSelected20.xlsx')



# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
#clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Evaluation For Decision Tree Without Feature Selection")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print 'F1 score Macro:', metrics.f1_score(y_test, y_pred, average="macro")
print 'F1 score Micro:', metrics.f1_score(y_test, y_pred, average="micro")

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)
#Train the model using the training sets
knn.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = knn.predict(X_test)
print("Evaluation For KNN Classification Without Feature Selection")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print 'F1 score Macro:', metrics.f1_score(y_test, y_pred, average="macro")
print 'F1 score Micro:', metrics.f1_score(y_test, y_pred, average="micro")


GausNB = GaussianNB()
GausNB.fit(X_train,y_train)
y_pred = GausNB.predict(X_test)
print("Evaluation For Gaussian NB Without Feature Selection")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print 'F1 score Macro:', metrics.f1_score(y_test, y_pred, average="macro")
print 'F1 score Micro:', metrics.f1_score(y_test, y_pred, average="micro")
















dff = pd.read_excel('Desktop/mutualInfoSelected20.xlsx')                 # Reading our excel dataset
dfColumns = dff.columns                                        # We keep column names for future use
df = dff.values     
X = df[:,0:20]                                              # Values of the features w/o labels
y = df[:, -1]                                                 # Values of last column, which are labels


                                                              # Splitting our dataset: 60% training and 40% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
#clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Evaluation For Decision Tree With Mutual Info Selection")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print 'F1 score Macro:', metrics.f1_score(y_test, y_pred, average="macro")
print 'F1 score Micro:', metrics.f1_score(y_test, y_pred, average="micro")




dff = pd.read_excel('Desktop/fStatsSelected.xlsx')                 # Reading our excel dataset
dfColumns = dff.columns                                        # We keep column names for future use
df = dff.values     
X = df[:,0:20]                                              # Values of the features w/o labels
y = df[:, -1]                                                 # Values of last column, which are labels


                                                              # Splitting our dataset: 60% training and 40% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
print("Evaluation For KNN Classification With FStats Selection")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print 'F1 score Macro:', metrics.f1_score(y_test, y_pred, average="macro")
print 'F1 score Micro:', metrics.f1_score(y_test, y_pred, average="micro")



dff = pd.read_excel('Desktop/chiSquareSelected20.xlsx')                 # Reading our excel dataset
dfColumns = dff.columns                                                 # We keep column names for future use
df = dff.values     
X = df[:,0:20]                                              # Values of the features w/o labels
y = df[:, -1]                                                 # Values of last column, which are labels


                                                              # Splitting our dataset: 60% training and 40% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

GausNB = GaussianNB()
GausNB.fit(X_train,y_train)
y_pred = GausNB.predict(X_test)
print("Evaluation For Gaussian NB With Chi Square")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print 'F1 score Macro:', metrics.f1_score(y_test, y_pred, average="macro")
print 'F1 score Micro:', metrics.f1_score(y_test, y_pred, average="micro")


# In[ ]:




