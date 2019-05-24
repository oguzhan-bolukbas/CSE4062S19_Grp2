import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif  # First Selection Method
from sklearn.feature_selection import f_classif  # Second Selection Method
from sklearn.feature_selection import SelectKBest  # Third Selection Method
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

dff = pd.read_excel('proteinDataSet.xlsx')  # Reading our excel dataset
dfColumns = dff.columns  # We keep column names for future use
df = dff.values
X = df[:, 0:7597]  # Values of the features w/o labels
y = df[:, -1]  # Values of last column, which are labels

# Splitting our dataset: 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

selectionWithMutualInfo = []
selectionWithFStats = []

# First, we'll see the selected features with mutual information selection method
# print("FEATURES WITH MUTUAL INFO CLASSIF: ")
feature_scores = mutual_info_classif(X_train, y_train)
for score, fname in sorted(zip(feature_scores, dfColumns), reverse=True)[:20]:
    # print(fname, score)
    selectionWithMutualInfo.append(fname)

# Second, we'll see the selected features with F-Statistics method
# print("FEATURES WITH F_CLASSIF: ")
feature_scores = f_classif(X_train, y_train)[0]
for score, fname in sorted(zip(feature_scores, dfColumns), reverse=True)[:20]:
    # print(fname, score)
    selectionWithFStats.append(fname)

# Third, we will select the features using chi square
test = SelectKBest(score_func=chi2, k=20)
# Fit the function for ranking the features by score
fit = test.fit(X, y)
# Summarize scores numpy.set_printoptions(precision=3) print(fit.scores_)
# Apply the transformation on to dataset
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
dfWithMutualInfo.to_excel('mutualInfoSelected20.xlsx')

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
dfWithFStats.to_excel('fStatsSelected.xlsx')

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
dfWithChiSquare.to_excel('chiSquareSelected20.xlsx')


# END OF FEATURE SELECTION, LET'S GO TO THE CLASSIFICATION PART
def getScore(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # print(metrics.f1_score(y_test, y_pred, average="macro"))
    return model.score(X_test, y_test)


def getF1Macro(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metrics.f1_score(y_test, y_pred, average="macro")


def getF1Micro(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metrics.f1_score(y_test, y_pred, average="micro")


# CLASSIFICATION WITHOUT FEATURE SELECTION
decisionTreeWithoutFeatureSelection = []
knnWithoutFeatureSelection = []
gaussianWithoutFeatureSelection = []
dTWithFeatureSelection = []
dTWithFeatureSelection2 = []
knnWithFeatureSelection = []
knnWithFeatureSelection2 = []
gausWithFeatureSelection = []

f1mac_1 = []
f1mac_2 = []
f1mac_3 = []
f1mac_4 = []
f1mac_5 = []
f1mac_6 = []
f1mac_7 = []
f1mac_8 = []

f1mic_1 = []
f1mic_2 = []
f1mic_3 = []
f1mic_4 = []
f1mic_5 = []
f1mic_6 = []
f1mic_7 = []
f1mic_8 = []

clf = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=7)
knn2 = KNeighborsClassifier(n_neighbors=5)
GausNB = GaussianNB()

folds = KFold(n_splits=10)
# folds = StratifiedKFold(n_splits=3)
scores_DT = []
scores_KNN = []
scores_Gaus = []
for train_index, test_index in folds.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    # print(getScore(clf, X_train, X_test, y_train, y_test))
    # print(getScore(knn, X_train, X_test, y_train, y_test))
    # print(getScore(GausNB, X_train, X_test, y_train, y_test))
    decisionTreeWithoutFeatureSelection.append(getScore(clf, X_train, X_test, y_train, y_test))
    f1mac_1.append(getF1Macro(clf, X_train, X_test, y_train, y_test))
    f1mic_1.append(getF1Micro(clf, X_train, X_test, y_train, y_test))
    knnWithoutFeatureSelection.append(getScore(knn, X_train, X_test, y_train, y_test))
    f1mac_2.append(getF1Macro(knn, X_train, X_test, y_train, y_test))
    f1mic_2.append(getF1Micro(knn, X_train, X_test, y_train, y_test))
    gaussianWithoutFeatureSelection.append(getScore(GausNB, X_train, X_test, y_train, y_test))
    f1mac_3.append(getF1Macro(GausNB, X_train, X_test, y_train, y_test))
    f1mic_3.append(getF1Micro(GausNB, X_train, X_test, y_train, y_test))

dff = pd.read_excel('mutualInfoSelected20.xlsx')  # Reading our excel dataset
dfColumns = dff.columns  # We keep column names for future use
df = dff.values
X = df[:, 0:20]  # Values of the features w/o labels
y = df[:, -1]  # Values of last column, which are labels

print("\n")
print("\n")
print("\n")
for train_index, test_index in folds.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    # print(getScore(clf, X_train, X_test, y_train, y_test))
    dTWithFeatureSelection.append(getScore(clf, X_train, X_test, y_train, y_test))
    f1mac_4.append(getF1Macro(clf, X_train, X_test, y_train, y_test))
    f1mic_4.append(getF1Micro(clf, X_train, X_test, y_train, y_test))

dff = pd.read_excel('fStatsSelected.xlsx')  # Reading our excel dataset
dfColumns = dff.columns  # We keep column names for future use
df = dff.values
X = df[:, 0:20]  # Values of the features w/o labels
y = df[:, -1]  # Values of last column, which are labels

print("\n")
print("\n")
print("\n")
for train_index, test_index in folds.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    # print(getScore(knn, X_train, X_test, y_train, y_test))
    knnWithFeatureSelection.append(getScore(knn, X_train, X_test, y_train, y_test))
    f1mac_5.append(getF1Macro(knn, X_train, X_test, y_train, y_test))
    f1mic_5.append(getF1Micro(knn, X_train, X_test, y_train, y_test))
    knnWithFeatureSelection2.append(getScore(knn2, X_train, X_test, y_train, y_test))
    f1mac_6.append(getF1Macro(knn2, X_train, X_test, y_train, y_test))
    f1mic_6.append(getF1Micro(knn2, X_train, X_test, y_train, y_test))

dff = pd.read_excel('chiSquareSelected20.xlsx')  # Reading our excel dataset
dfColumns = dff.columns  # We keep column names for future use
df = dff.values
X = df[:, 0:20]  # Values of the features w/o labels
y = df[:, -1]  # Values of last column, which are labels

print("\n")
print("\n")
print("\n")
for train_index, test_index in folds.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    # print(getScore(GausNB, X_train, X_test, y_train, y_test))
    gausWithFeatureSelection.append(getScore(GausNB, X_train, X_test, y_train, y_test))
    f1mac_7.append(getF1Macro(GausNB, X_train, X_test, y_train, y_test))
    f1mic_7.append(getF1Micro(GausNB, X_train, X_test, y_train, y_test))
    dTWithFeatureSelection2.append(getScore(clf, X_train, X_test, y_train, y_test))
    f1mac_8.append(getF1Macro(clf, X_train, X_test, y_train, y_test))
    f1mic_8.append(getF1Micro(clf, X_train, X_test, y_train, y_test))

print("The Score Of Decision Tree Without Feature Selection:")
print("Accuracy: ", np.mean(decisionTreeWithoutFeatureSelection))
print("F1 Macro: ", np.mean(f1mac_1))
print("F1 Micro: ", np.mean(f1mic_1))

print("\n")
print("The Score Of kNN Algorithm Feature Selection(k=7):")
print("Accuracy: ", np.mean(knnWithoutFeatureSelection))
print("F1 Macro: ", np.mean(f1mac_2))
print("F1 Micro: ", np.mean(f1mic_2))

print("\n")
print("The Score Of Gaussian Naive Bayes Without Feature Selection:")
print("Accuracy: ", np.mean(gaussianWithoutFeatureSelection))
print("F1 Macro: ", np.mean(f1mac_3))
print("F1 Micro: ", np.mean(f1mic_3))

print("The Score Of Decision Tree With Feature Selection(Mutual Info Gain Selected):")
print("\n")
print("Accuracy: ", np.mean(dTWithFeatureSelection))
print("F1 Macro: ", np.mean(f1mac_4))
print("F1 Micro: ", np.mean(f1mic_4))

print("The Score Of Decision Tree With Feature Selection(Chi Square Selected):")
print("\n")
print("Accuracy: ", np.mean(dTWithFeatureSelection2))
print("F1 Macro: ", np.mean(f1mac_8))
print("F1 Micro: ", np.mean(f1mic_8))

print("\n")
print("The Score Of kNN With Feature Selection(k=7)(FStats Selected):")
print("Accuracy: ", np.mean(knnWithFeatureSelection))
print("F1 Macro: ", np.mean(f1mac_5))
print("F1 Micro: ", np.mean(f1mic_5))

print("\n")
print("The Score Of kNN With Feature Selection(k=5)(FStats Selected):")
print("Accuracy: ", np.mean(knnWithFeatureSelection2))
print("F1 Macro: ", np.mean(f1mac_6))
print("F1 Micro: ", np.mean(f1mic_6))

print("\n")
print("The Score Of Gaussian Naive Bayes With Feature Selection(Chi Square Selected):")
print("Accuracy: ", np.mean(gausWithFeatureSelection))
print("F1 Macro: ", np.mean(f1mac_7))
print("F1 Micro: ", np.mean(f1mic_7))

