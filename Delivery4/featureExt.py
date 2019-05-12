import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from openpyxl import Workbook


# load data
fName = '/home/onur/Desktop/dataScienceSet.xlsx'                #File location with integer values
fName2 = '/home/onur/Desktop/orjDataSet.xlsx'                   #File location with floating real values

proteins = pd.read_excel(fName, sheet_name="Sayfa1")            #Variable to keep the dataset that was read.
orgProteins = pd.read_excel(fName2, sheet_name="Sayfa1")        #Variable to keep the dataset that was read.
array = proteins.values                                         # Taking values from the dataset

X = array[:,0:7590]
Y = array[:,7590]
# feature extraction
test = SelectKBest(score_func=chi2, k=10)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
features = fit.transform(X)


selectedTen = []                                # We keep the values of the first rows of selected columns
for x in range(10):
    selectedTen.append(features[0:1,x][0])


firstRow = []                                   # We keep the values of first row of dataset
for x in range(7590):
    firstRow.append(array[0:1,x][0])

orderList = []                                  # Now, we try to find the orders of the important features
for selectedOne in selectedTen:
    order = firstRow.index(selectedOne)
    orderList.append(order)

# Fetch the selected features and floating values
column1 = orgProteins.iloc[:,orderList[0]]
column2 = orgProteins.iloc[:,orderList[1]]
column3 = orgProteins.iloc[:,orderList[2]]
column4 = orgProteins.iloc[:,orderList[3]]
column5 = orgProteins.iloc[:,orderList[4]]
column6 = orgProteins.iloc[:,orderList[5]]
column7 = orgProteins.iloc[:,orderList[6]]
column8 = orgProteins.iloc[:,orderList[7]]
column9 = orgProteins.iloc[:,orderList[8]]
column10 = orgProteins.iloc[:,orderList[9]]

# We save the columns to another excel file
df = pd.DataFrame({'I1': column1,
                   'I2': column2,
                   'I3': column3,
                   'I4': column4,
                   'I5': column5,
                   'I6': column6,
                   'I7': column7,
                   'I8': column8,
                   'I9': column9,
                   'I10': column10,})
df.to_excel('/home/onur/Desktop/importance.xlsx')

print("1")
print(column1.corr(column1))
print(column1.corr(column2))
print(column1.corr(column3))
print(column1.corr(column4))
print(column1.corr(column5))
print(column1.corr(column6))
print(column1.corr(column7))
print(column1.corr(column8))
print(column1.corr(column9))
print(column1.corr(column10))


print("2")
print(column2.corr(column2))
print(column2.corr(column3))
print(column2.corr(column4))
print(column2.corr(column5))
print(column2.corr(column6))
print(column2.corr(column7))
print(column2.corr(column8))
print(column2.corr(column9))
print(column2.corr(column10))


print("3")
print(column3.corr(column3))
print(column3.corr(column4))
print(column3.corr(column5))
print(column3.corr(column6))
print(column3.corr(column7))
print(column3.corr(column8))
print(column3.corr(column9))
print(column3.corr(column10))

print("4")
print(column4.corr(column4))
print(column4.corr(column5))
print(column4.corr(column6))
print(column4.corr(column7))
print(column4.corr(column8))
print(column4.corr(column9))
print(column4.corr(column10))

print("5")
print(column5.corr(column5))
print(column5.corr(column6))
print(column5.corr(column7))
print(column5.corr(column8))
print(column5.corr(column9))
print(column5.corr(column10))



print("6")
print(column6.corr(column6))
print(column6.corr(column7))
print(column6.corr(column8))
print(column6.corr(column9))
print(column6.corr(column10))



print("7")
print(column7.corr(column7))
print(column7.corr(column8))
print(column7.corr(column9))
print(column7.corr(column10))


print("8")
print(column8.corr(column8))
print(column8.corr(column9))
print(column8.corr(column10))


print("9")
print(column9.corr(column9))
print(column9.corr(column10))



print("10")
print(column10.corr(column10))




# load data
fName = '/home/onur/Desktop/importance.xlsx'                #File loc variable
data = pd.read_excel(fName)            #Variable to keep the dataset that was read.

corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()


'''
# Creating correlation graph of the 10 features.

plt.matshow(importances.corr())
plt.xticks(range(len(importances.columns)), importances.columns)
plt.xticks(range(len(importances.columns)), importances.columns)
plt.colorbar()
plt.show()
'''