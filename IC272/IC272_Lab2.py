#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

missData = pd.read_csv('C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn2\\landslide_data3_miss.csv') # Reading a CSV file using pandas
missNumber = dict.fromkeys(list(missData.columns), 0) # Dictionary that contains the record for number of missing values.
columnName = list(missData.columns)
for i in range(len(missData.columns)):
    for j in range(len(missData[columnName[i]])):
        if pd.isnull(missData[columnName[i]][j]): #Checking if the value is nan or not.
            missNumber[columnName[i]] += 1
            
plt.figure(figsize=(15,7)) #Resizing graph to avoid overwriting of labels.
# 1
for i in range(len(columnName)):
    plt.bar(columnName[i], missNumber[columnName[i]]) #Ploting a bar graph
    plt.text(x=i-0.1, y=missNumber[columnName[i]]+0.5, s= str(missNumber[columnName[i]])) # Showing Value on top of each bar.
    
plt.xticks([i for i in range(len(columnName))], columnName)
plt.xlabel("Attributes -->")
plt.ylabel("Number of missing values -->")
plt.legend(columnName)
plt.show()


# In[2]:


# 2
import numpy as np
# a
stationIDdata = missData[missData["stationid"].notnull()] # notnull() function removes all the null value rows.
print(stationIDdata)
print("Number of Values missing:", missNumber['stationid']) # Prints the number of values missing from this attribute.

# b
# there are 8 rows (excluding date), so if have 3 or more values == NaN in a row, we will remove it.


# In[3]:


flag = 0
indexLst = []
for i in range(len(missData["dates"])):
    flag = 0
    for j in columnName:
        if pd.isnull(missData[j][i]):
            flag += 1
    if flag >= 3:
        indexLst.append(i) # This list contains the index values of rows that we need to drop.


# In[4]:


cleanData = missData.drop(missData.index[indexLst]).reset_index(drop=True) # Resets the index count 
print(cleanData)
print("Total rows dropped:", len(missData['dates']) - len(cleanData.dates)) # Total number of Rows deleted.


# In[5]:


# 3
anotherDic = dict.fromkeys(columnName, 0) # This Dictionary will contain the new data's number of deleted rows.
for i in range(len(columnName)):
    for j in range(len(cleanData[columnName[i]])):
        if pd.isnull(cleanData[columnName[i]][j]): #Checking if the value is nan or not.
            anotherDic[columnName[i]] += 1
            
cleanTable = pd.DataFrame({'Attributes': columnName,
                        'Number of Missing Values': anotherDic.values()}) # Creating a table for better understanding
print(cleanTable)
print("Total number of missing values:", sum(anotherDic.values()))


# In[6]:


add = 0
cnt = 0
editCol = columnName[2:] # Taking columns containing numeric values for finding mean
meanDic = dict.fromkeys(editCol, 0)
for i in range(len(editCol)):
    add = 0
    cnt = 0
    for j in range(len(missData[editCol[i]])):
        if pd.isnull(missData[editCol[i]][j]) == False: # Removing all the NaN values for calculating mean. 
             add += missData[editCol[i]][j]
             cnt += 1
    meanDic[editCol[i]] = add/cnt # This dictionary contains mean of all the attributes.
print(meanDic)


# In[7]:


convertedData = missData # Taking another variable for ease in calculation.
for i in editCol:
    convertedData[i] = convertedData[i].fillna(meanDic[i]) # Filling all the series of attributes
print(convertedData)


# In[8]:


# Before -- file with missing values
# As we know that mean calculated to fill data is the mean of the original data.
from scipy.stats import mode

originalData = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn2\\landslide_data3_original.csv")

meanBefore = pd.Series(meanDic.values())

medianBefore = pd.Series([missData[i].median(skipna=True) for i in editCol])

modeBefore = pd.Series([missData[i].mode(dropna=True)[0] for i in editCol]) # Taking 0 give us the mode by iterating through index.

stdBefore = pd.Series([missData[i].std(skipna=True) for i in editCol])

beforeData = pd.DataFrame({'Attributes': editCol,
                          'Mean': meanBefore,
                          'Median': medianBefore,
                          'Mode': modeBefore,
                          'S.D.': stdBefore})
print("File with Missing Values")
print(beforeData)
# After -- original file
meanAfter = pd.Series([np.mean(originalData[i]) for i in editCol])

medianAfter = pd.Series([np.median(originalData[i]) for i in editCol])

modeAfter = pd.Series([float(mode(originalData[i])[0]) for i in editCol]) # Using scipy as np doesn't have mode

stdAfter = pd.Series([np.std(originalData[i]) for i in editCol])

afterData = pd.DataFrame({'Attributes': editCol,
                          'Mean': meanAfter,
                          'Median': medianAfter,
                          'Mode': modeAfter,
                          'S.D.': stdAfter})
print("\nOriginal File")
print(afterData)


# In[9]:


from math import sqrt
missData = pd.read_csv('C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn2\\landslide_data3_miss.csv') 

missIndex = dict.fromkeys(editCol, 0) # Dictionary that contains indices of missing values in missData
for i in editCol:
    ind = [] # List that contains indices of missing values.
    for j in range(len(missData[i])):
        if pd.isnull(missData[i][j]):
            ind.append(j)
    missIndex[i] = ind

# RMSE
rmseDic = dict.fromkeys(editCol, 0) # Dictionary that contains RMSE values of each attributes.
for i in editCol:
    rmse = 0 
    for j in missIndex[i]:
        rmse += (meanDic[i]- originalData[i][j])*(meanDic[i]- originalData[i][j])
    rmseDic[i] = sqrt(rmse/len(missIndex[i]))
print("RMSE of the attributes are:")
print(rmseDic)
plt.figure(figsize=(8, 8))
plt.bar(editCol, rmseDic.values(), log=True) # Taking log scale for better readability of the graph.
plt.xlabel("Attributes -->")
plt.ylabel("RMSE Values -->")
plt.show()


# In[10]:


# Interpolation
interData = pd.DataFrame.interpolate(missData)
meanInter = pd.Series([np.mean(interData[i]) for i in editCol])

medianInter = pd.Series([np.median(interData[i]) for i in editCol])

modeInter = pd.Series([float(mode(interData[i])[0]) for i in editCol]) # Using scipy as np doesn't have mode

stdInter = pd.Series([np.std(interData[i]) for i in editCol])

interpolationData = pd.DataFrame({'Attributes': editCol,
                          'Mean': meanInter,
                          'Median': medianInter,
                          'Mode': modeInter,
                          'S.D.': stdInter})
print("Interpolated Data")
print(interpolationData)
print('\nOriginal Data')
print(afterData)


# In[11]:


# RMSE
rmseInter = dict.fromkeys(editCol, 0)
for i in editCol:
    rmse = 0
    for j in missIndex[i]:
        rmse += (interData[i][j] - originalData[i][j])*(interData[i][j] - originalData[i][j])
    rmseInter[i] = sqrt(rmse/len(missIndex[i]))
print("RMSE of interpolated data is:")
print(rmseInter)
plt.figure(figsize=(8, 7))
plt.bar(editCol, rmseInter.values(), log=True)
plt.show()


# In[12]:


# 5
# Outliers
# a
def outliers(x):
    q1 = interData[x].quantile(0.25) # First Quartile
    q3 = interData[x].quantile(0.75) # Third Quartile
    iqr = q3 - q1 # Inter-Quartile Range
    outlierLst = [] # List that contains index of outliers.
    for i in range(len(interData[x])):
        if interData[x][i] < q1 - 1.5*iqr or interData[x][i] > q3 + 1.5*iqr:
            outlierLst.append(i) # This List contains indices of the outliers.
    print(f"Number of outliers in {x}:", len(outlierLst))
    print("IQR:", iqr)
    print("Variance:", np.var(interData[x]))
    plt.boxplot(interData[x])
    plt.show()

outliers('temperature')
outliers('rain')


# In[14]:


# 5 b
def replaceWithMedian(x):
    q1 = interData[x].quantile(0.25)
    q3 = interData[x].quantile(0.75)
    iqr = q3 - q1
    outliers = 0
    med = np.median(interData[x])
    y = interData[x]
    for i in range(len(y)):
        if y[i] < q1 - 1.5*iqr or y[i] > q3 + 1.5*iqr:
            y[i] = med # Replacing the outlier with the median of the attribute
            outliers += 1
    print(f"Number of outliers: {outliers}")
    print("IQR:", iqr)
    print("Variance:", np.var(interData[x]))
    plt.boxplot(interData[x])
    plt.show()

replaceWithMedian('temperature')
replaceWithMedian('rain')


# In[ ]:




