#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn3\\pima-indians-diabetes.csv")
col = data.columns
q1Data = data.loc[:, col!= 'class'] # Data excluding class attribute.
for i in q1Data.columns:
    firstQuartile = q1Data[i].quantile(0.25)
    thirdQuartile = q1Data[i].quantile(0.75)
    iqr = thirdQuartile - firstQuartile
    med = np.median(q1Data[i])
    for j in range(len(q1Data[i])):
        if q1Data[i][j] < firstQuartile - 1.5*iqr or q1Data[i][j] > thirdQuartile + 1.5*iqr:
            q1Data[i][j] = med
    # Refined the outliers with the median of the respected attributes.
mMNorm = q1Data.copy() # Making another DataFrame.
# Min-Max Normalization
for i in q1Data.columns:
    M = max(mMNorm[i])
    m = min(mMNorm[i])
    for j in range(len(mMNorm[i])):
        mMNorm[i][j] = ((mMNorm[i][j] - m)/(M - m))*(12-5) + 5 # Specified Range = 5 to 12.
print("Min-Max Normalization")
stdNorm = q1Data.copy() # Another DataFrame for another normalization.
# Std Normalization
for i in q1Data.columns:
    meanValue = np.mean(q1Data[i])
    stdValue = np.std(q1Data[i])
    for j in range(len(stdNorm[i])):
        stdNorm[i][j] =(stdNorm[i][j] - meanValue)/stdValue

someDf = {'Attributes': col,
             'Before Normalization': {
                 'Maximum': [max(q1Data[i]) for i in q1Data.columns],
                 'Minimum': [min(q1Data[i]) for i in q1Data.columns]
             },
             'After Normalization': {
                 'Maximum': [max(mMNorm[i]) for i in q1Data.columns],
                 'Minimum': [min(mMNorm[i]) for i in q1Data.columns]
             }
             }
print(someDf)
anotherDf = {'Attributes': col,
             'Before Normalization': {
                 'Maximum': [max(q1Data[i]) for i in q1Data.columns],
                 'Minimum': [min(q1Data[i]) for i in q1Data.columns]
             },
             'After Normalization': {
                 'Maximum': [max(stdNorm[i]) for i in q1Data.columns],
                 'Minimum': [min(stdNorm[i]) for i in q1Data.columns]
             }
             }
print(anotherDf)


# In[2]:


# 2
from numpy.random import multivariate_normal
from numpy.linalg import eig, norm
# a
origin = [0, 0]
meanGiven = np.array([0, 0])
cov = np.array([[13, -3], [-3, 5]])
dist = multivariate_normal(meanGiven, cov, 1000).T

plt.scatter(dist[0],dist[1])
plt.xlabel("X -->")
plt.ylabel("Y -->")
plt.title("Scatter plot for original data")
plt.show()

# b 

eigenvalues,eigenvectors = eig(cov)
print("Eigen vectors are: ", eigenvectors)
eig_vec_1= eigenvectors[:,1]
eig_vec_2= eigenvectors[:,0]
plt.scatter(dist[0],dist[1])
plt.title("Scatter plot with eigen directions")
plt.xlabel("x1")
plt.ylabel("x2")
plt.quiver([0,0],[0,0],eig_vec_1,eig_vec_2,scale=5)
plt.show()

# c

A = np.dot(dist.T, eig_vec_1)
plt.scatter(dist[0] , dist[1])
plt.quiver(origin,origin,eig_vec_1,eig_vec_2,scale=5)
plt.scatter(A*eigenvectors[0][1],A*eigenvectors[0][0])
plt.title('Projected values on 1st Eigen vector')
plt.xlabel("X -->")
plt.ylabel("Y -->")
plt.show()


plt.scatter(dist[0] , dist[1])
plt.quiver(origin,origin,eig_vec_1,eig_vec_2,scale=5)
plt.scatter(A*eigenvectors[1][1],A*eigenvectors[1][0])
plt.title('Projected values on 2nd Eigen vector')
plt.xlabel("X -->")
plt.ylabel("Y -->")
plt.show()

#part(d)

some1= np.dot(dist.T, eigenvectors)
some2= np.dot(some1, eigenvectors.T)
error= np.square(np.subtract(dist.T,some2)).mean()
print("Error: ",round(error,3))



# In[3]:


# 3

from sklearn.decomposition import PCA

data=pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn3\\pima-indians-diabetes.csv")
data=data.iloc[:,[i for i in range(8)]]
columnName=list(data.columns)

for i in columnName:
    lst=list(data[i]) 
    lst=sorted(lst)
    n=len(lst)
    q1=data[i].quantile(0.25)
    q2=data[i].quantile(0.50)
    q3=data[i].quantile(0.75)
    iqr=q3-q1
    for j in range(len(data[i])):
        if data[i][j]<(q1-1.5*iqr) or data[i][j]>(q3+1.5*iqr):
            data[i][j]=data[i].median()
someData=data.copy()
for i in someData.columns:
    meanX=sum(data[i])/len(data[i])
    stdX=data[i].std()
    for j in range(len(data[i])):
        x=data[i][j]
        someData.loc[j,i]=(x-meanX)/stdX

# df_q3_a--outlier corrected standardized data
# finding eigonvectors and eigonvalues
corr_matrix = someData.corr()
evalues, evectors = eig(corr_matrix)
print(evectors)
#Finding 2 eigonvectors with highest eigonvalues
max_indices = []
for i in range(2):
    maxI = evalues[0]
    maxi = 0
    for j in range(len(evalues)):
        if evalues[j] > maxI:
            maxI = evalues[j]
            maxi = j
    print("Max eigned values is ", maxI,'and it is found at',maxi+1)
    max_indices.append(maxi)
    evalues[maxi]=0


eigenvector_1 = evectors[:, max_indices[0]]  #eigen vector with highest eigonvalue
eigenvector_2 = evectors[:, max_indices[1]]  #eigen vector with second highest eigonvalue

# Now reducing data into 2 dimension
pca_1 = PCA(n_components=2)
pca_1.fit(someData)
reduced_data_1 = pca_1.fit_transform(someData)
reduced_data_df = pd.DataFrame(reduced_data_1, columns=['A', 'B'])
print(f"Variance: {reduced_data_df.var()}")
print(reduced_data_df.var())
print(f"Eigen vectors: {pca_1.components_}")

print(f"Eigen vectors from numpy and cov matrix: {eigenvector_1, eigenvector_2}")

# Now ploting 2 dimensional data on scatter plot
plt.scatter(reduced_data_df['A'], reduced_data_df['B'], marker ='2')
plt.title('Data projected along 2 dimensions')
plt.xlabel('X -->')
plt.ylabel('Y -->')
plt.show()

evalues, evectors = eig(corr_matrix)
evalues = list(evalues)

# Sort
evalues.sort(reverse=True)
x = [i+1 for i in range(len(evalues))]
plt.bar(x, evalues)
plt.title('Descending Order')
plt.xlabel('Position -->')
plt.ylabel('Eigen Value -->')
plt.show()


error_record = []
for i in range(1, 9):
    pca = PCA(n_components=i)
    pca2 = pca.fit_transform(someData)
    pca2_proj_back = pca.inverse_transform(pca2)
    error_record.append(norm((someData-pca2_proj_back), None))#Numpy norm method to calculate the error

x = [i for i in range(1,len(error_record)+1)]
plt.plot(x, error_record)
plt.title('Graph with euclidian distance for different l upto 8')
plt.xlabel('L Dimensions -->')
plt.ylabel('Euclidian distance -->')
plt.show()
#part d

print(corr_matrix)
pca_8 = PCA(n_components=8)
pca_8_result = pca_8.fit_transform(someData)
pca_8_proj_back = pca_8.inverse_transform(pca_8_result)
print()
for i in range(1, 9):
    pca = PCA(n_components=i)
    pca2 = pca.fit_transform(someData)
    someDF=pd.DataFrame(pca2)
    print(f'----Covariance matrix for {i} dimension')
    print(someDF.cov())

    
   


# In[ ]:




