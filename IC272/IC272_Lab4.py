#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn4\\SteelPlateFaults-2class.csv")
dataClass = data['Class']

[modelTrain, modelTest, model_label_train, model_label_test] = train_test_split(data, dataClass, test_size=0.3, random_state=42, shuffle=True)


# Saving the train and test data in csv.

modelTrain.to_csv("SteelPlateFaults-train.csv", index=False)
modelTest.to_csv("SteelPlateFaults-test.csv", index=False)


# In[2]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

high_acc = 1
high_value = 0

def KNN(K):
    global high_acc, high_value
    
    neighb = KNeighborsClassifier(n_neighbors=K)
    neighb.fit(modelTrain, model_label_train)
    modelPredict = neighb.predict(modelTest)
    
    # Confusion matrix
    confMatrix = confusion_matrix(model_label_test.to_numpy(), modelPredict)
    print(confMatrix)
    print("")
    
    accScore = accuracy_score(model_label_test.to_numpy(), modelPredict)
    print(f"Accuracy for K={K} is:", accScore)
    print("")
    
    if accScore > high_value:
        high_acc = K
        high_value = accScore
        
KNN(1)
KNN(3)
KNN(5)

print("Highest Accuracy in K is:", high_acc)
print("Highest Value of Accuracy is:", high_value)
print("")


# In[3]:


# Q2
# Min-Max Normalization

dataQ2 = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn4\\SteelPlateFaults-train.csv")
colQ2 = dataQ2.columns

maxDict = dataQ2.max()
minDict = dataQ2.min()

normDf = dataQ2.copy()

for i in colQ2:
    normDf[i] = normDf[i].astype(float)
    for j in range(len(normDf[i])):
        x = normDf[i][j]
        normDf[i][j] = (maxDict[i] - x)/(maxDict[i]-minDict[i])
print(normDf)
normDf.to_csv("SteelPlateFaults-train-Normalised.csv")

dataTestQ2 = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn4\\SteelPlateFaults-test.csv")
colTestQ2 = dataTestQ2.columns

maxTestDict = dataTestQ2.max()
minTestDict = dataTestQ2.min()

normTestDf = dataTestQ2.copy()

for i in colTestQ2:
    normTestDf[i] = normTestDf[i].astype(float)
    for j in range(len(normTestDf[i])):
        x = normTestDf[i][j]
        normTestDf[i][j] = (maxTestDict[i] - x)/(maxTestDict[i] - minTestDict[i])
print(normTestDf)
normTestDf.to_csv("SteelPlateFaults-test-Normalised.csv")


# In[4]:


high_acc_norm = 1
high_value_norm = 0

dataTestClass = dataTestQ2['Class']

[normTrain, normTest, norm_label_train, norm_label_test] = train_test_split(dataTestQ2, dataTestClass, test_size=0.3,
                                                                                random_state=42, shuffle=True)
def KNN_Norm(K):
    global high_acc_norm, high_value_norm
    
    neighb = KNeighborsClassifier(n_neighbors=K)
    neighb.fit(normTrain, norm_label_train)
    normPredict = neighb.predict(normTest)
    
    # Confusion matrix
    confMatrix = confusion_matrix(norm_label_test.to_numpy(), normPredict)
    print(confMatrix)
    print("")
    
    # Accuracy Score
    accScore = accuracy_score(norm_label_test.to_numpy(), normPredict)
    print(f"Accuracy for K={K} is:", accScore)
    print("")
    
    if accScore > high_value_norm:
        high_acc_norm = K
        high_value_norm = accScore
        
KNN_Norm(1)
KNN_Norm(3)
KNN_Norm(5)

print("Highest Accuracy in K is:", high_acc_norm)
print("Highest Value of Accuracy is:", high_value_norm)
print("")


# In[15]:


# Q3
# Dropping X_Minimum, Y_Minimum, TypeOfSteel_A300 and TypeOfSteel_A400 attributes from both test and train files.
import numpy as np
dataQ2 = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn4\\SteelPlateFaults-train.csv")
dataQ2.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis=1)

dataTestQ2 = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn4\\SteelPlateFaults-test.csv")
dataTestQ2.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis=1)

# Bayes-Classifier
dataQ2 = dataQ2[dataQ2.columns[1:]]
dataTestQ2 = dataTestQ2[dataTestQ2.columns[1:]]
dataTestQ2 = dataTestQ2[dataTestQ2.columns[:-1]]

data0 = dataQ2[dataQ2["Class"] == 0]
data1 = dataQ2[dataQ2["Class"] == 1]

model0 = dataQ2[dataQ2.columns[:-1]]
model1 = dataTestQ2[dataTestQ2.columns[:-1]]

cov0 = np.cov(model0.T)
cov1 = np.cov(model1.T)

mean0 = np.mean(model0)
mean1 = np.mean(model1)
print("Mean0")
print(mean0)
print("\nMean1")
print(mean1)

print("----------------")
print("Cov0")
print(cov0)
print("\nCov1")
print(cov1)
def Likelihood(val, minVal, covMat):
    Mat = np.dot((val-minVal).T, np.linalg.inv(covMat))
    interior = -np.dot(Mat, (val-minVal))/2
    exterior = np.exp(interior)
    return exterior/(((2*np.pi)**5)* (np.linalg.det(covMat)**0.5))

pr0 = len(model0)/len(dataQ2)
pr1 = len(model1)/len(dataQ2)
someLst = []
for i, row in dataTestQ2.iterrows():
    p0 = Likelihood(row, mean0, cov0)*pr0
    p1 = Likelihood(row, mean1, cov1)*pr1
    if p0 > p1:
        someLst.append(0)
    else:
        someLst.append(1)
bayesAcc = accuracy_score(model_label_test.to_numpy(), someLst)
print("The Confusion Matrix are: \n", confusion_matrix(model_label_test, someLst))
print("Accuracy Score:", bayesAcc)


# In[11]:


# Q4 
print("Accuracy Scores")
print("KNN")
print(high_value)
print()
print("KNN Normalised")
print(high_value_norm)
print()
print("Bayes Classifier")
print(bayesAcc)
print()
print('Highest accuracy is achieved in KNN Normalised: ',high_value_norm)
print()


# In[ ]:




