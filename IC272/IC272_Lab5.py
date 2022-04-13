#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
# Part A
train = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn4\\SteelPlateFaults-train.csv")
test = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn4\\SteelPlateFaults-test.csv")

testClass = test['Class'].values

train = train.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
test = test.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)

# separation of data
train0 = train.groupby('Class').get_group(0).to_numpy()
train0 = np.delete(train0, 23, axis=1)

train1 = train.groupby('Class').get_group(1).to_numpy()
train1 = np.delete(train1, 23, axis=1)

# dropping 'Class' attribute
train = train.drop(['Class'], axis=1)
test = test.drop(['Class'], axis=1)

# n_components
x = [2, 4, 8, 16]
for i in x:
    predict = []
    # GMM
    gmm0 = GaussianMixture(n_components=i, covariance_type='full', reg_covar=1e-5).fit(train0)
    gmm1 = GaussianMixture(n_components=i, covariance_type='full', reg_covar=1e-5).fit(train1)
    
    # Weighted log probabilities
    lg0 = gmm0.score_samples(test) + np.log(len(train0)/len(train))
    lg1 = gmm1.score_samples(test) + np.log(len(train1)/len(train))
    for y in range(len(lg0)):
        if lg0[y] > lg1[y]:
            predict.append(0)
        else:
            predict.append(1)
    print("The confusion matrix for K=", i, "is\n", confusion_matrix(testClass, predict))
    print("The classification accuracy for Q=", i, "is", round(100 * accuracy_score(testClass, predict), 3))


# In[10]:


# Part B

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

# reading the csv file
ab = pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn5\\abalone.csv")
# splitting data into train and test data
train, test = train_test_split(ab, test_size=0.30, random_state=42, shuffle=True)

# Saving as CSV files
train.to_csv('C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn5\\abalone-train.csv', index=False)
test.to_csv('C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn5\\abalone-test.csv', index=False)

print("Q1\n")
#%% calculating the Pearson correlation coefficient of every attribute with the target attribute rings
input_var = ab[ab.columns[1:]].corr()['Rings'][:-1].idxmax()
print("Attribute with highest Pearson correlation coefficient with target attribute ring is",
      input_var)

lin_reg = LinearRegression().fit(np.array(train["Shell weight"]).reshape(-1, 1), train['Rings'])

# a
# 2923 is the length of training data

x = np.linspace(0, 1, 2923).reshape(-1, 1)
# Best fit line
ly = lin_reg.predict(x)

plt.scatter(train['Shell weight'], train['Rings'])
plt.plot(x, ly, linewidth=3, color='r')
plt.xlabel('Shell weight -->')
plt.ylabel('Rings -->')
plt.title('Best Fit Line')
plt.show()

# b
print('b:')
trainPredict = lin_reg.predict(np.array(train["Shell weight"]).reshape(-1, 1))
rmseTrain = (mse(train['Rings'], trainPredict)) ** 0.5
print("The RMSE for training data is", round(rmseTrain, 3))

# c
print('c:')
testPredict = lin_reg.predict(np.array(test["Shell weight"]).reshape(-1, 1))
rmseTest = (mse(test['Rings'].to_numpy(), testPredict)) ** 0.5
print("The RMSE for testing data is", round(rmseTest, 3))

# d
plt.scatter(test['Rings'].to_numpy(), testPredict,color = 'r')
plt.xlabel('Actual Rings -->')
plt.ylabel('Predicted Rings -->')
plt.title('Multivariate linear regression model')
plt.show()

print("Q2:")
X_train = train.iloc[:, :-1].values
Y_train = train.iloc[:, train.shape[1] - 1].values
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, test.shape[1] - 1].values

# a
print('a:')
regTrain = LinearRegression().fit(X_train, Y_train)
rmseTrain = (mse(Y_train, regTrain.predict(X_train))) ** 0.5
print("The RMSE for training data is", round(rmseTrain, 3))

# b
print('b:')
regTest = LinearRegression().fit(X_test, Y_test)
rmseTest = (mse(Y_test, regTest.predict(X_test))) ** 0.5
print("The RMSE for testing data is", round(rmseTest, 3))

# c
plt.scatter(Y_test, regTest.predict(X_test),color = 'r')
plt.xlabel('Actual Rings -->')
plt.ylabel('Predicted Rings -->')
plt.title('Multivariate linear regression model')
plt.show()

print("Q3:\n")
P = [2, 3, 4, 5]
# a
print('a:')
X = np.array(train['Shell weight']).reshape(-1, 1)
RMSE = []
for p in P:
    polyFeatures = PolynomialFeatures(p)  # p is the degree
    x_poly = polyFeatures.fit_transform(X)
    reg = LinearRegression()
    reg.fit(x_poly, Y_train)
    Y_pred = reg.predict(x_poly)
    rmse = (mse(Y_train, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The RMSE for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial) -->')
plt.ylabel('RMSE(training data) -->')
plt.title("Univariate non-linear regression model")
plt.show()

# b
print('b:')
RMSE = []
X = np.array(test['Shell weight']).reshape(-1, 1)
Y_pred = []
for i in P:
    polyFeatures = PolynomialFeatures(i)  # p is the degree
    x_poly = polyFeatures.fit_transform(X)
    reg = LinearRegression()
    reg.fit(x_poly, Y_test)
    Y_pred = reg.predict(x_poly)
    rmse = (mse(Y_test, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The RMSE for p=", i, 'is', round(rmse, 3))

# Bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE,color = 'r')
plt.xlabel('p (degree of polynomial) -->')
plt.ylabel('RMSE(test data) -->')
plt.title("Univariate non-linear regression model")
plt.show()

# c
# because p=5 has the lowest rmse
x_poly = PolynomialFeatures(5).fit_transform(x)
reg = LinearRegression()
reg.fit(x_poly, Y_train)
cy = reg.predict(x_poly)
plt.scatter(train['Shell weight'], train['Rings'])
plt.plot(np.linspace(0, 1, 2923), cy, linewidth=3, color='r')
plt.xlabel('Shell weight -->')
plt.ylabel('Rings -->')
plt.title('Best Fit Curve')
plt.show()

# d
# Best degree of polynomial is 5 as p=5 has minimum rmse
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual Rings -->')
plt.ylabel('Predicted Rings -->')
plt.title('Univariate non-linear regression model')
plt.show()

print("Q4:")
# a
print('a:')
RMSE = []
for p in P:
    polyFeatures = PolynomialFeatures(p)  # p is the degree
    x_poly = polyFeatures.fit_transform(X_train)
    reg = LinearRegression()
    reg.fit(x_poly, Y_train)
    Y_pred = reg.predict(x_poly)
    rmse = (mse(Y_train, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The RMSE for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial) -->')
plt.ylabel('RMSE(training data) -->')
plt.title("Univariate non-linear regression model")
plt.show()

# b
print('b:')
RMSE = []
Y_pred = []
for p in P:
    polyFeatures = PolynomialFeatures(p)  # p is the degree
    x_poly = polyFeatures.fit_transform(X_test)
    reg = LinearRegression()
    reg.fit(x_poly, Y_test)
    Y_pred = reg.predict(x_poly)
    rmse = (mse(Y_test, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The RMSE for p=", p, 'is', round(rmse, 3))
    # d
    # because the best degree of polynomial is 3 as p=3 has minimum rmse
    if p == 3:
        plt.scatter(Y_test, Y_pred,color = 'r')
        plt.xlabel('Actual Rings -->')
        plt.ylabel('Predicted Rings -->')
        plt.title('Multiivariate non-linear regression model')
        plt.show()

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial) -->')
plt.ylabel('RMSE(test data) -->')
plt.title("Multivariate non-linear regression model")
plt.show()


# In[ ]:




