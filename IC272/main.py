#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Aryan Apte, Roll No.- B20186, Mobile No.- 8770083396, Branch- EE

import pandas as pd
import numpy as np
from scipy.stats import mode
#reading csv file using pandas
data = pd.read_csv('C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn1\\pima-indians-diabetes.csv')

#Defining variables for better readability of the code
pregs = data.pregs
plas = data.plas
pres = data.pres
skin = data.skin
test = data.test
bmi = data.BMI
pedi = data.pedi
age = data.Age

#Using numpy for finding mean, median, std. deviation
meanSeries = pd.Series([np.mean(pregs), np.mean(plas), np.mean(pres), np.mean(skin), np.mean(test),
                       np.mean(bmi), np.mean(pedi), np.mean(age)])
medianSeries = pd.Series([np.median(pregs), np.median(plas), np.median(pres), np.median(skin), np.median(test),
                       np.median(bmi), np.median(pedi), np.median(age)])

#Using Scipy for mode as numpy doesn't have a mode function
modeSeries = pd.Series([mode(pregs)[0], mode(plas)[0], mode(pres)[0], mode(skin)[0], mode(test)[0],
                       mode(bmi)[0], mode(pedi)[0], mode(age)[0]])

#Simple min() and max() functions
minSeries = pd.Series([min(pregs), min(plas), min(pres), min(skin), min(test),
                       min(bmi), min(pedi), min(age)])
maxSeries = pd.Series([max(pregs), max(plas), max(pres), max(skin), max(test),
                       max(bmi), max(pedi), max(age)])

stdSeries = pd.Series([np.std(pregs), np.std(plas), np.std(pres), np.std(skin), np.std(test),
                       np.std(bmi), np.std(pedi), np.std(age)])

#Creating a DataFrame to for better readability of the output
table = pd.DataFrame({'Serial No.': [i+1 for i in range(0, 8)],
                        'Attributes': ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi', 'Age'],
                         'Mean': meanSeries,
                         'Median': medianSeries,
                         'Mode': modeSeries,
                         'Min.': minSeries,
                         'Max.': maxSeries,
                         'S.D.': stdSeries})
print(table)


# In[2]:


import matplotlib.pyplot as plt
#Using Matplotlib for plotting the Graphs

#Scatter Plots
plt.scatter(age, pregs)
plt.xlabel('Age(in years) -->')
plt.ylabel('Pregs -->')
plt.show()


# In[3]:


plt.scatter(age, plas)
plt.xlabel('Age(in years) -->')
plt.ylabel('Plas -->')
plt.show()


# In[4]:


plt.scatter(age, pres)
plt.xlabel('Age(in years) -->')
plt.ylabel('Pres -->')
plt.show()


# In[5]:


plt.scatter(age, skin)
plt.xlabel('Age(in years) -->')
plt.ylabel('Skin -->')
plt.show()


# In[6]:


plt.scatter(age, test)
plt.xlabel('Age(in years) -->')
plt.ylabel('test(in mm U/mL) -->')
plt.show()


# In[7]:


plt.scatter(age, bmi)
plt.xlabel('Age(in years) -->')
plt.ylabel('BMI (in kg/m2) -->')
plt.show()


# In[8]:


plt.scatter(age, pedi)
plt.xlabel('Age(in years) -->')
plt.ylabel('pedi -->')
plt.show()


# In[9]:


plt.scatter(bmi, pregs)
plt.xlabel('BMI (in kg/m2) -->')
plt.ylabel('Pregs -->')
plt.show()


# In[10]:


plt.scatter(bmi, plas)
plt.xlabel('BMI (in kg/m2) -->')
plt.ylabel('Plas -->')
plt.show()


# In[11]:


plt.scatter(bmi, pres)
plt.xlabel('BMI (in kg/m2) -->')
plt.ylabel('Pres(in mm Hg) -->')
plt.show()


# In[12]:


plt.scatter(bmi, skin)
plt.xlabel('BMI (in kg/m2) -->')
plt.ylabel('Skin(in mm) -->')
plt.show()


# In[13]:


plt.scatter(bmi, test)
plt.xlabel('BMI (in kg/m2) -->')
plt.ylabel('test(in mm U/mL) -->')
plt.show()


# In[14]:


plt.scatter(bmi, pedi)
plt.xlabel('BMI (in kg/m2) -->')
plt.ylabel('Pedi -->')
plt.show()


# In[15]:


plt.scatter(bmi, age)
plt.xlabel('BMI (in kg/m2) -->')
plt.ylabel('Age(in years) -->')
plt.show()


# In[16]:


#Using np.corrcoef() function for finding correlation
#As it returns a matrix, that's why we need [0, 1] or [1, 0] index value for correlation value
attri = ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi', 'Age']
corr = pd.DataFrame({'S.No.': [i+1 for i in range(0, 8)],
                     'Attributes': ['pregs', 'plas', 'pres(in mm Hg)', 'skin (in mm)'
                                    , 'test(in mu U/mL)', 'BMI(in kg/m2)', 'pedi', 'Age(in years)'],
                    'Correlation Coefficient Value': [np.corrcoef(data[attri[i]], age)[0, 1] for i in range(len(attri))]})
print(corr)


# In[17]:


#Using np.corrcoef() function for finding correlation
#As it returns a matrix, that's why we need [0, 1] or [1, 0] index value for correlation value

attri = ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi', 'Age']
corr = pd.DataFrame({'S.No.': [i+1 for i in range(0, 8)],
                     'Attributes': ['pregs', 'plas', 'pres(in mm Hg)', 'skin (in mm)'
                                    , 'test(in mu U/mL)', 'BMI(in kg/m2)', 'pedi', 'Age(in years)'],
                    'Correlation Coefficient Value': [np.corrcoef(data[attri[i]], bmi)[0, 1] for i in range(len(attri))]})
print(corr)


# In[18]:


#Histogram Plot starts here

plt.hist(pregs)
plt.show()


# In[19]:


plt.hist(skin)
plt.show()


# In[20]:


dataClass0 = data[data['class'] == 0] #Taking data where class elements are 0
plt.hist(dataClass0.pregs) # Scrapping pregs column from that as required by the question
plt.show()


# In[21]:


dataClass1 = data[data['class'] == 1] #Taking data where class elements are 1
plt.hist(dataClass1.pregs) # Scrapping pregs column from that as required by the question
plt.show()


# In[22]:


# Box Plots from here
plt.boxplot(pregs)
plt.xlabel('Pregs')
plt.ylabel('Value')
plt.show()
q1 = np.quantile(pregs, 0.25) # First Quartile
q3 = np.quantile(pregs, 0.75) # Third Quartile
med = np.median(pregs)
iqr = q3 - q1 # Inter - Quartile Range
print("First Quartile", q1)
print("Median", med)
print("Third Quantile", q3)
print("Inter-Quartile Range", iqr)
print("Variance", np.var(pregs))
#Outliers are the values that are more than 1.5*IQR distance away from Q1 and Q3.


# In[23]:


plt.boxplot(plas)
plt.xlabel('Plas')
plt.ylabel('Value')
plt.show()
q1 = np.quantile(plas, 0.25)
q3 = np.quantile(plas, 0.75)
med = np.median(plas)
iqr = q3 - q1
print("First Quartile", q1)
print("Median", med)
print("Third Quantile", q3)
print("Inter-Quartile Range", iqr)
print("Variance", np.var(plas))


# In[24]:


plt.boxplot(pres)
plt.xlabel('Pres(in mm Hg)')
plt.ylabel('Value')
plt.show()
q1 = np.quantile(pres, 0.25)
q3 = np.quantile(pres, 0.75)
med = np.median(pres)
iqr = q3 - q1
print("First Quartile", q1)
print("Median", med)
print("Third Quantile", q3)
print("Inter-Quartile Range", iqr)
print("Variance", np.var(pres))


# In[25]:


plt.boxplot(skin)
plt.xlabel('Skin(in mm)')
plt.ylabel('Value')
plt.show()
q1 = np.quantile(skin, 0.25)
q3 = np.quantile(skin, 0.75)
med = np.median(skin)
iqr = q3 - q1
print("First Quartile", q1)
print("Median", med)
print("Third Quantile", q3)
print("Inter-Quartile Range", iqr)
print("Variance", np.var(skin))


# In[26]:


plt.boxplot(test)
plt.xlabel('Test')
plt.ylabel('Value')
plt.show()
q1 = np.quantile(test, 0.25)
q3 = np.quantile(test, 0.75)
med = np.median(test)
iqr = q3 - q1
print("First Quartile", q1)
print("Median", med)
print("Third Quantile", q3)
print("Inter-Quartile Range", iqr)
print("Variance", np.var(test))


# In[27]:


plt.boxplot(bmi)
plt.xlabel('BMI(in kg/m2)')
plt.ylabel('Value')
plt.show()
q1 = np.quantile(bmi, 0.25)
q3 = np.quantile(bmi, 0.75)
med = np.median(bmi)
iqr = q3 - q1
print("First Quartile", q1)
print("Median", med)
print("Third Quantile", q3)
print("Inter-Quartile Range", iqr)
print("Variance", np.var(bmi))


# In[28]:


plt.boxplot(pedi)
plt.xlabel('Pedi')
plt.ylabel('Value')
plt.show()
q1 = np.quantile(pedi, 0.25)
q3 = np.quantile(pedi, 0.75)
med = np.median(pedi)
iqr = q3 - q1
print("First Quartile", q1)
print("Median", med)
print("Third Quantile", q3)
print("Inter-Quartile Range", iqr)
print("Variance", np.var(pedi))


# In[29]:


plt.boxplot(age)
plt.xlabel('Age')
plt.ylabel('Value')
plt.show()
q1 = np.quantile(age, 0.25)
q3 = np.quantile(age, 0.75)
med = np.median(age)
iqr = q3 - q1
print("First Quartile", q1)
print("Median", med)
print("Third Quantile", q3)
print("Inter-Quartile Range", iqr)
print("Variance", np.var(age))


# In[ ]:




