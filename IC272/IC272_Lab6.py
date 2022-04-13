#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
data = pd.read_csv('C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn6\\daily_covid_cases.csv')
dates = [dt.strptime(date, '%Y-%m-%d').date() for date in data.Date]
plt.figure(figsize=(15, 10))
# Q1
# a
plt.plot(dates, data['new_cases'], color='blue')
plt.xlabel("Dates -->")
plt.ylabel("Covid Cases per day-->")
# b
dayShift = data.new_cases.shift(1)
plt.title("Q1 part a")
plt.show()
print("Q1, part b autocorrelation coefficient:", data.new_cases.corr(dayShift))
print()


# In[9]:


# c
plt.scatter(data.new_cases, dayShift, s=5)
plt.xlabel("Given Time Series -->")
plt.ylabel("One Day Lag -->")
plt.title("Q1 part c")
plt.show()


# In[14]:


# d
lagg = [1, 2, 3, 4, 5, 6]
coeff = []
for i in lagg:
    pearsonCoeff = data.new_cases.corr(data.new_cases.shift(i))
    print(f"Pearson Coefficient Value for {i}-day(s) lag:", pearsonCoeff)
    coeff.append(pearsonCoeff)
plt.plot(lagg, coeff, marker='o')
plt.xlabel('Lag(in day(s)) -->')
plt.ylabel("Pearson Coefficient Value -->")
plt.title("Q1 part d")
plt.show()


# In[17]:


# e
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(x=data.new_cases, lags=50)
plt.xlabel("Lag value -->")
plt.ylabel("Correlation coffecient value -->")
plt.title("Q1 part e")
plt.show()


# In[25]:


# Q2
from statsmodels.tsa.ar_model import AutoReg as AR
import math
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
# Train test split
series = pd.read_csv('C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn6\\daily_covid_cases.csv',
 parse_dates=['Date'],
index_col=['Date'],
sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
window = 5 # The lag=1
model = AR(train, lags=window)
model_fit = model.fit() # fit/train the model
coeff = model_fit.params # Get the coefficients of AR model
print()
print("Q2 part a coefficients are :",coeff)
print()

#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coeff[0] # Initialize to w0
    for d in range(window):
        yhat += coeff[d+1] * lag[window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.
# b, 1
plt.scatter(test,predictions )
plt.xlabel('Actual cases -->')
plt.ylabel('Predicted cases -->')
plt.title('Q2 part b\n Part 1')
plt.show()

#b, 2
x=[i for i in range(len(test))]
plt.plot(x,test, label='Actual cases')
plt.plot(x,predictions , label='Predicted cases')
plt.legend()
plt.title('Q2 part b\n Part 2')
plt.show()

#b, 3
rmse=mean_squared_error(test, predictions,squared=False)
print("Q2 part b-1 persent RMSE :",rmse*100/(sum(test)/len(test)),"%")
print()

mape=mean_absolute_percentage_error(test, predictions)
print("Q2 part b-1 persent MAPE :",mape)


# In[28]:


# Q3
def ARmodel(train_data, test_data, lag):
    window=lag
    model = AR(train_data, lags=window)
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model

    #using these coefficients walk forward over time steps in test, one step each time
    history = train_data[len(train_data)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test_data)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1] # Add other values
        obs = test_data[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.'''
    rmse_=mean_squared_error(test_data, predictions,squared=False)*100/(sum(test_data)/len(test_data))
    mape_=mean_absolute_percentage_error(test_data, predictions)
    return rmse_, mape_

lag=[1,5,10,15,25]
rmse_list=[]
mape_list=[]
for i in lag:
    rmse, mape=ARmodel(train, test,i)
    rmse_list.append(rmse[0])
    mape_list.append(mape)

plt.bar(lag, rmse_list)
plt.ylabel('RMSE error -->')
plt.xlabel('Lag values -->')
plt.title("Q3\n Bar chart between RMSE and Lag values")
plt.xticks(lag)
plt.show()

plt.bar(lag, mape_list)
plt.ylabel('MAPE error -->')
plt.xlabel('Lag values -->')
plt.title("Q3\n Bar chart between MAPE and Lag values")
plt.xticks(lag)
plt.show()

# Q4
df_q3=pd.read_csv("daily_covid_cases.csv")
train_q4=df_q3.iloc[:int(len(df_q3)*0.65)]
train_q4=train_q4['new_cases']
i=0
corr = 1
# abs(AutoCorrelation) > 2/sqrt(T)
while corr > 2/(len(train_q4))**0.5:
    i += 1
    t_new = train_q4.shift(i)
    corr = train_q4.corr(t_new)

rmse_q4, mape_q4=ARmodel(train, test, i)
print("Q4 Lag(heuristic) value is :", i)
print(f"Q4 RMSE value for lag value = {i} is :",rmse_q4[0])
print(f"Q4 MAPE value for lag value = {i} is :",mape_q4)


# In[ ]:




