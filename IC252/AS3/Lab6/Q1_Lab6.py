import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = pd.read_excel('Covid19IndiaData_30032020.xlsx', sheet_name="Covid19IndiaData_30032019")
ageCnt = Counter(data['Age'])
ageSize = len(data.Age)

# 1 a
ageVar = np.var(data['Age']) #Variance
exp = 0 #Expectation
for i in ageCnt:
    exp += i*ageCnt[i]/ageSize
print("Expectation:", "{0:.2f}".format(exp))
print("Variance:", "{0:.2f}".format(ageVar))
print('')
plt.hist(data['Age'], edgecolor='black', bins=10, density=True)
plt.xticks([i for i in range(0, 101, 10)])
plt.xlabel('Age -->')
plt.ylabel('PMF -->')
plt.title("Probability Mass Function")
plt.show()

# 1 b
age = pd.Series(data.Age)
stats = pd.Series(data.StatusCode)
status = pd.DataFrame({'Age': age, 'StatusCode':stats})
need = status[status['StatusCode'] == 'Recovered']['Age']
need.append(status[status['StatusCode'] == 'Dead']['Age'])
lst = list(need)
lstCnt = Counter(lst)
lstSize = len(lst)
exp = 0 #Expectation
lstVar = np.var(lst)
for i in lstCnt:
    exp += i*lstCnt[i]/lstSize
print("------- Part 2 ---------")
print("Expectation:", "{0:.2f}".format(exp))
print("Variance:", "{0:.2f}".format(lstVar))

plt.hist(lst, bins=10, density=True, edgecolor='black')
plt.xlabel('Age -->')
plt.ylabel('PMF -->')
plt.title("PMF for Recovered and Dead Patients")
plt.show()

# 1 c
gender = pd.Series(data.GenderCode0F1M)
someData = pd.DataFrame({'GenderCode': gender, 'Age': age})
female = list(someData[someData['GenderCode'] == 0]['Age'])
male = list(someData[someData['GenderCode'] == 1]['Age'])

# male
fig, ax = plt.subplots(1, 2, figsize=(8, 8))
ax[0].hist(male, bins=10, edgecolor='black', density=True)
ax[0].set_title('Male')

# female
ax[1].hist(female, bins=10, edgecolor='black', density=True)
ax[1].set_title('Female')

plt.show()
