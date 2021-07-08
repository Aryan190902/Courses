import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from scipy import stats

def add(x):
    ad = 0
    for i in x:
        ad += i
    return ad/len(x)

def float_range(start, stop, step):
    lst = []
    x = start
    while x < stop:
        lst.append(float("{0:.3f}".format(x)))
        x += step
    return lst

#input size = m
m = [10, 100, 500, 1000, 5000, 10000, 50000]
expoMean = []
unifMean = []
bernMean = []

for i in m:
    expoSize = np.random.exponential(1, i)
    unifSize = np.random.uniform(1, 2, i)
    bernSize = stats.bernoulli.rvs(p=0.2, size=i)
    expoMean.append(st.mean(expoSize))
    unifMean.append(st.mean(unifSize))
    bernMean.append(add(bernSize))

#Graph
plt.plot([i + 1 for i in range(len(expoMean))], expoMean, color= 'red', label='Exponential')
plt.plot([i + 1 for i in range(len(unifMean))], unifMean, color='blue', label='Uniform')
plt.plot([i + 1 for i in range(len(bernMean))], bernMean, color='green', label='Bernoulli')
plt.xticks([i + 1 for i in range(len(expoMean))], m)
plt.yticks(float_range(0, 2, 0.20))
plt.xlabel('Sample Size -->')
plt.ylabel('Sample Mean -->')
plt.grid()
plt.legend(loc='upper right')
plt.show()
