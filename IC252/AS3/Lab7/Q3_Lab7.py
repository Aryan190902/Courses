from math import sqrt
import pandas as pd

def Relation(z):
    x = float(z)
    if x == 0:
        return "None"
    elif (x>0.0 and x<=0.1) or (x>=-0.1 and x<0.0):
        return "Weak"
    elif (x>0.1 and x<=0.3) or (x>=-0.3 and x<0.1):
        return "Moderate"
    elif (x>0.3 and x<=0.5) or (x>=-0.5 and x<0.3):
        return "Strong"
    elif (x>0.5 and x<1.0) or (x>-1 and x<0.5):
        return "Very Strong"
    elif x == 1:
        return "Perfect"

def mean(x):
    add = 0
    for i in x:
        add += i
    return add/len(x)

def Cov(x, y):
    add = 0
    for i in range(len(x)):
        add += (x[i] - mean(x))*(y[i] - mean(y))
    return add/(len(x) - 1)

def Corr(x, y):
    return Cov(x, y)/sqrt(Cov(x, x)*Cov(y, y))

data = pd.read_csv('BNG-Device.csv')
activeSeries = pd.Series(data['Active-Count'])
cpuSeries = pd.Series(data['CPU-Utilization'])
avgTempSeries = pd.Series(data['Average-Temperature'])
memorySeries = pd.Series(data['Total-Memory-Usage'])

# i part
y = "{0:.4f}".format(Corr(activeSeries, cpuSeries))
print("Correlation between Active-Count and CPU-Utilization:", 
y, "| Relation:",Relation(y))

#ii part
y = "{0:.4f}".format(Corr(cpuSeries, memorySeries.fillna(0)))
print("Correlation between CPU-Utilization and Total-Memory-Usage:", 
y, "| Relation:", Relation(y))

# iii part
y = "{0:.4f}".format(Corr(cpuSeries, avgTempSeries))
print("Correlation between CPU-Utilization and Average-Temperature:", 
y, "| Relation:", Relation(y))

# iv part
y = "{0:.4f}".format(Corr(activeSeries, avgTempSeries))
print("Correlation between Active-Count and Average-Temperature:", 
y, "| Relation:", Relation(y))

# v part
y = "{0:.4f}".format(Corr(memorySeries.fillna(0), avgTempSeries))
print("Correlation between CPU-Utilization and Average-Temperature:", 
y, "| Relation:", Relation(y))