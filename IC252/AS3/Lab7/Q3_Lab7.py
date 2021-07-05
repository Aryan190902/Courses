from math import sqrt
import pandas as pd
import numpy as np

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
    
data = pd.read_csv('BNG-Device.csv')
activeSeries = pd.Series(data['Active-Count'])
cpuSeries = pd.Series(data['CPU-Utilization'])
# i part
y = "{0:.4f}".format(np.corrcoef(activeSeries, cpuSeries)[0, 1])
print("Correlation between Active-Count and CPU-Utilization:", 
y, "| Relation:",Relation(y))

#ii part
avgTempSeries = pd.Series(data['Average-Temperature'])
memorySeries = pd.Series(data['Total-Memory-Usage'])
y = "{0:.4f}".format(np.corrcoef(cpuSeries, memorySeries.fillna(0))[0, 1])
print("Correlation between CPU-Utilization and Total-Memory-Usage:", 
y, "| Relation:", Relation(y))

# iii part
y = "{0:.4f}".format(np.corrcoef(cpuSeries, avgTempSeries)[0, 1])
avgTempSeries = pd.Series(data['Average-Temperature'])
print("Correlation between CPU-Utilization and Average-Temperature:", 
y, "| Relation:", Relation(y))

# iv part
y = "{0:.4f}".format(np.corrcoef(activeSeries, avgTempSeries)[0, 1])
print("Correlation between Active-Count and Average-Temperature:", 
y, "| Relation:", Relation(y))

# v part
y = "{0:.4f}".format(np.corrcoef(memorySeries.fillna(0), avgTempSeries)[0, 1])
print("Correlation between CPU-Utilization and Average-Temperature:", 
y, "| Relation:", Relation(y))