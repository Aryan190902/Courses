import pandas as pd
import numpy as np

data = pd.read_csv('IC252_Lab8.csv', sep=',')
data.Status[data.Status == "Hospitalized"] = 1
data.Status[data.Status == "Recovered"] = 2
data.Status[data.Status == "Dead"] = 3

stats = data.Status.tolist()
population = data.Population.tolist()
ratio = data.SexRatio.tolist()
litracy = data.Literacy.tolist()
age = data.Age.tolist()
smell = data.SmellTrend.tolist()
gender = data.Gender.tolist()

print(data.Status)
print("Correlation between:")
print("Status and Population:", "{0:.4f}".format(np.corrcoef(stats, population)[0, 1]))
print("Status and SexRatio:", "{0:.4f}".format(np.corrcoef(stats, ratio)[0 ,1]))
print("Status and Literacy:", "{0:.4f}".format(np.corrcoef(stats, litracy)[0 ,1]))
print("Status and Age:", "{0:.4f}".format(np.corrcoef(stats, age)[0 ,1]))
print("Status and SmellTrend:", "{0:.4f}".format(np.corrcoef(stats, smell)[0 ,1]))
print("Status and Gender:", "{0:.4f}".format(np.corrcoef(stats, gender)[0 ,1]))

answer = {
'Status and Population': "{0:.4f}".format(np.corrcoef(stats, population)[0, 1]),
'Status and SexRatio': "{0:.4f}".format(np.corrcoef(stats, ratio)[0 ,1]),
'Status and Literacy': "{0:.4f}".format(np.corrcoef(stats, litracy)[0 ,1]),
'Status and Age': "{0:.4f}".format(np.corrcoef(stats, age)[0 ,1]),
'Status and SmellTrend': "{0:.4f}".format(np.corrcoef(stats, smell)[0 ,1]),
'Status and Gender': "{0:.4f}".format(np.corrcoef(stats, gender)[0 ,1])
}
print(sorted(answer.items(), key= lambda k : (k[1], k[0]), reverse=True))
