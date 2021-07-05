#Corr(X, Y) = Cov(X, Y)/sqrt(Var(X)*Var(Y))
from math import sqrt
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

X = [15, 17, 20, 21, 25]
Y = [9, 13, 16, 18, 21]

print("Correlation of X and Y:", "{0:.4f}".format(Corr(X, Y)))