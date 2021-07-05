def mean(x):
    add = 0
    for i in x:
        add += i
    return add/len(x)
def Cov(x, y):
    add = 0
    for i in range(len(x)):
        add += (X[i] - mean(X))*(Y[i] - mean(Y))
    return add/(len(x) - 1)

X = [15, 17, 20, 21, 25]
Y = [9, 13, 16, 18, 21]
#Cov(X, Y) = Sum((X - mean(X))(Y - mean(Y)))/(n-1)

print("Covariance of X and Y:", "{0:.4f}".format(Cov(X, Y)))