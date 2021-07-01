import random
n = 10000
cntX1 = 0
cntX2 = 0
cntBoth = 0
cntZ = 0
cntX_Z = 0
for i in range(n):
    num1 = random.randint(0, 1)
    num2 = random.randint(0, 1)
    numZ = num1+num2
    if numZ == 1: #Let Favourable case be z = 1
        cntZ += 1
    if num1 == 1:
        cntX1 += 1
        if num2 == 0:
            cntX_Z += 1
    if num2 == 1:
        cntX2 += 1
    if num1 == 1 and num2 == 1:
        cntBoth += 1
print("\n") 
print("-----x--------x--------x-----")
print("P(X1):", cntX1/n)
print("P(X2):", cntX2/n)
print("\nP(X1 and X2):", cntBoth/n)
print("Product of P(X1) and P(X2):", "{0:.4f}".format((cntX1/n)*(cntX2/n)))
#So, they are independent functions

#part 2
print("\nZ:", cntZ)
print("P(Z and X1):", cntX_Z/n)
print("Product:", "{0:.4f}".format((cntZ/n)*(cntX1/n)))
#So, they are independent functions
#part 3
cntX1 = 0
cntX2 = 0
cntBoth = 0
cntZ = 0
cntX_Z = 0
condX1 = 0
condX2 = 0
for i in range(n):
    num1 = random.randint(0, 1)
    num2 = random.randint(0, 1)
    z = num1 + num2
    if z == 1:
        if num1 == 1:
            condX1 += 1
        if num2 == 1:
            condX2 += 1
#So , P(Z = 1 | X1 = 1) = condX1/n
#and P(Z = 1 | X2 = 1) = condX2/n
print("\nP(Z = 1 | X1 = 1)=", condX1/n)
print("P(Z = 1 | X2 = 1)=", condX2/n)
print("Product=", "{0:.4f}".format((condX1/n)*(condX2/n)))
#and as when X1 and X2 are 1, Z cannot be 1, so P(X1 = 1, X2 = 1 | Z = 1) = 0
#So they are dependent functions 
print("P(X1 = 1, X2 = 1 | Z = 1)=", 0)
print("-----x--------x--------x-----")