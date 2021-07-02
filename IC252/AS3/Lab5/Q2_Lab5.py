from numpy import random
import matplotlib.pyplot as plt
n = 10000
P = float(input("Probability of P(Y = 0 | X = 0): "))
Q = float(input("Probability of P(Y = 0 | X = 1): "))
y1x0 = 0
y1x1 = 0
y0x1 = 0
y0x0 = 0
outX0 = 0
outX1 = 0
for i in range(n):
    x = random.choice([0, 1])
    if x == 0:
        outX0 += 1
        output = random.choice([0, 1], p=[P, 1-P])
        if output == 1:
            y1x0 += 1
        else:
            y0x0 += 1
# As, P(Y = 0|X = 0)*P(X = 0) = P(Y = 0, X = 0)
    if x == 1:
        outX1 += 1
        output = random.choice([0, 1], p=[1-Q, Q])
        if output == 1:
            y1x1 += 1
        else:
            y0x1 += 1
print("X0=", outX0)
print("X1=", outX1)
print("P(Y = 0|X = 0)=", "{0:.4f}".format(y0x0/outX0))
print("P(Y = 0|X = 1)=", "{0:.4f}".format(y0x1/outX1))
print("\n---x---x---x---")

print("\nP(Y = 1|X = 0)=", "{0:.4f}".format(y1x0/outX0))
print("P(Y = 1|X = 1)=", "{0:.4f}".format(y1x1/outX1))
valueLst = ["{0:.4f}".format(y0x0/outX0),"{0:.4f}".format(y0x1/outX1),
"{0:.4f}".format(y1x0/outX0),"{0:.4f}".format(y1x1/outX1)]
for i in range(1, 5):
    plt.bar(i, float(valueLst[i-1]))
for i in range(1, 5):
    plt.text(i-0.2, float(valueLst[i-1]) + 0.01, s=valueLst[i-1])

plt.xticks([i for i in range(1, 5)], ["P(Y = 0|X = 0)", "P(Y = 0|X = 1)", 
"P(Y = 1|X = 0)", "P(Y = 1|X = 1)"])
plt.title("Joint Distribution for (X, Y)")
plt.axis([0, 5, 0, 1])
plt.show()
