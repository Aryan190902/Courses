from numpy import random
import matplotlib.pyplot as plt
n = 10000
P = float(input("Probability of P(Y = 0 | X = 0): "))
Q = float(input("Probability of P(Y = 0 | X = 1): "))
outLst = []
y1x0 = 0
y1x1 = 0
y0x1 = 0
y0x0 = 0
for i in range(n):
    x = random.choice([0, 1])
    if x == 0:
        output = random.choice([0, 1], p=[P, 1-P])
        outLst.append(output)
        if output == 1:
            y1x0 += 1
        else:
            y0x0 += 1
# As, P(Y = 0|X = 0)*P(X = 0) = P(Y = 0, X = 0)
    if x == 1:
        output = random.choice([0, 1], p=[1-Q, Q])
        outLst.append(output)
        if output == 1:
            y1x1 += 1
        else:
            y0x1 += 1
#1 a 
cnt1 = 0
cnt0 = 0
for i in outLst:
    if i == 1:
        cnt1 += 1
    else:
        cnt0 += 1
plt.hist(outLst, bins=4)
plt.xticks([0.125, 0.875], [0, 1])
plt.yticks([i for i in range(0, 10001,1000)])
plt.text(0.095, cnt0+10, str(cnt0))
plt.text(0.845, cnt1+10, str(cnt1))
plt.xlabel("Output -->")
plt.ylabel("Occurance of each output -->")
plt.show()

#1 b
print("P(Y = 0)=", cnt0/n)
print("P(Y = 0|X = 0)*P(X = 0)=", y0x0/n)
print("P(Y = 0|X = 1)*P(X = 1)=", y0x1/n)
print("Sum=", "{0:.4f}".format((y0x0/n) + (y0x1/n)))
print("---x---x---x---")
print("P(Y = 1)=", cnt1/n)
print("P(Y = 1|X = 0)*P(X = 0)=", y1x0/n)
print("P(Y = 1|X = 1)*P(X = 1)=", y1x1/n)
print("Sum=", "{0:.4f}".format((y1x0/n)+(y1x1/n)))
print("---x---x---x---\n")
print("Hence Proved")