from numpy import random
import math
import matplotlib.pyplot as plt
N = int(input("Number of throws: "))
P = float(input("Probability of getting a head: "))
num_of_times = 10000

head = 0
headLst = []
for i in range(num_of_times):
    head = 0
    for j in range(N):
        coin = random.choice([0,1], p=[1-P, P])
        if coin == 1:
            head += 1
    headLst.append(head)
plt.hist(headLst, bins=10)
plt.axis([0, N, 0, 3000])
plt.show()

