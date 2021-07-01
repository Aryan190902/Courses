from numpy import random
import matplotlib.pyplot as plt
P = float(input("Probability of sending the message correctly= "))
cnt = 0
lst = []

for i in range(10000):
    mssg = 0
    cnt = 0
    while mssg != 1:
        mssg = random.choice([0, 1], p= [1-P, P])
        if mssg != 1:
            cnt += 1
        else:
            cnt+= 1
            lst.append(cnt)

plt.hist(lst, bins=10)
plt.show()
