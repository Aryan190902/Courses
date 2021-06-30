import matplotlib.pyplot as plt
import random

def method(n):
    cntH = 0
    cntT = 0
    h = [1, 2] #Biased Case
    t = [3, 4, 5, 6]
    for i in range(n):
        d = random.randint(1, 6)
        if d in h:
            cntH += 1
        elif d in t:
            cntT += 1
    plt.bar(x = 1, height=cntH, label = 'H', color = 'blue')
    plt.bar(x = 2, height=cntT, label = 'T', color = 'green')
    plt.xticks([1, 2], ['Heads', 'Tails'])
    plt.yticks([i for i in range(0, 8000, 500)])
    plt.text(x = 1, y=cntH+1, s = str(cntH))
    plt.text(x = 2, y=cntT+1, s = str(cntT))
    plt.legend()
    plt.show()

n = int(input('Number of experiments: '))
method(n)