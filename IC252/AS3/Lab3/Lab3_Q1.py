import matplotlib.pyplot as plt
import random

def method1(n):
    cntH = 0
    cntT = 0
    h = [1, 2, 3]
    t = [4, 5, 6]
    for i in range(n):
        d = random.randint(1, 6)
        if d in h:
            cntH += 1
        elif d in t:
            cntT += 1
    plt.bar(x = 1, height=cntH, label = 'H', color = 'blue')
    plt.bar(x = 2, height=cntT, label = 'T', color = 'green')
    plt.xticks([1, 2], ['Heads', 'Tails'])
    plt.yticks([i for i in range(0, 7000, 500)])
    plt.text(x = 1, y=cntH+1, s = str(cntH))
    plt.text(x = 2, y=cntT+1, s = str(cntT))
    plt.legend()
    plt.show()

def method2(n):
    cntH = 0
    cntT = 0
    for i in range(n):
        d = random.randint(1,6)
        if d == 1:
            cntH += 1
        elif d == 2:
            cntT += 1
    plt.bar(x = 1, height=cntH, label = 'H', color = 'blue')
    plt.bar(x = 2, height=cntT, label = 'T', color = 'green')
    plt.xticks([1, 2], ['Heads', 'Tails'])
    plt.yticks([i for i in range(0, 5000, 500)])
    plt.text(x = 1, y=cntH+1, s = str(cntH))
    plt.text(x = 2, y=cntT+1, s = str(cntT))
    plt.legend()
    plt.show()

def method3(n): # Self-made
    cntH = 0
    cntT = 0
    for i in range(n):
        d = random.randint(1,6)
        if d%2 == 1:
            cntH += 1 # So, if number is odd, we return H
        elif d%2 == 0:
            cntT += 1 # So, if number is even, we return T
    plt.bar(x = 1, height=cntH, label = 'H', color = 'blue')
    plt.bar(x = 2, height=cntT, label = 'T', color = 'green')
    plt.xticks([1, 2], ['Heads', 'Tails'])
    plt.yticks([i for i in range(0, 7000, 500)])
    plt.text(x = 1, y=cntH+1, s = str(cntH))
    plt.text(x = 2, y=cntT+1, s = str(cntT))
    plt.legend()
    plt.show()

def methodSelection(m ,n):
    if m == 1:
        return method1(n)
    elif m == 2:
        return method2(n)
    elif m ==3:
        return method3(n)
m = int(input('Method number: '))
n = int(input('Times to repeat: '))
methodSelection(m, n)
