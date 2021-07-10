from numpy import random
from scipy import integrate
import numpy as np
# Q1
m = [100, 1000, 10000]
random.seed(17)
print("Correct Pi Value:", np.pi)
for i in m:
    add = 0
    for j in range(i):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1:
            add += 1
    print(f"\nFor m={i}:")  # This is the Value of Pi, using Monte Carlo Method
    print('Pi=', 4*add/i)

# Q2
random.seed(420)


def func(a):
    return 2/(1 + a*a)


print("\nCorrect Numerical Value:", integrate.quad(func, 0, 1)[0])
# Here as the function is decreasing Func, So maximum value will be 2 at x = 1
# ThereFore, c = 2, a = 0, b = 1, c*(b - a) = 2
for i in m:
    add = 0
    for j in range(i):
        x = random.randint(0, 1001)/1000
        y = random.randint(0, 2001)/1000
        if y <= func(x):
            add += 1
    print(f"\nFor m={i}:")
    print('Intergration:', 2*add/i)
print("So, the computated value is near to the original value")

# 3
print("Correct Value of e:",  2.71828)
for i in m:
    add = 0
    for k in range(10000):
        cnt = 1
        n = random.randint(1, i+1, size=i)
        for j in range(i):
            if n[j] == j+1:
                cnt = 0
                pass
        if cnt == 1:
            add += 1
    print(f"\nFor m={i}:")
    print('e:', 10000/add)