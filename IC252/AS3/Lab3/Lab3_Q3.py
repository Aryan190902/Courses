import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import Counter
from statistics import mean

def Earth(x):
    bDay = []
    for i in range(x):
        d = random.randint(1, 365)
        bDay.append(d)
    common = Counter(bDay)
    c = 0
    for i in common:
        if common[i] > 1:
            c += 2
    return c
def Mars(x):
    bDay = []
    for i in range(x):
        d = random.randint(1, 687)
        bDay.append(d)
    common = Counter(bDay)
    c = 0
    for i in common:
        if common[i] > 1:
            c += 2
    return c
print("For n=25, c=", Earth(25))
print("On mars, when n= 32, c=", Mars(32))
n = int(input("Enter number of people in 1 room: "))
experiment = []
prob = 0
probBelow = []
probAbove = []
for i in range(1000):
    bDay = []
    below = [i for i in range(1, 151)]
    cntBelow = 0
    cntAbove = 0
    for j in range(n):
        d = random.randint(1, 365)
        bDay.append(d)
        if d in below:
            cntBelow += 1
        else:
            cntAbove += 1
    probAbove.append(cntAbove/n)
    probBelow.append(cntBelow/n)
    common = Counter(bDay)
    c = 0
    for j in common:
        if common[j] > 1:
            c += 2
    experiment.append(c)
    if c>=2:
        prob += 1

print(f"For n = {n}, Probability(c>=2)=", prob/1000)

plt.plot([i for i in range(1, 1001)], experiment, label='Experiments')
plt.plot([i for i in range(1000)], [mean(experiment)]*1000, color='red', label='Mean')
plt.xlabel('Number of Experiment -->')
plt.ylabel('Value of c -->')
plt.yticks([i for i in range(1, 11)])
plt.legend()
plt.show()

#d part
plt.plot([i for i in range(1, 1001)], probAbove,
 label='Number in range(151, 365)', color= 'red')
plt.plot([i for i in range(1, 1001)], probBelow,
 color= 'blue', label='Number in range(1, 150)')
plt.plot([i for i in range(1, 1001)], [mean(probAbove)]*1000, color ='violet',
 label= 'Mean of red Plot')
plt.plot([i for i in range(1, 1001)], [mean(probBelow)]*1000, color='brown',
 label='Mean of Blue plot')

plt.legend()
plt.show()