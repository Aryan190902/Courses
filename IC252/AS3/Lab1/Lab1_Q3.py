import random
import matplotlib.pyplot as plt
import statistics
lst = []
probLst = []
for j in range(10):
    for i in range(1000):
        total_score = random.randint(2, 12)
        lst.append(total_score)
    favorable = 0
    for i in lst:
        if i > 8 or i%2 != 0 :
            favorable += 1
    lst = []
    probLst.append(favorable/1000)
print(probLst)
avg = [statistics.mean(probLst)]*10
plt.plot([i+1 for i in range(10)], avg, color='green')
plt.plot([i+1 for i in range(10)], probLst, color='blue')
plt.show()



