import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm

fig, ax = plt.subplots(3, 6)
n = [1, 2, 4, 8, 16, 32]
def mean(x):
    ad = 0
    for i in x:
        ad += i
    return ad/len(x)

expoMean = []
unifMean = []
bernMean = []
for i in n:
    exp = np.zeros(1000)
    unif = np.zeros(1000)
    bern = np.zeros(1000)
    for j in range(i):
        exp += np.random.exponential(1, size=1000)
        unif += np.random.uniform(1, 2, size=1000)
        bern += bernoulli.rvs(p=0.2, size=1000)
    expoMean.append(exp/i)
    unifMean.append(unif/i)
    bernMean.append(bern/i)

fig, axs = plt.subplots(3, 6, figsize=(12, 4))
#a part
axs[0, 0].set_ylabel("Exponential")
number, bins, patch = axs[0, 0].hist(expoMean[0], density=True)
axs[0, 0].plot(bins, norm.pdf(bins, loc=np.mean(expoMean[0]),
 scale=(np.std(expoMean[0])/n[0])))
axs[0, 0].set_title('n = 1')

number, bins, patch = axs[0, 1].hist(expoMean[1], density=True)
axs[0, 1].plot(bins, norm.pdf(bins, loc=np.mean(expoMean[1]),
 scale=(np.std(expoMean[1])/n[1])))
axs[0, 1].set_title('n = 2')

number, bins, patch = axs[0, 2].hist(expoMean[2], density=True)
axs[0, 2].plot(bins, norm.pdf(bins, loc=np.mean(expoMean[2]),
 scale=(np.std(expoMean[2])/n[2])))
axs[0, 2].set_title('n = 4')

number, bins, patch = axs[0, 3].hist(expoMean[3], density=True)
axs[0, 3].plot(bins, norm.pdf(bins, loc=np.mean(expoMean[3]),
 scale=(np.std(expoMean[3])/n[3])))
axs[0, 3].set_title('n = 8')

number, bins, patch = axs[0, 4].hist(expoMean[4], density=True)
axs[0, 4].plot(bins, norm.pdf(bins, loc=np.mean(expoMean[4]),
 scale=(np.std(expoMean[4])/n[4])))
axs[0, 4].set_title('n = 16')

number, bins, patch = axs[0, 5].hist(expoMean[5], density=True)
axs[0, 5].plot(bins, norm.pdf(bins, loc=np.mean(expoMean[5]),
 scale=(np.std(expoMean[5])/n[5])))
axs[0, 5].set_title('n = 32')

#b part
axs[1, 0].set_ylabel("Uniform")
number, bins, patch = axs[1, 0].hist(unifMean[0], density=True)
axs[1, 0].plot(bins, norm.pdf(bins, loc=np.mean(unifMean[0]),
 scale=(np.std(unifMean[0])/n[0])))

number, bins, patch = axs[1, 1].hist(unifMean[1], density=True)
axs[1, 1].plot(bins, norm.pdf(bins, loc=np.mean(unifMean[1]),
 scale=(np.std(unifMean[1])/n[1])))

number, bins, patch = axs[1, 2].hist(unifMean[2], density=True)
axs[1, 2].plot(bins, norm.pdf(bins, loc=np.mean(unifMean[2]),
 scale=(np.std(unifMean[2])/n[2])))

number, bins, patch = axs[1, 3].hist(unifMean[3], density=True)
axs[1, 3].plot(bins, norm.pdf(bins, loc=np.mean(unifMean[3]),
 scale=(np.std(unifMean[3])/n[3])))

number, bins, patch = axs[1, 4].hist(unifMean[4], density=True)
axs[1, 4].plot(bins, norm.pdf(bins, loc=np.mean(unifMean[4]),
 scale=(np.std(unifMean[4])/n[4])))

number, bins, patch = axs[1, 5].hist(unifMean[5], density=True)
axs[1, 5].plot(bins, norm.pdf(bins, loc=np.mean(unifMean[5]),
 scale=(np.std(unifMean[5])/n[5])))

#c part 
axs[2, 0].set_ylabel("Bernoulli")
number, bins, patch = axs[2, 0].hist(bernMean[0], density=True)
axs[2, 0].plot(bins, norm.pdf(bins, loc=np.mean(bernMean[0]),
 scale=(np.std(bernMean[0])/n[0])))

number, bins, patch = axs[2, 1].hist(bernMean[1], density=True)
axs[2, 1].plot(bins, norm.pdf(bins, loc=np.mean(bernMean[1]),
 scale=(np.std(bernMean[1])/n[1])))

number, bins, patch = axs[2, 2].hist(bernMean[2], density=True)
axs[2, 2].plot(bins, norm.pdf(bins, loc=np.mean(bernMean[2]),
 scale=(np.std(bernMean[2])/n[2])))

number, bins, patch = axs[2, 3].hist(bernMean[3], density=True)
axs[2, 3].plot(bins, norm.pdf(bins, loc=np.mean(bernMean[3]),
 scale=(np.std(bernMean[3])/n[3])))

number, bins, patch = axs[2, 4].hist(bernMean[4], density=True)
axs[2, 4].plot(bins, norm.pdf(bins, loc=np.mean(bernMean[4]),
 scale=(np.std(bernMean[4])/n[4])))

number, bins, patch = axs[2, 5].hist(bernMean[5], density=True)
axs[2, 5].plot(bins, norm.pdf(bins, loc=np.mean(bernMean[5]),
 scale=(np.std(bernMean[5])/n[5])))
plt.tight_layout()
plt.show()
