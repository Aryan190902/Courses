import numpy as np
from math import exp
import matplotlib.pyplot as plt
#1 a
lst = np.linspace(0, 0.3, 100)
values = []
for i in lst:
    values.append(57*exp(-57*i))
plt.plot(lst, values)
plt.xlabel('Waiting time in hours -->')
plt.ylabel('Probability Density -->')
plt.title("PDF")
plt.grid()


# 1 b
answer = 1 - exp(-57*(1/60))
print("P(X <= 1)=", "{0:.4f}".format(answer))

# 1 c
answer = exp(-57*(1/60)) - exp(-57*(2/60))
print("P(X>1, X<=2)=", "{0:.4f}".format(answer))

# 1 d
answer = exp(-57*(2/60))
print("P(X>2)=", "{0:.4f}".format(answer))

# 1 e
# now Lambda will be 57*2 = 114 
answer = exp(-114*(1/60)) - exp(-114*(2/60))
print("P(X>1, X<=2) when lambda is doubled:", "{0:.4f}".format(answer))

plt.show()