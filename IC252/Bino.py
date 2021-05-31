#Courses, Binomial Distribution
import math
def Bin():
  try:
    n = int(input('Value of n: '))
    p = float(input('Value of p: '))
    x = int(input('Number of successes needed:'))
    fact = math.factorial(n)
    if p>=0 and p<=1:
      add = (fact/(math.factorial(n - x)*math.factorial(x)))*(p**x)*((1-p)**(n - x))
      print(f'X ~ Bin({n}, {p}): ', add)
    else:
      print('')
      print('Range of probability p is between 0 to 1, Try again...')
      print('')
      return Bin()
  except ValueError as v:
    print('')
    print('Error: ', v)
    print('')
    return Bin()

Bin()