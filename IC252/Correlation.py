#Courses, Correlation Calculator
import statistics
import math
def Correlation():
  try: 
    x = list(map(float, input('Enter values of x: ').split(',')))
    y = list(map(float, input('Enter values of y: ').split(',')))
    if len(x) == len(y):
      add = 0
      addX = 0
      addY = 0
      meX = statistics.mean(x)
      meY = statistics.mean(y)
      for i in range(len(x)):
        add += (x[i] - meX)*(y[i] - meY)
        addX += (x[i] - meX)*(x[i] - meX)
        addY += (y[i] - meY)*(y[i] - meY)
      cor = add/math.sqrt(addX*addY)
      print('Correlation is:', cor)
    else:
      print('')
      print('Number of values in X and Y are not equal, Try again...')
      print('')
      return Correlation()
  except ValueError as v:
    print('')
    print('Error in the input values:', v)
    print('')
    return Correlation()

Correlation()