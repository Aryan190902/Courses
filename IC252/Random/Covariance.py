#Courses, Covariance Calculator:
import numpy
def Covariance():
  try:
    x = list(map(float, input('Enter the values of x separated by commas:').split(',')))
    y = list(map(float, input('Enter the values of y separated by commas:').split(',')))
    if len(x) == len(y):
      print("")
      print(numpy.cov(x, y))
    else:
      print("")
      print("Difference in number of values of x and y, Try again...")
      print("")
      return Covariance()
  except ValueError as v:
    print("")
    print('The input values are incorrect:', v)
    print("")
    return Covariance()

Covariance()
