import statistics
def Var():
  try:
    n = int(input('Total number of values of x:'))
    x = list(map(float, input('Enter the values of x separated with commas:').split(',')))
    if len(x) == n:
      var = statistics.variance(x)
      print("")
      print('Variance of the input x are:',var)
    else:
      print('The number of values provided are incorrect, Try again...')
      return Var()
  except ValueError as v:
    print('Error:', v)    
Var()
