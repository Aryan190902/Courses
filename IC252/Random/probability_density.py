#Courses, PDF finding values
import math
def PDF():
  try:
    me = float(input('Enter the mean value of the distribution: '))
    var = float(input('Enter the variance of the distribution: '))
    x = float(input('Value of x: '))

    exp = math.exp(-1*(x - me)*(x - me)/(2 * math.sqrt(var)))
    val = exp/math.sqrt(2*math.pi*var)

    print('')
    print('Probability Density: ', val)
  except ValueError as v:
    print('')
    print('Invalid input:', v)
    print('')
    return PDF()

PDF()