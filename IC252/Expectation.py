def Expectation():
  try:
    x = list(map(float, input("Enter the values of x separated with commas:").split(',')))
    Px = list(map(float, input("Enter the values of P(x) separated with commas:").split(','))) # Probablity corresponding to x of the same index
    add = 0
    if len(x) == len(Px):
      for i in range(len(x)):
        add += x[i] * Px[i]
      print("The Expectaiton of the given input is:", add)
    else:
      print("number of values of x and P(x) is not equal, try again")
      return Expectation()
  except ValueError as v:
    print("Invalid Input:", v)
    return Expectation()

Expectation()