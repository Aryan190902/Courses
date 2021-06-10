#Courses, IC152, fibonacci
memo = {}
def Fibo(n):
  if n == 0:
    return 0
  elif n == 1:
    return 1
  else:
    try:
       return memo[f'Fibo({n-1})'] + memo[f'Fibo({n-2})']
    except KeyError:
      memo[f'Fibo({n-1})'] = Fibo(n-1)
      memo[f'Fibo({n-2})'] = Fibo(n-2)
      return memo[f'Fibo({n-1})'] + memo[f'Fibo({n-2})']


n = int(input('Find the nth number of Fibonacci Series: '))
print(Fibo(n))