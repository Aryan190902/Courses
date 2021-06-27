from math import factorial

founders = int(input('Enter the number of founders: '))
num_of_post = int(input('Enter the number of posts: '))

def perm(n, r):
    permutation = factorial(n)/factorial(n-r)
    return permutation

print(f'Total number of founders: {founders}')
print(f'Number of posts: {num_of_post}')
print(f'The probability of getting all four of the 1s \
in a hand: {1/perm(founders, num_of_post)}')
