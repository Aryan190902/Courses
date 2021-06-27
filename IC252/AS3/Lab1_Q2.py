from math import factorial
total_cards = int(input('Enter the number of unique cards: '))
hand = int(input('Number of cards in 1 hand: '))

def comb(n, r):
    comb = factorial(n)/(factorial(n-r)*factorial(r))
    return comb

def prob():
    prob = comb(total_cards - 4, 4)/comb(total_cards, hand)
    return '{0:.4f}'.format(prob)
print(f"Total unique cards: {total_cards}")
print(f'Cards in one hand: {hand}')
print(f"The probability of getting all four of the 1s: {prob()}")