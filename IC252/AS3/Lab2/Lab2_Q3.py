path = str(input('Enter the path of file: '))
X = str(input('First Alphabet: '))
Y = str(input('Second Alphabet: '))
with open(path, encoding='utf8') as f:
    text = str(f.read().splitlines())

small = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
 "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
number = 0
for i in text:
    if i.lower() in small:
        number += 1

def Prob(x):
    cnt = 0
    for i in text:
        if i.lower() == x.lower():
            cnt += 1
    p = cnt/number
    return "{0:.4f}".format(p)

def conditionalProbability(x, y):
    cnt = text.lower().count(y.lower() + x.lower())
    cntY = text.lower().count(y.lower())
    return "{0:.4f}".format(cnt/cntY)

print("Total characters: ", number)
print("P(X): ", Prob(X))
print("P(Y): ", Prob(Y))
print("P(X|Y): ", conditionalProbability(X, Y))
print("P(Y|X): ", conditionalProbability(Y, X))

