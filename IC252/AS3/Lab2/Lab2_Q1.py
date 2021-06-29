from fractions import Fraction
# You can also input fraction values.
dot_sent = float(Fraction(input('Probability of sending Dot: ')))
dot_mistake = float(Fraction(input('Probability of mistakenly receiving a dash: ')))

def Prob(x, y):
    p = (x*(1-y))/(x*(1-y) + (1-x)*y)
    return "{0:.3f}".format(p) 

print(f"P(dot sent | dot recieved): {Prob(dot_sent, dot_mistake)}")