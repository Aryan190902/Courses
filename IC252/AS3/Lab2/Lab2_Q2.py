defectiveX = int(input("Defective: "))
partY = int(input("Partially Defective: "))
goodZ = int(input('Acceptable: '))

def Prob(x, y, z):
    p = z/(y+z)
    return "{0:.3f}".format(p) 
print(f"P(accepted | non-defective) = {Prob(defectiveX, partY, goodZ)}")
