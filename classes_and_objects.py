
# consistent way of creating objects
l = []
# function in classes with a ".", we call them "methods".
l.append(1)

class storedSum():
    def __init__(self, startingValue):
        self.startingValue = startingValue
        self.sum = startingValue
    def addValue(self, value):
        self.sum += value
        self.isBig()
    def isBig(self):
        if self.sum > 100:
            print("This is big!")
    def __str__(self):
        return "the number is " + str(self.sum)
    def __mul__(self, x):
        self.sum *= x.sum
        return self

sumA = storedSum(10)

sumA.addValue(200)

sumB = storedSum(13)

print(sumA)
sumA * sumB * sumC


a = sumA.sum 
b = sumB.startingValue
print(a +  b)
print(sumA.sum, sumB.sum)


