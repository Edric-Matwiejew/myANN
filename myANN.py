import numpy as np
from scipy.linalg import expm
from watchpoints import watch

class DenseLayer():

    def __init__(self, inputLength, N, f):

        self.f = f
        limit = np.sqrt(6/(inputLength + N))
        self.weights = np.random.uniform(-limit,  limit, size = (N, inputLength))
        self.biases = np.random.uniform(-limit, limit, size = (N,1))

        self.yOut = np.empty((N,1))
        self.fPrime = np.empty((N,1))
        self.z = np.zeros((N,1))

        self.delta = np.zeros((N,1))
        self.weightsGradient = np.zeros((N, inputLength))

    def input(self, yIn):
        self.z[:] = np.dot(self.weights, yIn) + self.biases
        self.yOut[:] = self.f(self.z)
        return self.yOut

class neuralNetwork():

    def __init__(self, layers):

        self.layers = layers

    def forwardPass(self, xs):
        yOut = xs
        for layer in self.layers:
            yOut = layer.input(yOut)
        return yOut

    def updateTheta(self, learningRate, batchSize):
        for layer in self.layers:
            layer.weights -= (learningRate*layer.weightsGradient/batchSize).astype(np.float64)
            layer.weightsGradient[:] = 0

    def costFunction(self, yNetwork, yExpected):
        return np.sum((yNetwork - yExpected)**2)

    def validate(self, validationData):

        xValidate = validationData[0]
        yValidate = validationData[1]
        nSamples = len(xValidate)

        cost = 0

        for i in range(nSamples):
            yNetwork = self.forwardPass(xValidate[i][:,:])
            yExpected = yValidate[i][:,:]
            cost += self.costFunction(yNetwork, yExpected)

        return cost/nSamples

    def train(self, trainingData, validationData, batchSize, epochs, learningRate):

        xTraining = trainingData[0]
        yTraining = trainingData[1]

        nBatches = len(xTraining)//batchSize

        for epoch in range(epochs + 1):

            currentIndex = 0
            cost = 0

            for i in range(batchSize):

                yNetwork = self.forwardPass[xTraining[currentIndex]]
                yExpected = yTraining[currentIndex]
                cost += self.costFunction(yNetwork, yExpected)
                self.backPropagation(trainingData, yExpected)

                currentIndex += 1
                if currentIndex == len(xTraining):
                    currentIndex = 0

            trainingCost /= batchSize
            validationCost = self.validate(validationData)
            print(trainingCost, validationCost)

    def backPropagation(self, trainingData, yExpected):

totalSamples = 100

xs = np.empty((totalSamples, 1, 1))
ys = np.empty((totalSamples, 1, 1))

trainingData = [xs[:80,:,:], ys[:80,:,:]]
validationData = [xs[:80,:,:], ys[:80,:,:]]

dense1 = DenseLayer(1, 2, none)
dense2 = DenseLayer(2 1, none)

layers = [dense1, dense2]

batchSize = 5
epochs = 10
learningRate = 1e-4

myNetwork = neuralNetwork(layers, costFunction)
myNetwork.train(trainingData, validationData, batchSize, epochs, learningRate)
