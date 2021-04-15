import numpy as np

class DenseLayer():

    def __init__(self, inputLength, N, f):

        self.f = f
        limit = np.sqrt(6/(inputLength + N))
        self.weights = np.random.uniform(-limit,  limit, size = (N, inputLength))
        self.biases = np.random.uniform(-limit, limit, size = (N,1))

        self.yOut = np.empty((N,1))
        self.z = np.zeros((N,1))

        self.delta = np.zeros((N,1))
        self.weightsGradient = np.zeros((N, inputLength))
        self.deltaSum = np.zeros((N,1))

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
            layer.weights -= learningRate*layer.weightsGradient/batchSize
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
            trainingCost = 0

            for i in range(batchSize):

                yNetwork = self.forwardPass[xTraining[currentIndex]]
                yExpected = yTraining[currentIndex]
                self.backPropagation(trainingData, yExpected)
                
                trainingCost += self.costFunction(yNetwork, yExpected)

                currentIndex += 1
                if currentIndex == len(xTraining):
                    currentIndex = 0

            trainingCost /= batchSize
            validationCost = self.validate(validationData)
            print("Training Cost: ", trainingCost, "Validation Cost: ", validationCost)

    def backPropagation(self, yNetwork, yExpected):
        """
        This method should take the output of the network (yNetwork) and
        the expected value of y (yExpected). These values should then be
        used in conjunction with the cost function, layer weights, activation 
        function, and previously calculated deltas, to determine the gradient at
        each layer.
        
        To access the weights of the first hidden layer:
        
            self.layers[0].weights
            
        or to evaluate the derivative of the activation function assigned 
        to the last layer (if f is defined following the format described
        below):
            
            z = self.layers[-1].z
            fPrime = self.layers[-1].f(z, derivative = True)
            
        Over a training batch of "batchSize" samples, the gradient and delta 
        results for each layer "n" should be summed to:
        
            self.layer[n].weightsGradient 
        
        and
            
            self.layer[n].deltaSum
            
        so when these arrays are devided by "batchSize" in self.updateTheta
        the weights and biases are updated based on the average gradient values 
        for the batch.
        
        """
        
def identity(z, derivative = false):
# An example activation function, it can be passed to the "Dense" class and then 
# called with "derivative = True" durring backpropagation. You'll need to define
# your own activation functions. Different layers can use different activation 
# functions.
    if derivative:
        return np.ones(len(z))
    else:
        return z

totalSamples = 100

# Arrays to contain (X, Y).
# Note that each of the x and y samples are defined as having two dimensions
# even though they are both 'vectors'. This is required to maintain dimensional
# consistecy (from NumPy's perspective) over all required matrix operations.
xs = np.empty((totalSamples, 10, 1))
ys = np.empty((totalSamples, 2, 1))

trainingData = [xs[:80,:,:], ys[:80,:,:]]
validationData = [xs[80:,:,:], ys[80:,:,:]]

# dense1 and dense1 define the layers for an ANN with an input, hidden and output layer.
# The first argument gives the input size, the second argument is the number of neurons
# in the layer and the final argument is the activation function assigned to that layer.
# There is no layer object for the input layer as it is equal to the input data.
# Note that the number of neurons in the first layer dictates the input size of the
# next layer.
dense1 = DenseLayer(1, 2, identity)
dense2 = DenseLayer(2 1, identiy)

layers = [dense1, dense2]

batchSize = 5
epochs = 10
learningRate = 1e-4

myNetwork = neuralNetwork(layers)
# With a backpropagation method defined the line below should initiate training of the 
# ANN, though as the xs and ys are empty arrays, it'll be training on the noise of
# your CPU!
myNetwork.train(trainingData, validationData, batchSize, epochs, learningRate)
