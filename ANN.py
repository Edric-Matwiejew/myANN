import numpy as np

class DenseLayer():

    def __init__(self, inputLength, N, f):

        self.type = 'dense'
        self.f = f
        limit = np.sqrt(6/(inputLength + N))
        self.weights = np.random.uniform(-limit,  limit, size = (N, inputLength))
        self.biases = np.random.uniform(-limit, limit, size = (N,1))

        self.yOut = np.empty((N,1))
        self.z = np.zeros((N,1))

        self.delta = np.zeros((N, 1))
        self.weightsGradient = np.zeros((N, inputLength))
        self.deltaSum = np.zeros((N, 1))

    def input(self, yIn):
        self.z[:] = np.dot(self.weights, yIn) + self.biases
        self.yOut[:] = self.f(self.z)
        return self.yOut

class InputLayer:

    def __init__(self, inputLength):

        self.type = 'input'
        self.yOut = np.empty((inputLength,1))

    def input(self, yIn):
        self.yOut[:] = yIn
        return self.yOut

class neuralNetwork():

    def __init__(self, layers):

        self.layers = layers
        self.costFunction = None

    def forwardPass(self, xs):
        yOut = xs
        for layer in self.layers:
            yOut = layer.input(yOut)
        return yOut

    def updateTheta(self, learningRate, batchSize):
        for layer in self.layers:
            if layer.type != 'input':
                layer.weights -= learningRate*layer.weightsGradient/batchSize
                layer.biases -= learningRate*layer.deltaSum/batchSize
                layer.weightsGradient[:] = 0
                layer.deltaSum[:] = 0

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

    def train(self, trainingData, validationData, batchSize, epochs, learningRate, costFunction, shuffle = False):

        self.costFunction = costFunction
        
        # An 'epoch' is a complete pass through the training set.
        
        xTraining = trainingData[0]
        yTraining = trainingData[1]

        nBatches = len(xTraining)//batchSize

        self.trainingPlot = []
        self.validationPlot = []
        self.batchPlot = []

        nBatch = 0

        indexes = np.arange(len(xTraining))

        for epoch in range(epochs + 1):

            print('Epoch: ', epoch, 'of ', epochs)
            
            epochComplete = False
            currentIndex = 0
            
            if shuffle:
                np.random.shuffle(indexes)
            
            while not epochComplete:
                
                trainingCost = 0
                
                for i in range(batchSize):
                    
                    dataIndex = indexes[currentIndex]

                    yNetwork = self.forwardPass(xTraining[dataIndex])
                    yExpected = yTraining[dataIndex]
                    self.backPropagation(yNetwork, yExpected)
                    self.updateTheta(learningRate, batchSize)
                    trainingCost += self.costFunction(yNetwork, yExpected)

                    currentIndex += 1
                    if currentIndex == len(xTraining):
                        epochComplete = True
                        break


                trainingCost /= i + 1
                validationCost = self.validate(validationData)

                nBatch += 1
                self.trainingPlot.append(trainingCost)
                self.validationPlot.append(validationCost)
                self.batchPlot.append(nBatch)

                print("Training Cost: ", trainingCost, "Validation Cost: ", validationCost, end = "\r")

            print("\n", end = "")

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
        
            self.layers[n].weightsGradient[:,:] 
        
        and
            
            self.layers[n].deltaSum[:]
            
        so when these arrays are devided by "batchSize" in self.updateTheta
        the weights and biases are updated based on the average gradient values 
        for the batch.
        
        """
        z = self.layers[-1].z
        self.layers[-1].delta[:] = self.costFunction(yNetwork, yExpected, derivative = True)*self.layers[-1].f(z, derivative = True)
        self.layers[-1].deltaSum[:] += self.layers[-1].delta 
        y_previous = self.layers[-2].yOut
        self.layers[-1].weightsGradient[:] += np.outer(self.layers[-1].delta, y_previous)

        for i in range(len(self.layers) - 2, 0, -1):
            z = self.layers[i].z
            # your code here
            self.layers[i].weightsGradient[:] += # your code here
            self.layers[i].deltaSum += # your code here

def identity(z, derivative = False):
# An example activation function, it can be passed to the "Dense" class and then 
# called with "derivative = True" durring backpropagation. You'll need to define
# your own activation functions. Different layers can use different activation 
# functions.
    if derivative:
        return np.ones(z.shape)
    else:
        return z

def relu(z, derivative = False):
    if derivative:
        return np.where(z >= 0, 1, 0)
    else:
        return np.clip(z, 0, np.inf)

def sigmoid(z, derivative = False):
    if derivative:
        return sigmoid(z)*(1 - sigmoid(z))
    else:
        return 1/(1 + np.exp(-z))

def MSE(yNetwork, yExpected, derivative = False):
    if derivative:
        return 2*(yNetwork - yExpected).sum()
    else:
        return ((yNetwork - yExpected)**2).sum()

