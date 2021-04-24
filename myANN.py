import numpy as np
import matplotlib.pyplot as plt

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
            layer.biases -= learningRate*layer.deltaSum/batchSize
            layer.weightsGradient[:] = 0
            layer.deltaSum[:] = 0

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
        
        # An 'epoch' is a complete pass through the training set.
        
        xTraining = trainingData[0]
        yTraining = trainingData[1]

        nBatches = len(xTraining)//batchSize

        trainingPlot = []
        validationPlot = []
        batchPlot = []

        nBatch = 0

        indexes = np.arange(len(xTraining))

        for epoch in range(epochs + 1):
            
            epochComplete = False
            currentIndex = 0
            
            #Uncomment the line below to shuffle your data with each epoch.
            #np.random.shuffle(indexes)
            
            while not epochComplete:
                
                trainingCost = 0
                
                for i in range(batchSize):
                    
                    dataIndex = indexes[currentIndex]

                    yNetwork = self.forwardPass(xTraining[dataIndex])
                    yExpected = yTraining[dataIndex]
                    self.backPropagation(xTraining[dataIndex], yNetwork, yExpected)
                    self.updateTheta(learningRate, batchSize)
                    trainingCost += self.costFunction(yNetwork, yExpected)

                    currentIndex += 1
                    if currentIndex == len(xTraining):
                        epochComplete = True
                        break


                trainingCost /= i + 1
                validationCost = self.validate(validationData)

                nBatch += 1
                trainingPlot.append(trainingCost)
                validationPlot.append(validationCost)
                batchPlot.append(nBatch)
                plt.clf()
                plt.plot(batchPlot, trainingPlot, color = 'orange', label = "training")
                plt.plot(batchPlot, validationPlot, color = 'green', label = "validation")
                plt.legend()
                plt.show(block = False)
                plt.pause(0.00001)

                print("Training Cost: ", trainingCost, "Validation Cost: ", validationCost)

    def backPropagation(self, x, yNetwork, yExpected):
        """
        This method should take the input data (x), the output of the network (yNetwork) and
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


totalSamples = 100

# Arrays to contain (X, Y).
# Note that each of the x and y samples are defined as having two dimensions
# even though they are both 'vectors'. This is required to maintain dimensional
# consistecy (from NumPy's perspective) over all required matrix operations.
xShape = (10,1)
yShape = (1,1)
xs = np.empty((totalSamples, xShape[0], xShape[1]))
ys = np.empty((totalSamples, yShape[0], yShape[1]))

for i in range(totalSamples):
    xs[i] = np.random.uniform(low = 0, high = 1, size = xShape)
    ys[i] = np.sum(xs[i])

xs = xs/np.max(xs)
ys = ys/np.max(ys)

trainingData = [xs[:80,:,:], ys[:80,:,:]]
validationData = [xs[80:,:,:], ys[80:,:,:]]

# dense1 and dense1 define the layers for an ANN with an input, hidden and output layer.
# The first argument gives the input size, the second argument is the number of neurons
# in the layer and the final argument is the activation function assigned to that layer.
# There is no layer object for the input layer as it is equal to the input data.
# Note that the number of neurons in the first layer dictates the input size of the
# next layer.
dense1 = DenseLayer(10, 100, relu) 
dense2 = DenseLayer(100, 200, sigmoid)
dense3 = DenseLayer(200, 100, sigmoid)
dense4 = DenseLayer(100, 1, identity)
 

layers = [dense1, dense2, dense3, dense4]

batchSize = 8
epochs = 50 
learningRate = 1e-4

myNetwork = neuralNetwork(layers)
# With a backpropagation method defined the line below should initiate training of the 
# ANN, though as the xs and ys are empty arrays, it'll be training on the noise of
# your CPU!
myNetwork.train(trainingData, validationData, batchSize, epochs, learningRate)

for x, y in zip(validationData[0], validationData[1]):
    print("y expected: ", y, "y Network:", myNetwork.forwardPass(x))

plt.show(block = True)
