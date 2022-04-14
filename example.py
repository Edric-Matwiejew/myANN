import numpy as np
from ANN import *

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

# dense1 to dense4 define the layers for an ANN with an input, 3 hidden layers and output layer.
# The first argument gives the input size, the second argument is the number of neurons
# in the layer and the final argument is the activation function assigned to that layer.
# There is no layer object for the input layer as it is equal to the input data.
# Note that the number of neurons in the first layer dictates the input size of the
# next layer.
inlayer = InputLayer(10) 
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
myNetwork.train(
        trainingData,
        validationData,
        batchSize,
        epochs,
        learningRate,
        MSE)

for x, y in zip(validationData[0], validationData[1]):
    print("y expected: ", y, "y Network:", myNetwork.forwardPass(x))

plt.show(block = True)
