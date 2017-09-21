import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import expit


training_set = "./data/digitstrain.txt"
validation_set = "./data/digitsvalid.txt"
test_set = "./data/digitstest.txt"
epochs = 10     # 200
layer_size = {'1': 100, '2':10}
weights = {}
biases = {}

def a():
    """
        Train the model
    """
    eta = 0.1   # learning rate
    # Load Training Data (3000, 785)
    # (3000, 784), (3000, 1)
    x_train, y_train = load_data(training_set)
    num_training_example = x_train.shape[0]
    batch_size = num_training_example
    # batch_size = 10
    layer_size['0'] = x_train.shape[1]
    # Initialize weights(a dictionary holds all the weightss)
    biases = {'1': 0, '2': 0}
    weights['1'] = init_weights(layer_size['0'], layer_size['1'])   # (784, 100)
    weights['2'] = init_weights(layer_size['1'], layer_size['2'])   # (100, 10)
    # print weights['1'].shape, weights['2'].shape
    loss = 0    # Cross entropy loss
    # Run epochs times
    # One epoch
    # TODO: Add validation set
    for e in range(epochs):
        for i in range(batch_size):
            x = x_train[i, :].reshape(len(x_train[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['2'], 1))
            y[int(y_train[i,0])] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of last layer
            a2 = feedforward(weights['2'], h1, biases['2'])
            o = softmax(a2)
            loss += cross_entropy(o, y)
            print "Before SGD: Loss %s" % loss
            # Update weights['2']
            w2_gradient = np.dot(h1, np.transpose(softmax_derivative(o, y)))   # 100*10
            sgd(w2_gradient, '2', eta)
            # Update weights['1']
            loss_over_h1 = np.dot(weights['2'], softmax_derivative(o, y))   # (100, 1)
            loss_over_a1 = np.multiply(loss_over_h1, sigmoid_derivative(a1))    #(100, 1)
            w1_gradient = np.dot(x, np.transpose(loss_over_a1))
            sgd(w1_gradient, '1', eta)
            print "After SGD: Loss %s" % loss

        print "##### Epoch %s Loss %s" % (e + 1, loss / num_training_example)

    # print loss

# TODO: Backpropagation
# def backprop(a1, f, y, eta):
#     w2_gradient = softmax_derivative(f, y)
#     print w2_gradient.shape
#     # update weights['2']
#
#     # update weights['1']
#
#     print w1_gradient.shape



# TODO: Implement SGD
def sgd(gradient, layer, eta):
    weights[layer] += eta * gradient




# Feedforward
def feedforward(W, x, b):
    """
        Input:
            W: weight of this layer
            x: neuron inputs
        Output:
            a = b + np.dot(W.T, x)
    """
    return b + np.dot(np.transpose(W), x)

# Initialize weights
def init_weights(size_k_1, size_k):
    """
        Sample weights from uniform distribution
        as discussed in class.
        Input:
            size_k_1, size_k: sizes of layer k - 1 and k
        Ouput:
            matrix of initialized weights: size_k_1 * size_k
    """
    b = np.sqrt(6) / np.sqrt(size_k + size_k_1)
    weights = np.random.uniform(-b, b, (size_k_1, size_k))
    return weights

# Calculate cross entropy
def cross_entropy(a, y):
    """
        Input:
            a: output of the neural nets (vector, output of softmax function)
            y: desired output (vector, 0s or 1s)
        Output:
            cross entropy of this training example
    """
    return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

def sigmoid(x):
    """
        Compute sigmoid function:
        return 1/(1 + exp(-x))
    """
    return expit(x)

def sigmoid_derivative(x):
    """
        Return the derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    """
        Input: an array
        Output: an array of softmax function of each element
    """
    return np.exp(x) / float(sum(np.exp(x)))

# Gradient of the softmax layer (output layer)
def softmax_derivative(f, y):
    """
        Input:
            f: output of the softmax layer
            y: indicator function(desired output)
        Output:
            partial derivative of softmax layer
    """
    return -(y - f)

def load_data(data_file):
    """
        Input: file to be converted
        Output: x, y numpy array (float)
    """
    data_array = np.loadtxt(data_file, delimiter=',')
    row = data_array.shape[0]
    col = data_array.shape[1]

    x = data_array[:,0:col - 1]
    y = data_array[:, col - 1].reshape(row, 1)
    # Test
    print "### Load Data File", data_file + " ###"
    print x.shape, y.shape
    return x, y



''' Load Validation Data '''
# (1000, 785), (1000, 784), (1000, 1)
# x_valid, y_valid = load_data(validation_set)

''' Load Test Data '''
# (3000, 784), (3000, 1)
# x_test, y_test = load_data(test_set)

''' Visualization '''
# plt.imshow(x_train[0, :])
# plt.imshow(x_test[0, :].reshape(28, 28), cmap='gray', origin='lower')
# plt.show()

''' Test functions'''
# a = np.array([1, 2, 3])
# b = np.array([2, 3, 5, 6])
# print a
# print sigmoid(a)
# print softmax(a)
# print softmax(b)

# Function chooser
func_arg = {"-a": a}
if __name__ == "__main__":
    func_arg[sys.argv[1]]()
