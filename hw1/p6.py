import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import expit


training_set = "./data/digitstrain.txt"
validation_set = "./data/digitsvalid.txt"
test_set = "./data/digitstest.txt"
epochs = 200
layer_size = {'1': 100, '2':10}

def a():
    """
        Train the model
    """
    eta = 0.1   # learning rate
    # Load Training Data (3000, 785)
    # (3000, 784), (3000, 1)
    x_train, y_train = load_data(training_set)
    # Initialize weights(a dictionary holds all the weightss)
    weights = {}
    layer_size['0'] = x_train.shape[1]
    weights['1'] = init_weights(layer_size['0'], layer_size['1'])
    weights['2'] = init_weights(layer_size['1'], layer_size['2'])
    print weights['1'].shape, weights['2'].shape


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

# TODO: Feedforward

# Calculate cross entropy
def cross_entropy(a, y):
    """
        Input:
            a: output of the neural nets (vector, output of softmax function)
            y: desired output (vector, 0s or 1s)
        Output:
            cross entropy of this training example
    """
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

# TODO: Backpropagation

# TODO: Implement SGD



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
