import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import expit

training_set = "./data/digitstrain.txt"
validation_set = "./data/digitsvalid.txt"
test_set = "./data/digitstest.txt"
epochs = 200
learning_rate = 0.1

# def a():

# TODO: Initialize weights and biases
"""
    Sample weights from uniform distribution
    as discussed in class.
"""

# TODO: Feedforward

# TODO: Calculate cross entropy
# def cross_entropy():

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

''' Load Training Data'''
# (3000, 785), (3000, 784), (3000, 1)
# x_train, y_train = load_data(training_set)

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
a = np.array([1, 2, 3])
b = np.array([2, 3, 5, 6])
print a
print sigmoid(a)
print softmax(a)
print softmax(b)

#Function chooser
# func_arg = {"-a": a}
# if __name__ == "__main__":
#     func_arg[sys.argv[1]]()
