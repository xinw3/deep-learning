import sys

import numpy as np
import matplotlib.pyplot as plt


# Data sets
training_set = "../mnist_data/digitstrain.txt"
validation_set = "../mnist_data/digitsvalid.txt"
test_set = "../mnist_data/digitstest.txt"

# tunable parameters
cd_steps = 1    # run cd_steps iterations of Gibbs Sampling
num_hidden_units = 100  # number of hidden units

# parameters of normal distribution in weights initialization
mean = 0    # mean
stddev = 0.1    # standard deviation

def a():
    '''
        Basic Generalization
    '''
    # Load Training Data (3000, 785)
    x_train, y_train = load_data(training_set)     # (3000, 784), (3000, 1)
    # Load Validation Data (1000, 785)
    x_valid, y_valid = load_data(validation_set)    # (1000, 784), (1000, 1)
    # Load Test Data (3000, 785)
    x_test, y_test = load_data(test_set)            # (3000, 784), (3000, 1)

    # Get number of examples
    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]
    num_test_example = x_test.shape[0]

    num_input = x_train.shape[1]
    num_hidden = num_hidden_units

    weights, visbias, hidbias = \
            init_params(mean, stddev, num_input, num_hidden)


    print weights.shape, visbias.shape, hidbias.shape

# Initialize weights
def init_params(mean, stddev, size_k_1, size_k):
    """
        Sample weights from normal distribution with mean 0 stddev 0.1
        Input:
            weights, biases: the parameters to be initialized
            mean: the mean of the normal distribution (default 0)
            stddev: the standard deviation of normal distribution (default 0.1)
            size_k_1, size_k: sizes of layer k-1(input) and k(hidden)
        Ouput:
            weights: matrix of initialized weights: size_k_1 * size_k
            biases: biases terms of size_k
    """
    weights = np.random.normal(mean, stddev, (size_k_1, size_k))
    visbias = np.zeros((size_k_1, 1))
    hidbias = np.zeros((size_k, 1))
    return weights, visbias, hidbias

def load_data(data_file):
    """
        Input: file to be converted
        Output: x, y numpy array (float)
    """
    data_array = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data_array)
    row = data_array.shape[0]
    col = data_array.shape[1]

    x = data_array[:,:col - 1]
    y = data_array[:, col - 1].reshape(row, 1)
    # Test
    print "### Load Data File %s ###" % data_file
    print x.shape, y.shape
    return x, y

func_arg = {"-a": a}
if __name__ == "__main__":
    func_arg[sys.argv[1]]()
