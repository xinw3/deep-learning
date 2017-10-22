import sys

import numpy as np
import matplotlib.pyplot as plt


# Data sets
training_set = "../mnist_data/digitstrain.txt"
validation_set = "../mnist_data/digitsvalid.txt"
test_set = "../mnist_data/digitstest.txt"


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

    fig, axs = plt.subplots(10, 10)
    # Remove horizontal space between axes
    fig.subplots_adjust(wspace=0, hspace=0)
    count = 0
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, count)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(np.reshape(np.ravel(x_test[count, :], order='F'),(28,28)), cmap='gray', origin='lower')
            count += 1
    plt.show()

def load_data(data_file):
    """
        Input: file to be converted
        Output: x, y numpy array (float)
    """
    data_array = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data_array)
    row = data_array.shape[0]
    col = data_array.shape[1]
    print "row=%s, col=%s" % (row, col)

    x = data_array[:,:col - 1]
    y = data_array[:, col - 1].reshape(row, 1)
    # Test
    print "### Load Data File %s ###" % data_file
    print x.shape, y.shape
    return x, y

func_arg = {"-a": a}
if __name__ == "__main__":
    func_arg[sys.argv[1]]()
