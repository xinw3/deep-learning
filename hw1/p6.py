import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import expit

training_set = "./data/digitstrain.txt"
validation_set = "./data/digitsvalid.txt"
test_set = "./data/digitstest.txt"

# def a():

"""
    Compute sigmoid function:
    return 1/(1 + exp(-x))
"""
def sigmoid(x):
    return expit(x)

"""
    Input: an array
    Output: an array of softmax function of each element
"""
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

"""
    Input: file to be converted
    Output: x, y numpy array (float)
"""
def load_data(data_file):
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
a = np.array([[3.0, 1.0, 0.2]])
print softmax(a)

#Function chooser
# func_arg = {"-a": a}
# if __name__ == "__main__":
#     func_arg[sys.argv[1]]()
