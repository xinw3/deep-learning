import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

training_set = "./data/digitstrain.txt"
validation_set = "./data/digitsvalid.txt"
test_set = "./data/digitstest.txt"

# def a():


def load_data(data_file):
    data_array = np.loadtxt(data_file, delimiter=',')
    row = data_array.shape[0]
    col = data_array.shape[1]

    x = data_array[:,0:col - 1]
    y = data_array[:, col - 1].reshape(row, 1)
    # Visualization
    print "### Load Data File", data_file + " ###"
    print x.shape, y.shape
    return x, y

''' Load Training Data'''
# (3000, 785), (3000, 784), (3000, 1)
x_train, y_train = load_data(training_set)

''' Load Validation Data '''
# (1000, 785), (1000, 784), (1000, 1)
x_valid, y_valid = load_data(validation_set)

#Function chooser
# func_arg = {"-a": a}
# if __name__ == "__main__":
#     func_arg[sys.argv[1]]()
