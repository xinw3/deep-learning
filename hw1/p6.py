import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

training_set = "./data/digitstrain.txt"
validation_set = "./data/digitsvalid.txt"
test_set = "./data/digitstest.txt"

# def a():


def load_data(training_set, validation_set, test_set):

    ''' Load Training Data'''
    # (3000, 785)
    training_data = np.loadtxt(training_set, delimiter=',')
    train_row = training_data.shape[0]
    train_col = training_data.shape[1]
    # (3000, 784)
    x_train = training_data[:,0:train_col - 1]
    # (3000, 1)
    y_train = training_data[:, train_col - 1].reshape(train_row, 1)

    ''' Load Validation Data '''
    validation_data = np.loadtxt(validation_set, delimiter=',')
    valid_row = validation_data.shape[0]
    valid_col = validation_data.shape[1]

    x_valid = validation_data[:,0:valid_col - 1]
    y_valid = validation_data[:, valid_col - 1].reshape(valid_row, 1)


    # Visualization
    print x_valid.shape, y_valid.shape
    # return x_train, y_train, x_validate, y_validate, x_test, y_test


load_data(training_set, validation_set, test_set)

#Function chooser
# func_arg = {"-a": a}
# if __name__ == "__main__":
#     func_arg[sys.argv[1]]()
