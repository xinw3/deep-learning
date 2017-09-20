import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

training_set = "./data/digitstrain.txt"
validation_set = "./data/digitsvalid.txt"
test_set = "./data/digitstest.txt"

# def a():


def load_data(training_set, validation_set, test_set):

    #(3000, 784)
    x_train = np.loadtxt(training_set, delimiter=',',  usecols=range(784))
    # y_train = np.l
    # Visualization
    print x_train.shape
    # return x_train, y_train, x_validate, y_validate, x_test, y_test


load_data(training_set, validation_set, test_set)

#Function chooser
# func_arg = {"-a": a}
# if __name__ == "__main__":
#     func_arg[sys.argv[1]]()
