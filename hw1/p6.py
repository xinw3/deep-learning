import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

training_set = "digitstrain.txt"
validation_set = "digitsvalid.txt"
test_set = "digitstest.txt"

def a():


def load_data(training_set, validation_set, test_set):

    x_train, y_train = p.loadtxt(training_set, delimiter=',',  usecols=(0:783, 784))
    # Visualization
    print x_train.shape, y_train.shape
    # return x_train, y_train, x_validate, y_validate, x_test, y_test


load_data(training_set, validation_set, test_set)

#Function chooser
# func_arg = {"-a": a}
# if __name__ == "__main__":
#     func_arg[sys.argv[1]]()
