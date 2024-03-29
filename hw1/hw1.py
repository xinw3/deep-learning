import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
from scipy.interpolate import spline

def q1_1():
    x_axis = np.array([6, 7, 8, 9, 10, 11, 12])
    training_error = np.array([100, 50, 35, 30, 29, 28, 27])
    test_error = np.array([110, 55, 38, 33, 40, 55, 70])
    x_new = np.linspace(x_axis.min(),x_axis.max(),300) #300 represents number of points to make between T.min and T.max
    training_smooth = spline(x_axis,training_error,x_new)
    test_smooth = spline(x_axis,test_error,x_new)
    plt.text(6, 20, str(0))
    # remove the x, y label
    plt.xticks([])
    plt.yticks([])
    # add x, y label to plot
    plt.xlabel("Model Complexity")
    plt.ylabel("Error Rate")
    plt.plot(x_new,training_smooth, label='training error')
    plt.plot(x_new,test_smooth, label='test error')
    plt.axvline(x=9, color='k', linestyle='--')
    # show legend
    plt.legend();
    plt.show()

def q1_2():

    x_axis = np.array([6, 7, 8, 9, 10, 11, 12])
    training_error = np.array([10, 30, 45, 55, 57, 57, 57])
    test_error = np.array([100, 90, 80, 70, 63, 62, 62])
    x_new = np.linspace(x_axis.min(),x_axis.max(),300) #300 represents number of points to make between T.min and T.max
    training_smooth = spline(x_axis,training_error,x_new)
    test_smooth = spline(x_axis,test_error,x_new)
    plt.text(6, 10, str(0))
    # remove the x, y label
    plt.xticks([])
    plt.yticks([])
    # add x, y label to plot
    plt.xlabel("Training Set Size")
    plt.ylabel("Error Rate")
    plt.plot(x_new,training_smooth, label='training error')
    plt.plot(x_new,test_smooth, label='test error')
    # show legend
    plt.legend()
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def q5():
    from scipy.special import expit
    x_axis = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, \
                        4, 5, 6, 7, 8, 9, 10])
    tanh = np.tanh(x_axis)
    sig = expit(x_axis)
    x_new = np.linspace(x_axis.min(),x_axis.max(),300) #300 represents number of points to make between T.min and T.max
    tanh_smooth = spline(x_axis,tanh,x_new)
    sig_smooth = spline(x_axis,sig,x_new)
    # remove the x, y label
    # plt.xticks([])
    # plt.yticks([])
    # add x, y label to plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_new,tanh_smooth, label='tanh')
    plt.plot(x_new,sig_smooth, label='sigmoid')
    # show legend
    plt.legend()
    plt.show()

#Function chooser
func_arg = {"-q1_1": q1_1, "-q1_2": q1_2, "-q5": q5}
if __name__ == "__main__":
    func_arg[sys.argv[1]]()
