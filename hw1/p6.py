import sys
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import expit

# Data sets
training_set = "./data/digitstrain.txt"
validation_set = "./data/digitsvalid.txt"
test_set = "./data/digitstest.txt"
# Tunable parameters
epochs = 800     # 200
eta = 0.001   # learning rate
momentum = 0
reg_lambda = 0.0001 # regularization strength
layer_size = {'1': 400, '2': 100, 'output':10}     # number of hidden units
batch_size = 32     # batch size for batch normalization
# Parameter dictionaries
weights = {}
best_weights = {}
biases = {}


def a():
    """
        Train the model
        get the training error, validation error and test error
    """
    # Load Training Data (3000, 785)
    x_train, y_train = load_data(training_set)     # (3000, 784), (3000, 1)
    # Load Validation Data (1000, 785)
    x_valid, y_valid = load_data(validation_set)    # (1000, 784), (1000, 1)
    # Load Test Data (3000, 785)
    x_test, y_test = load_data(test_set)            # (3000, 784), (3000, 1)
    min_valid_error = sys.maxint
    # Get number of examples
    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]
    num_test_example = x_test.shape[0]

    layer_size['0'] = x_train.shape[1]
    # Initialize parameters(a dictionary holds all the weightss)
    weights['1'], biases['1'] = init_params('1', layer_size['0'], layer_size['1'])   # (784, 100), (100, 1)
    weights['2'], biases['2'] = init_params('2' ,layer_size['1'], layer_size['output'])   # (100, 10), (10, 1)
    w1_prev_gradient = np.zeros(weights['1'].shape)
    w2_prev_gradient = np.zeros(weights['2'].shape)
    b1_prev_gradient = np.zeros(biases['1'].shape)
    b2_prev_gradient = np.zeros(biases['2'].shape)
    # Creat lists for containing the errors
    training_error_list = []
    valid_error_list = []
    test_error_list = []
    # Run epochs times
    for e in range(epochs):
        training_error = 0      # training cross entropy
        valid_error = 0         # valid cross entropy
        test_error = 0          # test error
        ''' Training '''
        for i in range(num_training_example):
            x = x_train[i, :].reshape(len(x_train[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['output'], 1))
            label = int(y_train[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of last layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (10, 1)
            o = softmax(a2)     # (10, 1)
            training_error += cross_entropy(o, y)
            # Update weights['1']
            loss_over_h1 = np.dot(weights['2'], softmax_derivative(o, y))   # (100, 1)
            loss_over_a1 = np.multiply(loss_over_h1, sigmoid_derivative(a1))    #(100, 1)
            # w1 gradient, weight decay
            w1_curr_gradient = np.dot(x, np.transpose(loss_over_a1))
            w1_gradient = get_gradient(w1_curr_gradient, w1_prev_gradient, \
                                    momentum, reg_lambda, weights['1'])
            # b1 gradient, no bias decay
            b1_curr_gradient = loss_over_a1
            b1_gradient = get_gradient(b1_curr_gradient, b1_prev_gradient, \
                                    momentum, 0, biases['1'])
            sgd(w1_gradient, b1_gradient, '1', eta)
            # Update weights['2']
            loss_over_a2 = np.transpose(softmax_derivative(o, y))
            # w2 gradient
            w2_curr_gradient = np.dot(h1, loss_over_a2)   # 100*10
            w2_gradient = get_gradient(w2_curr_gradient, w2_prev_gradient,\
                                    momentum, reg_lambda, weights['2'])
            # b2 gradient
            b2_curr_gradient = softmax_derivative(o, y)
            b2_gradient = get_gradient(b2_curr_gradient, b2_prev_gradient, \
                                    momentum, 0, biases['2'])

            sgd(w2_gradient, b2_gradient, '2', eta)
            # update gradient parameters
            w1_prev_gradient = w1_gradient
            b1_prev_gradient = b1_gradient

            w2_prev_gradient = w2_gradient
            b2_prev_gradient = b2_gradient

        ''' Validation '''
        for i in range(num_valid_example):
            x = x_valid[i, :].reshape(len(x_valid[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['output'], 1))
            label = int(y_valid[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of last layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (10, 1)
            o = softmax(a2)     # (10, 1)
            valid_error += cross_entropy(o, y)

        ''' Test '''
        for i in range(num_test_example):
            x = x_test[i, :].reshape(len(x_test[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['output'], 1))
            label = int(y_test[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of last layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (10, 1)
            o = softmax(a2)     # (10, 1)
            test_error += cross_entropy(o, y)

        # Add regularizer
        training_error_avg = training_error / num_training_example
        training_error_avg = get_l2_loss(training_error_avg, weights, reg_lambda)

        valid_error_avg = valid_error / num_valid_example
        test_error_avg = test_error / num_test_example

        if valid_error_avg < min_valid_error:
            min_valid_error = valid_error_avg
            best_weights = deepcopy(weights)
        # elif sys.argv[1] == '-c':
        #     break

        training_error_list.append(training_error_avg)
        valid_error_list.append(valid_error_avg)
        test_error_list.append(test_error_avg)
        print "##### Epoch %s training_error = %s, valid_error = %s, test_error = %s" % \
            (e + 1, training_error_avg, valid_error_avg, test_error_avg)
    # Plot the figures
    if sys.argv[1] == '-a':
        plt.xlabel("# epochs")
        plt.ylabel("error")
        plt.plot(training_error_list, label='training error')
        plt.plot(valid_error_list, label='valid error')
        plt.plot(test_error_list, label='test error')
        plt.title('Cross Entropy (learning rate = %s, momentum = %s, layer_size = %s)'\
                % (eta, momentum, layer_size['1']))
        plt.legend()
        plt.show()
    elif sys.argv[1] == '-c':
        return best_weights

def b():
    """
        Get the classification error
    """
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

    layer_size['0'] = x_train.shape[1]
    # Initialize weights(a dictionary holds all the weightss)
    weights['1'], biases['1'] = init_params('1', layer_size['0'], layer_size['1'])   # (784, 100), (100, 1)
    weights['2'], biases['2'] = init_params('2' ,layer_size['1'], layer_size['output'])   # (100, 10), (10, 1)
    w1_prev_gradient = np.zeros(weights['1'].shape)
    w2_prev_gradient = np.zeros(weights['2'].shape)
    b1_prev_gradient = np.zeros(biases['1'].shape)
    b2_prev_gradient = np.zeros(biases['2'].shape)
    # Creat lists for containing the errors
    training_error_list = []
    valid_error_list = []
    test_error_list = []
    # Run epochs times
    for e in range(epochs):
        training_classify_error = 0     # training classification error
        valid_classify_error = 0        # valid classification error
        test_classify_error = 0          # test classification error
        ''' Training '''
        for i in range(num_training_example):
            x = x_train[i, :].reshape(len(x_train[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['output'], 1))
            label = int(y_train[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of last layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (10, 1)
            o = softmax(a2)     # (10, 1)
            training_classify_error += classification_error(o, label)
            # Update weights['1']
            loss_over_h1 = np.dot(weights['2'], softmax_derivative(o, y))   # (100, 1)
            loss_over_a1 = np.multiply(loss_over_h1, sigmoid_derivative(a1))    #(100, 1)
            # w1 gradient
            w1_curr_gradient = np.dot(x, np.transpose(loss_over_a1))
            w1_gradient = get_gradient(w1_curr_gradient, w1_prev_gradient, \
                                    momentum, reg_lambda, weights['1'])
            # b1 gradient
            b1_curr_gradient = loss_over_a1
            b1_gradient = get_gradient(b1_curr_gradient, b1_prev_gradient, \
                                    momentum, 0, biases['1'])
            sgd(w1_gradient, b1_gradient, '1', eta)
            # Update weights['2']
            loss_over_a2 = np.transpose(softmax_derivative(o, y))
            # w2 gradient
            w2_curr_gradient = np.dot(h1, loss_over_a2)   # 100*10
            w2_gradient = get_gradient(w2_curr_gradient, w2_prev_gradient,\
                                    momentum, reg_lambda, weights['2'])
            # b2 gradient
            b2_curr_gradient = softmax_derivative(o, y)
            b2_gradient = get_gradient(b2_curr_gradient, b2_prev_gradient, \
                                    momentum, 0, biases['2'])

            sgd(w2_gradient, b2_gradient, '2', eta)
            # update gradient parameters
            w1_prev_gradient = w1_gradient
            b1_prev_gradient = b1_gradient

            w2_prev_gradient = w2_gradient
            b2_prev_gradient = b2_gradient

        ''' Validation '''
        for i in range(num_valid_example):
            x = x_valid[i, :].reshape(len(x_valid[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['output'], 1))
            label = int(y_valid[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of last layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (10, 1)
            o = softmax(a2)     # (10, 1)
            valid_classify_error += classification_error(o, label)

        ''' Test '''
        for i in range(num_test_example):
            x = x_test[i, :].reshape(len(x_test[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['output'], 1))
            label = int(y_test[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of last layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (10, 1)
            o = softmax(a2)     # (10, 1)
            test_classify_error += classification_error(o, label)

        # Add the errors into lists
        training_error_avg = float(training_classify_error) / num_training_example * 100
        valid_error_avg = float(valid_classify_error) / num_valid_example * 100
        test_error_avg = float(test_classify_error) / num_test_example * 100

        training_error_list.append(training_error_avg)
        valid_error_list.append(valid_error_avg)
        test_error_list.append(test_error_avg)
        print "##### Epoch %s training_classify_error = %s%%, valid_classify_error = %s%%, test_error = %s%%" % \
            (e + 1, training_error_avg, valid_error_avg, test_error_avg)
    # Plot the figures
    plt.xlabel("# epochs")
    plt.ylabel("error(%)")
    plt.plot(training_error_list, label='training classification error')
    plt.plot(valid_error_list, label='valid classification error')
    plt.plot(test_error_list, label='test classification error')
    plt.title('Classification Error (learning rate = %s, momentum = %s, layer_size = %s)'\
            % (eta, momentum, layer_size['1']))
    plt.legend()
    plt.show()

# Visualizing parameters
def c():
    ''' Visualization '''
    best_weights = a()
    fig, axs = plt.subplots(10, 10)
    # Remove horizontal space between axes
    fig.subplots_adjust(wspace=0, hspace=0)
    count = 1
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, count)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(best_weights['1'][:, count - 1].reshape(28, 28), cmap='gray', origin='lower')
            count += 1
    plt.show()

def g():
    '''
        Two layer neural net
    '''
    # Load Training Data (3000, 785)
    x_train, y_train = load_data(training_set)     # (3000, 784), (3000, 1)
    # Load Validation Data (1000, 785)
    x_valid, y_valid = load_data(validation_set)    # (1000, 784), (1000, 1)
    # Load Test Data (3000, 785)
    x_test, y_test = load_data(test_set)            # (3000, 784), (3000, 1)
    min_valid_error = sys.maxint
    # Get number of examples
    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]
    num_test_example = x_test.shape[0]

    layer_size['0'] = x_train.shape[1]
    # Initialize weights(a dictionary holds all the weightss)
    weights['1'], biases['1'] = init_params('1', layer_size['0'], layer_size['1'])   # (784, 100), (100, 1)
    weights['2'], biases['2'] = init_params('2' ,layer_size['1'], layer_size['2'])   # (100, 100), (100, 1)
    weights['3'], biases['3'] = init_params('3' ,layer_size['2'], layer_size['output'])   # (100, 10), (10, 1)

    w1_prev_gradient = np.zeros(weights['1'].shape)
    w2_prev_gradient = np.zeros(weights['2'].shape)
    w3_prev_gradient = np.zeros(weights['3'].shape)

    b1_prev_gradient = np.zeros(biases['1'].shape)
    b2_prev_gradient = np.zeros(biases['2'].shape)
    b3_prev_gradient = np.zeros(biases['3'].shape)

    # Creat lists for containing the cross entropy errors
    training_error_list = []
    valid_error_list = []
    test_error_list = []

    # Creat lists for containing the classification errors
    training_classify_error_list = []
    valid_classify_error_list = []
    test_classify_error_list = []

    # Run epochs times
    for e in range(epochs):
        training_error = 0      # training cross entropy
        valid_error = 0         # valid cross entropy
        test_error = 0          # test error
        training_classify_error = 0     # training classification error
        valid_classify_error = 0        # valid classification error
        test_classify_error = 0          # test classification error
        ''' Training '''
        for i in range(num_training_example):
            x = x_train[i, :].reshape(len(x_train[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['output'], 1))
            label = int(y_train[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the 1st hidden layer, input of the 2nd hidden layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (100, 1)
            h2 = sigmoid(a2)  # Output of the 2nd hidden layer, input of the last layer
            a3 = feedforward(weights['3'], h2, biases['3']) # (10, 1)
            o = softmax(a3)     # (10, 1)
            training_error += cross_entropy(o, y)
            training_classify_error += classification_error(o, label)

            ''' Backprops (compute the gradients) '''
            # w3 gradient
            loss_over_a3 = np.transpose(softmax_derivative(o, y))   # (1, 10)
            w3_curr_gradient = np.dot(h2, loss_over_a3)   # 100*10
            w3_gradient = get_gradient(w3_curr_gradient, w3_prev_gradient,\
                                    momentum, reg_lambda, weights['3'])
            # b3 gradient
            b3_curr_gradient = softmax_derivative(o, y)
            b3_gradient = get_gradient(b3_curr_gradient, b3_prev_gradient, \
                                    momentum, 0, biases['3'])

            # w2 gradient, weight decay
            loss_over_h2 = np.dot(weights['3'], softmax_derivative(o, y))   # (100, 1)
            loss_over_a2 = np.multiply(loss_over_h2, sigmoid_derivative(a2))    #(100, 1)

            w2_curr_gradient = np.dot(h1, np.transpose(loss_over_a2))
            w2_gradient = get_gradient(w2_curr_gradient, w2_prev_gradient, \
                                    momentum, reg_lambda, weights['2'])
            # b2 gradient, no bias decay
            b2_curr_gradient = loss_over_a2
            b2_gradient = get_gradient(b2_curr_gradient, b2_prev_gradient, \
                                    momentum, 0, biases['2'])
            # w1 gradient, weight decay
            loss_over_h1 = np.dot(weights['2'], loss_over_a2)   # (100, 1)
            loss_over_a1 = np.multiply(loss_over_h1, sigmoid_derivative(a1))    #(100, 1)

            w1_curr_gradient = np.dot(x, np.transpose(loss_over_a1))
            w1_gradient = get_gradient(w1_curr_gradient, w1_prev_gradient, \
                                    momentum, reg_lambda, weights['1'])
            # b1 gradient, no bias decay
            b1_curr_gradient = loss_over_a1
            b1_gradient = get_gradient(b1_curr_gradient, b1_prev_gradient, \
                                    momentum, 0, biases['1'])
            ''' SGD update parameters '''
            # sgd update parameters
            # Update weights['3']
            sgd(w3_gradient, b3_gradient, '3', eta)
            # Update weights['2']
            sgd(w2_gradient, b2_gradient, '2', eta)
            # Update weights['1']
            sgd(w1_gradient, b1_gradient, '1', eta)
            # update previous gradient parameters
            w1_prev_gradient = w1_gradient
            b1_prev_gradient = b1_gradient

            w2_prev_gradient = w2_gradient
            b2_prev_gradient = b2_gradient

            w3_prev_gradient = w3_gradient
            b3_prev_gradient = b3_gradient

        ''' Validation '''
        for i in range(num_valid_example):
            x = x_valid[i, :].reshape(len(x_valid[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['output'], 1))
            label = int(y_valid[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of the 2nd hidden layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (100, 1)
            h2 = sigmoid(a2)  # Output of the 2nd hidden layer, input of the last layer
            a3 = feedforward(weights['3'], h2, biases['3']) # (10, 1)
            o = softmax(a3)     # (10, 1)
            valid_error += cross_entropy(o, y)
            valid_classify_error += classification_error(o, label)

        ''' Test '''
        for i in range(num_test_example):
            x = x_test[i, :].reshape(len(x_test[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['output'], 1))
            label = int(y_test[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of last layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (100, 1)
            h2 = sigmoid(a2)  # Output of the 2nd hidden layer, input of the last layer
            a3 = feedforward(weights['3'], h2, biases['3']) # (10, 1)
            o = softmax(a3)     # (10, 1)
            test_error += cross_entropy(o, y)
            test_classify_error += classification_error(o, label)

        # Add regularizer cross entropy
        training_error_avg = training_error / num_training_example
        training_error_avg = get_l2_loss(training_error_avg, weights, reg_lambda)

        valid_error_avg = valid_error / num_valid_example
        test_error_avg = test_error / num_test_example

        # Add classification errors into lists
        training_classify_error_avg = float(training_classify_error) / num_training_example * 100
        valid_classify_error_avg = float(valid_classify_error) / num_valid_example * 100
        test_classify_error_avg = float(test_classify_error) / num_test_example * 100

        if valid_error_avg < min_valid_error:
            min_valid_error = valid_error_avg
            best_weights = deepcopy(weights)

        # cross entropy error lists
        training_error_list.append(training_error_avg)
        valid_error_list.append(valid_error_avg)
        test_error_list.append(test_error_avg)

        # classification error lists
        training_classify_error_list.append(training_classify_error_avg)
        valid_classify_error_list.append(valid_classify_error_avg)
        test_classify_error_list.append(test_classify_error_avg)

        print "##### Epoch %s epoch=%s, momentum=%s, eta=%s, lambda=%s, hidden1=%s, hidden2=%s #####\n training_error = %s, valid_error = %s, test_error = %s\n \
training_classify_error = %s%%, valid_classify_error = %s%%, test_error = %s%% #####\n\n \
               " % (e + 1, epochs, momentum, eta, reg_lambda, layer_size['1'],layer_size['2'], training_error_avg, valid_error_avg, test_error_avg, \
                    training_classify_error_avg, valid_classify_error_avg, test_classify_error_avg)

    ''' Visualization '''
    # Cross Entropy
    plt.figure(1)
    plt.xlabel("# epochs")
    plt.ylabel("error")
    plt.plot(training_error_list, label='training error')
    plt.plot(valid_error_list, label='valid error')
    plt.plot(test_error_list, label='test error')
    plt.title('Cross Entropy\n (learning rate = %s, momentum = %s, reg_lambda=%s, hidden 1 = %s, hidden 2 = %s)'\
            % (eta, momentum, reg_lambda, layer_size['1'], layer_size['2']))
    plt.legend()
    # Classification Error
    plt.figure(2)
    plt.xlabel("# epochs")
    plt.ylabel("error(%)")
    plt.plot(training_classify_error_list, label='training classification error')
    plt.plot(valid_classify_error_list, label='valid classification error')
    plt.plot(test_classify_error_list, label='test classification error')
    plt.title('Classification Error (learning rate = %s, momentum = %s, hidden 1 = %s, hidden 2 = %s)'\
            % (eta, momentum, layer_size['1'], layer_size['2']))
    plt.legend()
    # Best weights
    fig, axs = plt.subplots(10, 10)
    # Remove horizontal space between axes
    fig.subplots_adjust(wspace=0, hspace=0)
    count = 1
    for i in range(int(np.sqrt(layer_size['1']))):
        for j in range(int(np.sqrt(layer_size['1']))):
            plt.subplot(int(np.sqrt(layer_size['1'])), int(np.sqrt(layer_size['1'])), count)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(best_weights['1'][:, count - 1].reshape(28, 28), cmap='gray', origin='lower')
            count += 1

    plt.show()

def h():
    '''
        Two layer neural net with batch normalization
    '''
    # parameters of batch normalization
    gamma = {}
    beta = {}
    eps = 1e-5
    # Load Training Data (3000, 785)
    x_train, y_train = load_data(training_set)     # (3000, 784), (3000, 1)
    # Load Validation Data (1000, 785)
    x_valid, y_valid = load_data(validation_set)    # (1000, 784), (1000, 1)

    min_valid_error = sys.maxint
    # Get number of examples
    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]

    num_features = x_train.shape[1]

    layer_size['0'] = x_train.shape[1]
    # Initialize parameters (a dictionary holds all the weightss)
    weights['1'], biases['1'] = init_params('1', layer_size['0'], layer_size['1'])   # (784, 100), (100, 1)
    weights['2'], biases['2'] = init_params('2' ,layer_size['1'], layer_size['2'])   # (100, 100), (100, 1)
    weights['3'], biases['3'] = init_params('3' ,layer_size['2'], layer_size['output'])   # (100, 10), (10, 1)
    # batch normalization parameters
    gamma['1'] = np.ones((layer_size['1'],))
    gamma['2'] = np.ones((layer_size['2'],))
    beta['1'] = np.zeros((layer_size['1'],))
    beta['2'] = np.zeros((layer_size['2'],))

    w1_prev_gradient = np.zeros(weights['1'].shape)
    w2_prev_gradient = np.zeros(weights['2'].shape)
    w3_prev_gradient = np.zeros(weights['3'].shape)

    b1_prev_gradient = np.zeros(biases['1'].shape)
    b2_prev_gradient = np.zeros(biases['2'].shape)
    b3_prev_gradient = np.zeros(biases['3'].shape)

    gamma1_prev_gradient = np.zeros(gamma['1'].shape)
    gamma2_prev_gradient = np.zeros(gamma['2'].shape)

    beta1_prev_gradient = np.zeros(beta['1'].shape)
    beta2_prev_gradient = np.zeros(beta['2'].shape)

    # Creat lists for containing the cross entropy errors
    training_error_list = []
    valid_error_list = []

    # Run epochs times
    for e in range(epochs):
        training_error = 0      # training cross entropy
        valid_error = 0         # valid cross entropy
        i_train = 0
        i_valid = 0
        ''' Training '''
        while i_train < num_training_example:
            j_train = i_train + batch_size
            # get x, y
            if j_train < num_training_example:
                x = np.transpose(x_train[i_train:j_train, :])    # (784, 32)
                y = np.zeros((layer_size['output'], batch_size))    # (10, 32)
                # get the indicator function y
                for row in range(batch_size):
                    label = int(y_train[i_train+row,0])
                    y[label, row] = 1
            else:
                remaining_size = num_training_example - i_train
                x = np.transpose(x_train[i_train:num_training_example, :])  # (784, 24) 24+93*32 = 3000
                y = np.zeros((layer_size['output'], remaining_size))  # (10, 24)
                # get the indicator function y
                for row in range(remaining_size):
                    label = int(y_train[i_train+row,0])
                    y[label, row] = 1

            ''' Forward training process '''
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 32)
            b1, cache1 = batch_norm_forward(np.transpose(a1), gamma['1'], beta['1'], eps)    # b1 (32, 100)
            h1 = sigmoid(np.transpose(b1))  # (100, 32)
            a2 = feedforward(weights['2'], h1, biases['2']) # (100, 32)
            b2, cache2 = batch_norm_forward(np.transpose(a2), gamma['2'], beta['2'], eps)    # b2 (32, 100)
            h2 = sigmoid(np.transpose(b2))  # (100, 32)
            a3 = feedforward(weights['3'], h2, biases['3']) # (10, 32)
            o = softmax(a3)     # (10, 32)
            training_error += cross_entropy(o, y)

            ''' Backprops (compute the gradients) '''
            # w3 gradient
            d_softmax_output = softmax_derivative(o, y)     # (32, 10)
            loss_over_a3 = np.transpose(d_softmax_output)   # (32, 10)
            w3_curr_gradient = np.dot(h2, loss_over_a3)   # 100*10
            w3_gradient = get_gradient(w3_curr_gradient, w3_prev_gradient,\
                                    momentum, reg_lambda, weights['3'])
            # b3 gradient
            b3_curr_gradient = np.sum(d_softmax_output, axis=1).reshape(biases['3'].shape)    # (10, 1)
            b3_gradient = get_gradient(b3_curr_gradient, b3_prev_gradient, \
                                    momentum, 0, biases['3'])
            # w2 gradient, weight decay
            loss_over_h2 = np.dot(weights['3'], np.transpose(loss_over_a3))   # (100, 32)
            loss_over_b2 = np.multiply(loss_over_h2, np.transpose(sigmoid_derivative(b2)))    # (100, 32)
            b2_over_a2, b2_over_gamma2 = batch_norm_backward(b2, cache2)    # (32, 100)
            loss_over_a2 = np.multiply(np.transpose(loss_over_b2), b2_over_a2)  # (32, 100)
            # print "h1.shape = %s, b1.shape = %s, a2.shape = %s, b2 shape = %s, loss_over_a2 = %s" % (h1.shape, b1.shape, a2.shape, b2.shape, loss_over_a2.shape)

            w2_curr_gradient = np.dot(h1, loss_over_a2)     #(400, 100) (h2,h1)
            w2_gradient = get_gradient(w2_curr_gradient, w2_prev_gradient, \
                                    momentum, reg_lambda, weights['2'])
            # b2 gradient, no bias decay
            loss_over_a2 = np.multiply(loss_over_b2, np.transpose(b2_over_a2))  # (100, 32)
            b2_curr_gradient = np.sum(loss_over_a2, axis=1).reshape(biases['2'].shape)     # (100, 1)
            b2_gradient = get_gradient(b2_curr_gradient, b2_prev_gradient, \
                                    momentum, 0, biases['2'])

            # gamma2 gradient, add weight decay factor?
            loss_over_gamma2 = np.multiply(loss_over_b2, b2_over_gamma2)  # (100, 32)
            gamma2_curr_gradient = np.sum(loss_over_gamma2, axis=1)     # (100, 1)
            gamma2_gradient = get_gradient(gamma2_curr_gradient, gamma2_prev_gradient, \
                                    momentum, reg_lambda, gamma['2'])

            # beta2 gradient, no weight decay
            beta2_curr_gradient = np.sum(loss_over_b2, axis=1)  # (100, 1)
            beta2_gradient = get_gradient(beta2_curr_gradient, beta2_prev_gradient, \
                                    momentum, 0, beta['2'])

            loss_over_h1 = np.dot(weights['2'], loss_over_a2)   # (100, 32)
            loss_over_b1 = np.multiply(loss_over_h1, sigmoid_derivative(a1))    #(100, 32)
            b1_over_a1, b1_over_gamma1 = batch_norm_backward(b1, cache1)    # (100, 32), (100, 1)
            loss_over_a1 = np.multiply(loss_over_b1, np.transpose(b1_over_a1))    # (100, 32)

            # w1 gradient, weight decay
            w1_curr_gradient = np.dot(x, np.transpose(loss_over_a1))    # (784, 100)
            w1_gradient = get_gradient(w1_curr_gradient, w1_prev_gradient, \
                                    momentum, reg_lambda, weights['1'])

            # b1 gradient, no bias decay
            b1_curr_gradient = np.sum(loss_over_a1, axis=1).reshape(biases['1'].shape)
            b1_gradient = get_gradient(b1_curr_gradient, b1_prev_gradient, \
                                    momentum, 0, biases['1'])

            # gamma1 gradient
            loss_over_gamma1 = np.multiply(loss_over_b1, b1_over_gamma1)  # (100, 32)
            gamma1_curr_gradient = np.sum(loss_over_gamma1, axis=1)     # (100, 1)
            gamma1_gradient = get_gradient(gamma1_curr_gradient, gamma1_prev_gradient, \
                                    momentum, reg_lambda, gamma['1'])
            # beta1 gradient, no weight decay
            beta1_curr_gradient = np.sum(loss_over_b1, axis=1)  # (100, 1)
            beta1_gradient = get_gradient(beta1_curr_gradient, beta1_prev_gradient, \
                                    momentum, 0, beta['1'])

            ''' SGD update parameters '''
            # Update weights['3']
            sgd(w3_gradient, b3_gradient, '3', eta)
            # Update weights['2']
            sgd(w2_gradient, b2_gradient, '2', eta)
            # Update weights['1']
            sgd(w1_gradient, b1_gradient, '1', eta)\
            # Update parameters of batch normalization
            gamma['2'] -= eta * gamma2_gradient
            gamma['1'] -= eta * gamma1_gradient
            beta['2'] -= eta * beta2_gradient
            beta['1'] -= eta * beta1_gradient
            # update previous gradient parameters
            w1_prev_gradient = w1_gradient
            b1_prev_gradient = b1_gradient

            w2_prev_gradient = w2_gradient
            b2_prev_gradient = b2_gradient

            w3_prev_gradient = w3_gradient
            b3_prev_gradient = b3_gradient

            gamma2_prev_gradient = gamma2_gradient
            gamma1_prev_gradient = gamma1_gradient

            beta2_prev_gradient = beta2_gradient
            beta1_prev_gradient = beta1_gradient
            # update counter
            i_train = j_train
        ''' Validation '''
        while i_valid < num_valid_example:
            j_valid = i_valid + batch_size
            if j_valid < num_valid_example:
                x = np.transpose(x_valid[i_valid:j_valid, :])    # (784, 32)
                y = np.zeros((layer_size['output'], batch_size))    # (10, 32)
                # get the indicator function y
                for row in range(batch_size):
                    label = int(y_valid[i_valid+row,0])
                    y[label, row] = 1
            else:
                remaining_size = num_valid_example - i_valid
                x = np.transpose(x_valid[i_valid:num_valid_example, :])  # (784, 24) 24+93*32 = 3000
                y = np.zeros((layer_size['output'], remaining_size))  # (10, 24)
                # get the indicator function y
                for row in range(remaining_size):
                    label = int(y_valid[i_valid+row,0])
                    y[label, row] = 1

            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 32)
            b1, cache1 = batch_norm_forward(np.transpose(a1), gamma['1'], beta['1'], eps)    # b1 (32, 100)
            h1 = sigmoid(np.transpose(b1))  # (100, 32)
            a2 = feedforward(weights['2'], h1, biases['2']) # (100, 32)
            b2, cache2 = batch_norm_forward(np.transpose(a2), gamma['2'], beta['2'], eps)    # b2 (32, 100)
            h2 = sigmoid(np.transpose(b2))  # (100, 32)
            a3 = feedforward(weights['3'], h2, biases['3']) # (10, 32)
            o = softmax(a3)     # (10, 32)
            valid_error += cross_entropy(o, y)
            i_valid = j_valid

        valid_error_avg = valid_error / num_valid_example
        training_error_avg = training_error / num_training_example
        # cross entropy error lists
        training_error_list.append(training_error_avg)
        valid_error_list.append(valid_error_avg)
        print "##### Epoch %s ######\n epoch=%s, momentum=%s, eta=%s, lambda=%s, hidden1=%s, hidden2=%s\n training_error = %s, valid_error = %s\n" \
            % (e + 1, epochs, momentum, eta, reg_lambda, layer_size['1'],layer_size['2'], training_error_avg, valid_error_avg)
    ''' Visualization '''
    # Cross Entropy
    plt.xlabel("# epochs")
    plt.ylabel("error")
    plt.plot(training_error_list, label='training error')
    plt.plot(valid_error_list, label='valid error')
    plt.plot(test_error_list, label='test error')
    plt.title('Cross Entropy\n (learning rate = %s, momentum = %s, reg_lambda=%s, hidden 1 = %s, hidden 2 = %s)'\
            % (eta, momentum, reg_lambda, layer_size['1'], layer_size['2']))
    plt.legend()
    plt.show()

def batch_norm_forward(x, gamma, beta, eps):
    """
        Input:
            x: to be normalized inputs. (N, D): N: batch_size, D: hidden units
        gamma: (D,)
         beta: (D,)
    """

    N, D = x.shape
    # 1. mini-batch mean
    mu = 1./N * np.sum(x, axis = 0)
    # 2. subtract mean vector of every trainings example
    x_subtracted_mu = x - mu
    # 3. following the lower branch - calculation denominator
    x_subtracted_mu_squared = x_subtracted_mu ** 2
    # 4. mini-batch variance
    var = 1./N * np.sum(x_subtracted_mu_squared, axis = 0)
    # 5. get denominator of normalized x
    denominator = np.sqrt(var + eps)
    # 6. invert denominator
    inverted_denominator = 1./denominator
    # 7. Normalize x
    x_hat = x_subtracted_mu * inverted_denominator
    # 8. Get rx
    gamma_x_hat = gamma * x_hat
    # 9. Output
    out = gamma_x_hat + beta
    #store intermediate
    cache = (x_hat, gamma, x_subtracted_mu,\
            inverted_denominator,denominator,var,eps)

    return out, cache

def batch_norm_backward(y, cache):
    '''
        Get partial derivatives of dy/dx, dy/dr, dy/db
        Output:
            y_over_x: dy/dx (N, D)
        y_over_gamma: dy/dr (D, 1)
    '''
    (x_hat, gamma, x_subtracted_mu,\
        inverted_denominator,denominator,var,eps) = cache
    N,D = y.shape
    xmu_mean = -1./N * np.sum(x_subtracted_mu, axis = 0)
    x_hat_over_x = inverted_denominator + (x_subtracted_mu) * xmu_mean * np.power(inverted_denominator, 3)
    # dx
    y_over_x = gamma * x_hat_over_x
    # dgamma
    y_over_gamma = np.sum(x_hat, axis=0).reshape(D, 1)
    return y_over_x, y_over_gamma


def batch_norm_backward_input_derivative(dout, cache):
    # Get the variables stored in forward process
    (x_normalized, gamma, x_subtracted_mu,\
        inverted_denominator,denominator,var,eps) = cache
    # Get the dimensions of the input/output
    N,D = dout.shape
    # 9.
    d_beta = np.sum(dout, axis=0)
    d_gammax = dout #not necessary, but more understandable
    # 8.
    d_gamma = np.sum(dgammax * x_normalized, axis=0)
    d_x_normalized = dgammax * gamma
    # 7.
    d_inverted_denominator = np.sum(d_x_normalized * x_subtracted_mu, axis=0)
    dxmu1 = d_x_normalized * inverted_denominator
    # 6.
    d_denominator = -1. /(denominator ** 2) * d_inverted_denominator
    # 5.
    dvar = 0.5 * 1. /np.sqrt(var+eps) * d_denominator
    # 4.
    d_x_subtracted_mu_squared = 1. /N * np.ones((N,D)) * dvar
    # 3.
    dxmu2 = 2 * x_subtracted_mu * d_x_subtracted_mu_squared
    # 2.
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
    # 1.
    dx2 = 1. /N * np.ones((N,D)) * dmu
    # 0.
    dx = dx1 + dx2

    return dx, d_gamma, d_beta


def sgd(w_gradient, b_gradient, layer, eta):
    weights[layer] -= eta * w_gradient
    biases[layer] -= eta * b_gradient

# Compute gradient
def get_gradient(curr_gradient, prev_gradient, momentum, decay_factor, weights):
    return curr_gradient + momentum * prev_gradient + decay_factor * weights

# Feedforward
def feedforward(W, x, b):
    """
        Input:
            W: weight of this layer
            x: neuron inputs
        Output:
            a = b + np.dot(W.T, x)
    """
    return b + np.dot(np.transpose(W), x)

# Initialize weights
def init_params(layer, size_k_1, size_k):
    """
        Sample weights from uniform distribution
        as discussed in class.
        Input:
            size_k_1, size_k: sizes of layer k - 1 and k
        Ouput:
            weights: matrix of initialized weights: size_k_1 * size_k
            biases: biases terms of size_k
    """
    b = np.sqrt(6) / np.sqrt(size_k + size_k_1)
    weights = np.random.uniform(-b, b, (size_k_1, size_k))
    biases = np.zeros((size_k, 1))
    return weights, biases

# Classification Error
def classification_error(o, label):
    """
        If it is classified incorrectly, return 1.
        Or else return 0.
        Input:
            o: outpupt of the softmax layer
            label: the correct laybel
    """
    predicted_label = np.argmax(o)
    if predicted_label == label:
        return 0
    else:
        return 1


# Calculate cross entropy
def cross_entropy(o, y):
    """
        Input:
            o: output of the neural nets (vector, output of softmax function)
            y: desired output (number)
        Output:
            cross entropy of this example
    """
    bias = np.power(10, -10)
    return np.sum(np.nan_to_num(-y * np.log(o + bias) - (1-y) * np.log(1-o + bias)))

# loss function with regularization factor
def get_l2_loss(avg_loss, weights, reg_lambda):
    return avg_loss + \
        reg_lambda/2 * (np.sum(np.square(weights['1'])) + np.sum(np.square(weights['2'])))

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
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# Gradient of the softmax layer (output layer)
def softmax_derivative(f, y):
    """
        Input:
            f: output of the softmax layer
            y: indicator function(desired output)
        Output:
            partial derivative of softmax layer
    """
    return -(y - f)

def load_data(data_file):
    """
        Input: file to be converted
        Output: x, y numpy array (float)
    """
    data_array = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data_array)
    row = data_array.shape[0]
    col = data_array.shape[1]

    x = data_array[:,0:col - 1]
    y = data_array[:, col - 1].reshape(row, 1)
    # Test
    print "### Load Data File %s ###" % data_file
    print x.shape, y.shape
    return x, y


''' Test functions'''
# a = np.array([1, 2, 3])
# b = np.array([2, 3, 5, 6])
# print a
# print sigmoid(a)
# print softmax(a)
# print softmax(b)

# Function chooser
func_arg = {"-a": a, "-b": b, "-c": c, "-g": g, "-h": h}
if __name__ == "__main__":
    func_arg[sys.argv[1]]()
