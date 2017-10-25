import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


# Data sets
training_set = "../mnist_data/digitstrain.txt"
validation_set = "../mnist_data/digitsvalid.txt"
test_set = "../mnist_data/digitstest.txt"

# tunable parameters
cd_steps = 20    # run cd_steps iterations of Gibbs Sampling
num_hidden_units = 100  # number of hidden units
epochs = 40     # epochs for RBM training
lr = 0.005   # learning rate for RBM
mini_batch = 10

# parameters of normal distribution in weights initialization
mean = 0    # mean
stddev = 0.1    # standard deviation

# other global variables used
# best weights and biases from rbm
best_weights_rbm = []
best_hidbias_rbm = []
best_visbias_rbm = []
# best weights and biases from Autoencoder
best_weights_ae = []
best_hidbias_ae = []
best_visbias_ae = []

# parameters of neural network
layer_size = {'1': 100, '2':10}
def a():
    '''
        Basic Generalization
    '''
    global lr
    # Load Training Data (3000, 785)
    x_train, y_train = load_data(training_set)     # (3000, 784), (3000, 1)
    # Load Validation Data (1000, 785)
    x_valid, y_valid = load_data(validation_set)    # (1000, 784), (1000, 1)

    # Get number of examples
    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]

    num_input = x_train.shape[1]
    num_hidden = num_hidden_units

    weights, visbias, hidbias = \
            init_params(mean, stddev, num_input, num_hidden)

    train_recon_error_list = []
    valid_recon_error_list = []
    for e in range(epochs):
        eta = lr
        train_recon_error = 0
        valid_recon_error = 0
        i_train = 0
        i_valid = 0
        ''' Training '''
        while i_train < num_training_example:
            j_train = i_train + mini_batch
            size = mini_batch
            if j_train < num_training_example:
                x = np.transpose(x_train[i_train:j_train, :])    # (784, mini_batch)
                size = mini_batch
            else:
                remaining_size = num_training_example - i_train
                x = np.transpose(x_train[i_train:num_training_example, :])  # (784, remaining_size)
                size = remaining_size

            # positive phase
            h_probs = update_hidden(x, hidbias, weights)    # (hidden_units, mini_batch)
            pos_mean = np.dot(x, h_probs.T)    # (input, hidden_units)

            # negative phase
            x_tilde = gibbs_sampling(x, hidbias, visbias, cd_steps, weights)
            prob_h_given_xtilde = update_hidden(x_tilde, hidbias, weights)
            neg_mean = np.dot(x_tilde, prob_h_given_xtilde.T)

            # compute gradient
            weights += eta * (pos_mean - neg_mean)
            h_gradient = np.sum(h_probs - prob_h_given_xtilde,axis=1).reshape(hidbias.shape) / size
            hidbias += eta * h_gradient
            x_gradient = np.sum((x - x_tilde),axis=1).reshape(visbias.shape) / size
            visbias += eta * x_gradient

            # get cross entropy reconstruction error
            h_recon = update_hidden(x, hidbias, weights)
            x_recon = update_visible(h_recon, visbias, weights)

            train_recon_error += cross_entropy(x_recon, x)
            # update counter
            i_train = j_train

        ''' Validation '''
        while i_valid < num_valid_example:
            j_valid = i_valid + mini_batch
            if j_valid < num_valid_example:
                x = np.transpose(x_valid[i_valid:j_valid, :])    # (784, mini_batch)
            else:
                remaining_size = num_valid_example - i_valid
                x = np.transpose(x_valid[i_valid:num_valid_example, :])  # (784, 24) 24+93*32 = 3000

            # get cross entropy reconstruction error
            h_recon = update_hidden(x, hidbias, weights)
            x_recon = update_visible(h_recon, visbias, weights)

            valid_recon_error += cross_entropy(x_recon, x)
            i_valid = j_valid

        train_recon_error_avg = train_recon_error / num_training_example
        train_recon_error_list.append(train_recon_error_avg)

        valid_recon_error_avg = valid_recon_error / num_valid_example
        valid_recon_error_list.append(valid_recon_error_avg)
        print "##### Epoch %s ######\n \
            epoch=%s, eta=%s, hidden_units=%s, batch_size=%s, cd_steps=%s\n \
            training_error = %s valid_error=%s\n" \
            % (e + 1, epochs, eta, num_hidden_units, mini_batch, cd_steps, \
                train_recon_error_avg, valid_recon_error_avg)

    ''' Visualization '''
    if sys.argv[1] == '-a':
        # Cross Entropy
        plt.figure(1)
        plt.xlabel("# epochs")
        plt.ylabel("error")
        plt.plot(train_recon_error_list, label='training error')
        plt.plot(valid_recon_error_list, label='valid error')
        plt.title('Cross Entropy Reconstruction Error\n \
                (learning rate=%s, hidden_units=%s, batch_size=%s, cd_steps=%s)'\
                % (eta, num_hidden_units, mini_batch, cd_steps))
        plt.legend()

        # weights
        fig, axs = plt.subplots(10, 10)
        # Remove horizontal space between axes
        fig.subplots_adjust(wspace=0, hspace=0)
        count = 1
        num_pictures = int(np.sqrt(num_hidden_units))
        for i in range(num_pictures):
            for j in range(num_pictures):
                plt.subplot(num_pictures, num_pictures, count)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(weights[:, count - 1].reshape(28, 28), cmap='gray', clim=(-3,3), origin='lower')
                count += 1

        plt.show()
    elif sys.argv[1] == '-c' or sys.argv[1] == '-d':
        print "#### Problem %s ####" % (sys.argv[1])
        best_weights_rbm = weights
        best_hidbias_rbm = hidbias
        best_visbias_rbm = visbias
        return best_weights_rbm, best_hidbias_rbm, best_visbias_rbm

def c():
    """
        Sampling from the RBM Model
    """
    # Get the best_weights and best biases from a()
    best_weights_rbm, best_hidbias_rbm, best_visbias_rbm = a()     # (input,hidden),(hidden,1)
    steps = 1000
    num_samples = 100
    num_input = best_weights_rbm.shape[0]   # 784
    num_hidden = best_weights_rbm.shape[1]  # 100
    x_random = np.random.normal(mean, stddev, (num_input, num_samples))
    x_recon = gibbs_sampling(x_random, best_hidbias_rbm, best_visbias_rbm, steps, best_weights_rbm)

    ''' Visualization '''
    # x_recon
    fig, axs = plt.subplots(10, 10)
    # Remove horizontal space between axes
    fig.subplots_adjust(wspace=0, hspace=0)
    count = 1
    num_pictures = int(np.sqrt(num_samples))
    for i in range(num_pictures):
        for j in range(num_pictures):
            plt.subplot(num_pictures, num_pictures, count)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x_recon[:, count - 1].reshape(28, 28), cmap='gray', origin='lower')
            count += 1

    plt.show()


def e():
    """
        Autoencoder
        using tied weights
    """
    print "### Autoencoder ###"
    global lr
    # Load Training Data (3000, 785)
    x_train, y_train = load_data(training_set)     # (3000, 784), (3000, 1)
    # Load Validation Data (1000, 785)
    x_valid, y_valid = load_data(validation_set)    # (1000, 784), (1000, 1)

    # Get number of examples
    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]

    num_input = x_train.shape[1]
    num_hidden = num_hidden_units

    weights, visbias, hidbias = \
        init_params(mean, stddev, num_input, num_hidden)

    for e in range(epochs):
        train_recon_error = 0
        valid_recon_error = 0
        ''' Training '''
        for i in range(num_training_example):
            x = x_train[i, :].reshape(num_input, 1)    # (784, 1)
            # encode
            h = update_hidden(x, hidbias, weights)    # (hidden_units, 1) sigmoid
            # decode
            x_hat = update_visible(h, visbias, weights) # sigmoid

            loss_over_xhat = x_hat - x  # (784,1)
            xhat_over_ahat = sigmoid_derivative(x_hat)
            loss_over_ahat = np.multiply(loss_over_xhat, xhat_over_ahat)  # (784,1)

            a_over_h = weights    #(784, 100)
            loss_over_h = np.dot(a_over_h.T, loss_over_ahat)    # (100,1)
            h_over_a = sigmoid_derivative(h)
            loss_over_a = np.multiply(loss_over_h, h_over_a)    # (100, 1)

            ahat_over_w = h   # (100,1)
            w_gradient = np.dot(loss_over_ahat, ahat_over_w.T) / 2
            visbias_gradient = loss_over_ahat
            hidbias_gradient = loss_over_a

            # backprop
            weights -= lr * w_gradient  # (784,100)
            visbias -= lr * visbias_gradient    # (784,1)
            hidbias -= lr * hidbias_gradient    # (100, 1)

            train_recon_error += cross_entropy(x_hat, x)
        ''' Validation '''
        for i in range(num_valid_example):
            x = x_valid[i, :].reshape(num_input, 1)    # (784, 1)

            # get cross entropy reconstruction error
            h_probs = update_hidden(x, hidbias, weights)
            x_hat = update_visible(h_probs, visbias, weights)

            valid_recon_error += cross_entropy(x_hat, x)

        train_recon_error_avg = train_recon_error / num_training_example
        valid_recon_error_avg = valid_recon_error / num_valid_example
        print "##### Epoch %s ######\n \
            epoch=%s, eta=%s, hidden_units=%s\n training_error = %s valid_error=%s\n" \
            % (e + 1, epochs, lr, num_hidden_units, train_recon_error_avg, valid_recon_error_avg)

    ''' Visualization '''
    if sys.argv[1] == '-e':
        # weights
        fig, axs = plt.subplots(10, 10)
        # Remove horizontal space between axes
        fig.subplots_adjust(wspace=0, hspace=0)
        count = 1
        num_pictures = int(np.sqrt(num_hidden_units))
        for i in range(num_pictures):
            for j in range(num_pictures):
                plt.subplot(num_pictures, num_pictures, count)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(weights[:, count - 1].reshape(28, 28), cmap='gray', origin='lower')
                count += 1

        plt.show()
    elif sys.argv[1] == '-e2':
        print "### Problem e: get best weights ###"
        best_weights_ae = weights
        best_hidbias_ae = hidbias
        best_visbias_ae = visbias
        return best_weights_rbm, best_hidbias_rbm, best_visbias_rbm

def d():
    """
        Unsupervised Learning as Pretraining
    """
    global a
    global e
    nn_epochs = 100
    eta = 0.01
    # Get the best_weights and best biases from a() or d()
    best_weights = []
    best_hidbias = []
    best_visbias = []
    if sys.argv[1] == '-d':
        best_weights, best_hidbias, best_visbias = a()
    if sys.argv[1] == '-e2':
        best_weights, best_hidbias, best_visbias = e()
    # Load Training Data (3000, 785)
    x_train, y_train = load_data(training_set)     # (3000, 784), (3000, 1)
    # Load Validation Data (1000, 785)
    x_valid, y_valid = load_data(validation_set)    # (1000, 784), (1000, 1)
    # Get number of examples
    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]

    layer_size['0'] = x_train.shape[1]
    # dictionary for weights and biases
    weights = {}
    biases = {}

    # initialize weights and biases
    weights['1'] = best_weights
    biases['1'] = best_hidbias
    # random initialize second layer parameters
    b = np.sqrt(6) / np.sqrt(layer_size['2'] + layer_size['1'])
    weights['2'] = np.random.uniform(-b, b, (layer_size['1'], layer_size['2']))
    biases['2'] = np.zeros((layer_size['2'], 1))
    # Creat lists for containing the errors
    training_classify_error_list = []
    valid_classify_error_list = []

    for e in range(nn_epochs):
        training_classify_error = 0     # training classification error
        valid_classify_error = 0        # valid classification error
        ''' Training Part '''
        for i in range(num_training_example):
            x = x_train[i, :].reshape(len(x_train[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['2'], 1))
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
            w1_gradient = np.dot(x, np.transpose(loss_over_a1))
            b1_gradient = loss_over_a1
            sgd(weights, biases, w1_gradient, b1_gradient, '1', eta)
            # Update weights['2']
            loss_over_a2 = np.transpose(softmax_derivative(o, y))
            w2_gradient = np.dot(h1, loss_over_a2)   # 100*10
            b2_gradient = softmax_derivative(o, y)
            sgd(weights, biases, w2_gradient, b2_gradient, '2', eta)

        ''' Validation Part '''
        for i in range(num_valid_example):
            x = x_valid[i, :].reshape(len(x_valid[i, :]), 1)    # (784, 1)
            y = np.zeros((layer_size['2'], 1))
            label = int(y_valid[i,0])
            y[label] = 1
            a1 = feedforward(weights['1'], x, biases['1'])  # (100, 1)
            h1 = sigmoid(a1)  # Output of the hidden layer, input of last layer
            a2 = feedforward(weights['2'], h1, biases['2']) # (10, 1)
            o = softmax(a2)     # (10, 1)
            valid_classify_error += classification_error(o, label)

        # Add the errors into lists
        training_classify_error_avg = float(training_classify_error) / num_training_example * 100
        valid_classify_error_avg = float(valid_classify_error) / num_valid_example * 100

        training_classify_error_list.append(training_classify_error_avg)
        valid_classify_error_list.append(valid_classify_error_avg)
        print "##### Epoch %s training_classify_error = %s%%, valid_classify_error = %s%%" % \
            (e + 1, training_classify_error_avg, valid_classify_error_avg)
    # Plot the figures
    plt.xlabel("# epochs")
    plt.ylabel("error(%)")
    plt.plot(training_classify_error_list, label='training classification error')
    plt.plot(valid_classify_error_list, label='valid classification error')
    plt.legend()
    plt.show()


def sgd(weights, biases, w_gradient, b_gradient, layer, eta):
    # print "##### layer = %s, w_gradient = %s, b_gradient = %s ########" % (layer, w_gradient[0, :], b_gradient[0, :])
    weights[layer] -= eta * w_gradient
    biases[layer] -= eta * b_gradient



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

# Calculate cross entropy
def cross_entropy(o, y):
    """
        Input:
            o: output of the model
            y: desired output (original input)
        Output:
            cross entropy of this example
    """
    bias = np.exp(-20)
    return np.sum(np.nan_to_num(-y * np.log(o + bias) - (1-y) * np.log(1-o + bias)))


def gibbs_sampling(vis, hidbias, visbias, steps, weights):
    '''
        Perform steps iterations of Gibbs Sampling
        Output:
            h_tilde: sampled hidden units binary values
            x_tilde: sampled x~ binary values
    '''
    # calculate p(h|x~)
    x_tilde = vis
    num_input = vis.shape[0]
    prob_h_given_xtilde = []
    # Constrastive Divergence steps
    for i in range(steps):
        prob_h_given_xtilde = update_hidden(x_tilde, hidbias, weights)
        # sample h~ from the probs above (binomial distribution)
        h_tilde = get_binary_values(prob_h_given_xtilde)
        # calculate p(x~|h)
        prob_x_given_h = update_visible(h_tilde, visbias, weights)
        # sample x~ from the probs above (binomial distribution)
        x_tilde = get_binary_values(prob_x_given_h)

    return x_tilde

def get_binary_values(probs):
    samples = np.random.uniform(size=probs.shape)
    probs[samples < probs] = 1.
    # set any other non-one values to 0
    np.floor(probs, probs)
    return probs


def update_visible(hid, visbias, weights):
    '''
        Update visible units
        output sigmoid values
    '''
    vis = np.dot(weights, hid)      # (input, 1)
    vis += visbias
    vis = sigmoid(vis)

    return vis

def update_hidden(vis, hidbias, weights):
    '''
        Update hidden units
        output the sigmoid values (hidden_units, mini_batch)
    '''
    hid = np.dot(weights.T, vis)    # (hidden_units, mini_batch)
    hid += hidbias
    hid = sigmoid(hid)

    return hid

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

def softmax_derivative(f, y):
    """
        Gradient of the softmax layer (output layer)
        Input:
            f: output of the softmax layer
            y: indicator function(desired output)
        Output:
            partial derivative of softmax layer
    """
    return -(y - f)

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

# Initialize weights
def init_params(mean, stddev, size_k_1, size_k):
    """
        Sample weights from normal distribution with mean 0 stddev 0.1
        Input:
            mean: the mean of the normal distribution (default 0)
            stddev: the standard deviation of normal distribution (default 0.1)
            size_k_1, size_k: sizes of layer k-1(input) and k(hidden)
        Ouput:
            weights: matrix of initialized weights: size_k_1 * size_k
            visbias: bias for input layer, (size_k_1, 1)
            hidbias: bias for hidden layer, (size_k, 1)
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

func_arg = {"-a": a, "-c":c, "-d":d, "-e":e, "-e2":d}
if __name__ == "__main__":
    func_arg[sys.argv[1]]()
