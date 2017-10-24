import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


# Data sets
training_set = "../mnist_data/digitstrain.txt"
validation_set = "../mnist_data/digitsvalid.txt"
test_set = "../mnist_data/digitstest.txt"

# tunable parameters
cd_steps = 1    # run cd_steps iterations of Gibbs Sampling
num_hidden_units = 100  # number of hidden units
epochs = 1000
lr = 0.01   # learning rate
mini_batch = 10

# parameters of normal distribution in weights initialization
mean = 0    # mean
stddev = 0.1    # standard deviation

# other global variables used
best_weights = []
best_biases = []

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
            if j_train < num_training_example:
                x = np.transpose(x_train[i_train:j_train, :])    # (784, mini_batch)
            else:
                remaining_size = num_training_example - i_train
                x = np.transpose(x_train[i_train:num_training_example, :])  # (784, remaining_size)

            # positive phase
            h_probs = update_hidden(x, hidbias, weights)    # (hidden_units, mini_batch)
            h = get_binary_values(h_probs)
            pos_mean = np.dot(x, h.T)    # (input, hidden_units)

            # negative phase
            h_tilde, x_tilde = gibbs_sampling(x, hidbias, visbias, cd_steps, weights)
            neg_mean = np.dot(x_tilde, h_tilde.T)

            # compute gradient
            weights += eta * (pos_mean - neg_mean)
            h_gradient = np.sum(h - h_tilde,axis=1).reshape(hidbias.shape) / mini_batch
            hidbias += eta * h_gradient
            x_gradient = np.sum((x - x_tilde),axis=1).reshape(visbias.shape) / mini_batch
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
            plt.imshow(weights[:, count - 1].reshape(28, 28), cmap='gray', clim=(-3, 3), origin='lower')
            count += 1

    plt.show()

def d():
    """
        Unsupervised Learning as Pretraining
    """

    # Load Training Data (3000, 785)
    x_train, y_train = load_data(training_set)     # (3000, 784), (3000, 1)
    # Load Validation Data (1000, 785)
    x_valid, y_valid = load_data(validation_set)    # (1000, 784), (1000, 1)
    # Get number of examples
    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]
    # dictionary for weights and biases
    layer_size = {'1': 100, '2':10}
    layer_size['0'] = x_train.shape[1]

    weights = {}
    biases = {}

    # TODO: initialize weights and biases
    weights['1'] = best_weights
    biases['1'] = best_biases
    # Creat lists for containing the errors
    training_classify_error_list = []
    valid_classify_error_list = []

    for e in range(epochs):
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
            sgd(w1_gradient, b1_gradient, '1', eta)
            # Update weights['2']
            loss_over_a2 = np.transpose(softmax_derivative(o, y))
            w2_gradient = np.dot(h1, loss_over_a2)   # 100*10
            b2_gradient = softmax_derivative(o, y)
            sgd(w2_gradient, b2_gradient, '2', eta)

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
        training_classify_error_avg = float(training_classify_error) / num_training_example
        valid_classify_error_avg = float(valid_classify_error) / num_valid_example

        training_classify_error_list.append(training_classify_error_avg)
        valid_classify_error_list.append(valid_classify_error_avg)
        print "##### Epoch %s training_classify_error = %s, valid_classify_error = %s" % \
            (e + 1, training_classify_error_avg, valid_classify_error_avg)
    # TODO: Plot the figures
    plt.xlabel("# epochs")
    plt.ylabel("error")
    plt.plot(training_classify_error_list, label='training classification error')
    plt.plot(valid_classify_error_list, label='valid classification error')
    plt.legend()
    plt.show()


def sgd(w_gradient, b_gradient, layer, eta):
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
    # Constrastive Divergence steps
    for i in range(steps):
        prob_h_given_x = update_hidden(x_tilde, hidbias, weights)
        # sample h~ from the probs above (binomial distribution)
        h_tilde = get_binary_values(prob_h_given_x)
        # calculate p(x~|h)
        prob_x_given_h = update_visible(h_tilde, visbias, weights)
        # sample x~ from the probs above (binomial distribution)
        x_tilde = get_binary_values(prob_x_given_h)

    return h_tilde, x_tilde

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

func_arg = {"-a": a}
if __name__ == "__main__":
    func_arg[sys.argv[1]]()
