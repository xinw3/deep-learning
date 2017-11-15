import sys
import numpy as np
from copy import deepcopy


# tunable parameters
epochs = 100
eta = 0.1
num_dim = 16
num_hid = 128
batch_size = 512

train_file = 'train_ngram.txt'
val_file = 'val_ngram.txt'

voc_file_name = 'output8000'
voc_size = 8000
N = 4   # n-grams

def p32():

    val_total_words = 0
    val_total_words = get_total_words()
    print "total words in val %s" % (val_total_words)
    ''' Load Data '''
    # process input
    x_train, y_train = load_data(train_file)    # (81180, 3), (81180, 1)
    # Load Validation Data
    x_valid, y_valid = load_data(val_file)    # (10031, 3), (10031, 1)

    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]
    n = x_train.shape[1]      # n = N - 1

    weights = {}
    biases = {}
    ''' Initialization Parameters '''
    # initializing weights
    weights[0] = init_weights(voc_size, num_dim)    # word_embedding_weights
    weights[1] = init_weights(n * num_dim, num_hid) # embed_hid_weights
    weights[2] = init_weights(num_hid, voc_size)    # hid_output_weights

    # initialize biases
    biases[1] = np.zeros((1, num_hid))
    biases[2] = np.zeros((1, voc_size))

    # Creat lists for containing the cross entropy errors
    training_error_list = []
    valid_error_list = []
    val_ppl_list = []

    for e in range(epochs):
        training_error = 0
        valid_error = 0
        i_train = 0
        i_valid = 0
        val_perplexity = 0
        ''' Traninig '''
        while i_train < num_training_example:
            j_train = i_train + batch_size
            x_indices = np.zeros((batch_size, n))
            if j_train < num_training_example:
                x = np.zeros((batch_size, n * num_dim))
                y = np.zeros((batch_size, voc_size))
                x_indices = x_train[i_train:j_train,:]
                for i in range(batch_size):
                    temp = weights[0][x_indices[i,:],:]
                    x[i,:] = temp.flatten()
                    y[i,y_train[i + i_train]] = 1
            else:
                remaining_size = num_training_example - i_train
                x = np.zeros((remaining_size, n * num_dim))
                y = np.zeros((remaining_size, voc_size))
                x_indices = x_train[i_train:num_training_example, :]
                for i in range(remaining_size):
                    temp = weights[0][x_indices[i,:],:]
                    x[i,:] = temp.flatten()
                    y[i,y_train[i + i_train]] = 1

            ''' Feed Forward '''
            o1 = feedforward(x, weights[1], biases[1])
            a1 = tanh(o1)

            o2 = feedforward(a1, weights[2], biases[2])
            a2 = softmax(o2)

            training_error += cross_entropy(a2, y)

            ''' Backprop '''
            dl_do2 = _deriv_crossEnt_softmax(a2, y)
            dl_db2 = np.sum(dl_do2, axis=0).reshape(biases[2].shape) / batch_size
            dl_dW2 = np.dot(a1.T, dl_do2) / batch_size
            da1_do1 = _deriv_tanh(a1)
            dl_da1 = np.dot(dl_do2, weights[2].T)
            dl_do1 = np.multiply(da1_do1, dl_da1)
            dl_db1 = np.sum(dl_do1, axis=0).reshape(biases[1].shape) / batch_size
            dl_dW1 = np.dot(x.T, dl_do1) / batch_size

            # NOTE: W0 gradients, to the corresponding indices in x_indices
            dl_dx = np.dot(dl_do1, weights[1].T)

            ''' SGD Update '''
            weights[2] = sgd(weights[2], dl_dW2, eta)
            weights[1] = sgd(weights[1], dl_dW1, eta)
            # weights[0] updates
            actual_batch = x_indices.shape[0]
            for i in range(actual_batch):
                for j in range(n):
                    row = x_indices[i][j]
                    weights[0][row] = \
                        sgd(weights[0][row], dl_dx[i][j*num_dim : (j+1)*num_dim], eta)

            biases[2] = sgd(biases[2], dl_db2, eta)
            biases[1] = sgd(biases[1], dl_db1, eta)

            i_train = j_train

        ''' Validation '''
        while i_valid < num_valid_example:
            j_valid = i_valid + batch_size
            x_indices = np.zeros((batch_size, n))
            if j_valid < num_valid_example:
                x = np.zeros((batch_size, n * num_dim))
                y = np.zeros((batch_size, voc_size))
                x_indices = x_valid[i_valid:j_valid,:]
                for i in range(batch_size):
                    temp = weights[0][x_indices[i,:],:]
                    x[i,:] = temp.flatten()
                    y[i,y_valid[i + i_valid]] = 1
            else:
                remaining_size = num_valid_example - i_valid
                x = np.zeros((remaining_size, n * num_dim))
                y = np.zeros((remaining_size, voc_size))
                x_indices = x_valid[i_valid:num_valid_example, :]
                for i in range(remaining_size):
                    temp = weights[0][x_indices[i,:],:]
                    x[i,:] = temp.flatten()
                    y[i,y_valid[i + i_valid]] = 1

            ''' Feed Forward '''
            o1 = feedforward(x, weights[1], biases[1])
            a1 = tanh(o1)

            o2 = feedforward(a1, weights[2], biases[2])
            a2 = softmax(o2)

            # get perplexity
            val_perplexity += get_perplexity(val_total_words, a2, y)
            valid_error += cross_entropy(a2, y)
            i_valid = j_valid

        training_error_avg = training_error / num_training_example
        valid_error_avg = valid_error / num_valid_example
        val_ppl_avg = val_perplexity

        # cross entropy error lists
        training_error_list.append(training_error_avg)
        valid_error_list.append(valid_error_avg)
        val_ppl_list.append(val_ppl_avg)

        print "##### Epoch %s ######\n \
eta=%s, hidden=%s, batch_size=%s \n \
training_error = %s, valid_error = %s, perplexity=%s\n" \
            % (e + 1, eta, num_hid, batch_size, training_error_avg, valid_error_avg, val_ppl_avg)

    ''' Visualization '''
    # Cross Entropy
    plt.xlabel("# epochs")
    plt.ylabel("error")
    plt.plot(training_error_list, label='training error')
    plt.plot(valid_error_list, label='valid error')
    plt.title('Cross Entropy\n (learning rate=%s, hidden=%s)'\
            % (eta, num_hid))
    plt.legend()
    plt.show()

def get_perplexity(val_total_words, p, y):
    '''
        get the perplexity according to the input
    '''
    l = np.sum(y * np.log2(p)) / p.shape[0] / val_total_words
    ppl = np.power(2., -l)
    return ppl

def sgd(params, gradient, eta):
    params -= eta * gradient
    return params

def _deriv_tanh(x):
    '''
        input: x is the output of tanh layer
               i.e. x = tanh(o)
    '''
    return 1 - x ** 2

def _deriv_crossEnt_softmax(f,y):
    """
        Input:
            f: output of the softmax layer
            y: indicator function(desired output)
        Output:
            partial derivative of softmax layer
    """
    return f - y

# Calculate cross entropy
def cross_entropy(o, y):
    """
        Input:
            o: output of the neural nets (vector, output of softmax function)
            y: desired output (number)
        Output:
            cross entropy of this example
    """
    bias = np.power(10., -10)
    return -np.sum(y * np.log(o + bias))

def softmax(x):
    """
        Input: an array
        Output: an array of softmax function of each element
    """
    x_copy = deepcopy(x)
    x_copy -= np.max(x_copy)
    return np.exp(x_copy) / np.sum(np.exp(x_copy))

def tanh(x):
    return np.tanh(x)

# Feedforward
def feedforward(x, W, b):
    """
        Input:
            W: weight of this layer
            x: neuron inputs
        Output:
            a = b + np.dot(x, W)
    """
    return b + np.dot(x, W)

def init_weights(n_in, n_out):
    '''
        Xavier initialization of weights
    '''
    a = np.sqrt(6. / (n_in + n_out))
    return a * np.random.uniform(-1., 1., (n_in, n_out))

def get_total_words():
    '''
        streaming through the validation file to get the total word counts

        input: linux pipeline
        output: word count in the file
    '''
    count = 0
    for line in sys.stdin:
        words = line.split()
        count += len(words)
        count += 2  # add START and END tags to the counts

    return count


def load_data(data_file):
    data_array = np.loadtxt(data_file, dtype='int32')
    np.random.shuffle(data_array)
    row = data_array.shape[0]
    col = data_array.shape[1]

    x = data_array[:,0:col - 1]
    y = data_array[:, col - 1].reshape(row, 1)

    # Test
    print "### Load Data File %s ###" % data_file
    print x.shape, y.shape
    return x, y

func_arg = {"-p32": p32}
if __name__ == "__main__":
    func_arg[sys.argv[1]]()
