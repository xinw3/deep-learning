import sys
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import random

# tunable parameters
epochs = 1
eta = 0.1
num_dim = 2
num_hid = 128
batch_size = 512

voc_file_name = 'output8000'
voc_size = 8000
N = 4   # n-grams

train_file = 'train_ngram_indices.txt'
val_file = 'val_ngram_indices.txt'

# where best params stored
weights_file = 'weights_p35.pickle'
biases_file = 'biases_p35.pickle'

def get_best_params(epochs):
    """
        Train the model and get best parameters
        The best parameters are written to files
    """

    ''' Load Data '''
    print 'training to get the best params...'
    # process input
    x_train, y_train = load_data(train_file)    # (81180, 3), (81180, 1)
    x_valid, y_valid = load_data(val_file)    # (10031, 3), (10031, 1)

    num_training_example = x_train.shape[0]
    num_valid_example = x_valid.shape[0]
    n = x_train.shape[1]      # n = N - 1

    weights = {}
    biases = {}
    best_weights = {}
    best_biases = {}
    min_valid_error = sys.maxint

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
    train_ppl_list = []

    for e in range(epochs):
        training_error = 0
        valid_error = 0
        i_train = 0
        i_valid = 0
        val_ppl = 0
        train_ppl = 0
        ''' Traninig '''
        while i_train < num_training_example:
            j_train = i_train + batch_size
            x_indices = np.zeros((batch_size, n))
            x = np.zeros((batch_size, n * num_dim))
            y = np.zeros((batch_size, voc_size))
            if j_train < num_training_example:
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
            train_ppl += get_perplexity(num_training_example, a2, y)

            actual_batch = x_indices.shape[0]
            ''' Backprop '''
            dl_do2 = _deriv_crossEnt_softmax(a2, y) / actual_batch
            dl_db2 = np.sum(dl_do2, axis=0).reshape(biases[2].shape)
            dl_dW2 = np.dot(a1.T, dl_do2)
            da1_do1 = _deriv_tanh(a1)
            dl_da1 = np.dot(dl_do2, weights[2].T)
            dl_do1 = np.multiply(da1_do1, dl_da1)
            dl_db1 = np.sum(dl_do1, axis=0).reshape(biases[1].shape)
            dl_dW1 = np.dot(x.T, dl_do1)

            # NOTE: W0 gradients, to the corresponding indices in x_indices
            dl_dx = np.dot(dl_do1, weights[1].T)

            ''' SGD Update '''
            weights[2] = sgd(weights[2], dl_dW2, eta)
            weights[1] = sgd(weights[1], dl_dW1, eta)
            # weights[0] updates
            for i in range(actual_batch):
                for j in range(n):
                    row = x_indices[i, j]
                    weights[0][row,:] = \
                        sgd(weights[0][row,:], dl_dx[i, j*num_dim:(j+1)*num_dim], eta)

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
            val_ppl += get_perplexity(num_valid_example, a2, y)
            valid_error += cross_entropy(a2, y)
            i_valid = j_valid

        training_error_avg = training_error / num_training_example
        valid_error_avg = valid_error / num_valid_example

        if valid_error_avg < min_valid_error:
            min_valid_error = valid_error_avg
            best_weights = deepcopy(weights)
            best_biases = deepcopy(biases)

        print "##### Epoch %s ######\n \
total epoch=%s, eta=%s, hidden=%s, batch_size=%s \n \
training_error = %s, valid_error = %s, val_ppl=%s, train_ppl=%s\n" \
            % (e + 1, epochs, eta, num_hid, batch_size, training_error_avg, valid_error_avg, val_ppl, train_ppl)

    # write weights and biases to files
    print 'writing parameters...'
    with open(weights_file, 'wb') as f:
        pickle.dump(best_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(biases_file, 'wb') as f:
        pickle.dump(best_biases, f, protocol=pickle.HIGHEST_PROTOCOL)

    return best_weights, best_biases

def visualization(training_error_list, valid_error_list, val_ppl_list):
    '''
        visualize the train error and valid error
    '''
    # Cross Entropy
    plt.figure(1)
    plt.xlabel("# epochs")
    plt.ylabel("error")
    plt.plot(training_error_list, label='training error')
    plt.plot(valid_error_list, label='valid error')
    plt.title('Cross Entropy\n (learning rate=%s, hidden=%s)'\
            % (eta, num_hid))
    plt.legend()
    # Perplexity
    plt.figure(2)
    plt.xlabel("# epochs")
    plt.ylabel("perplexity")
    # plt.plot(train_ppl_list, label='training perplexity')
    plt.plot(val_ppl_list, label='validation perplexity')
    plt.title('Val Perplexity\n (learning_rate=%s, hidden=%s)'\
            % (eta, num_hid))

    plt.show()

def get_perplexity(val_total_words, p, y):
    '''
        get the perplexity according to the input
        input:
            p: output of the model
            y: desired output
    '''
    l = np.sum(y * np.log(p)) / val_total_words
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
    return -np.sum(y * np.log(o))

def softmax(x):
    """
        Input: an array
        Output: an array of softmax function of each element
    """
    x_copy = deepcopy(x)
    x_copy -= x_copy.max(1).reshape(x_copy.shape[0],1)
    return np.exp(x_copy) / np.sum(np.exp(x_copy), axis=1).reshape(x_copy.shape[0],1)

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

def load_data(data_file):
    '''
        load data from txt file.
    '''
    data_array = np.loadtxt(data_file, dtype='int32')
    # np.random.shuffle(data_array)
    row = data_array.shape[0]
    col = data_array.shape[1]

    x = data_array[:,0:col - 1]
    y = data_array[:, col - 1].reshape(row, 1)

    # Test
    print "### Load Data File %s ###" % data_file
    print x.shape, y.shape
    return x, y

def get_word_mapping(voc_file_name):
    '''
        get words and indices mapping.
        promise to be the same index for every word each time because
        we create the dict from a file

        return:
            dictionary: key: word, value: index
    '''
    word_dict = dict()
    index = 0
    with open(voc_file_name) as f:
        for line in f:
            word = line.strip()
            word_dict[word] = index
            index += 1
    return word_dict

def predict(weights, biases, three_word, next_num_words):
    '''
        Use left and right pointer to get 3 word each time and predict next word
        then move the left and right to predict next word

        input:
            weights, biases: best weights and biases from LM
            three_word: a list of word

        output:
            a list of 13 words(3 original + 10 predicted)
    '''
    n = N - 1
    global voc_file_name
    word_dict = get_word_mapping(voc_file_name)

    print 'predict for %s ...' % (three_word)

    x = np.zeros((1, weights[1].shape[0]))
    y = np.zeros((1, voc_size))
    for i in range(next_num_words):
        left = i
        # construct x
        for j in range(n):
            x[:,j*num_dim : (j+1)*num_dim] = weights[0][word_dict[three_word[left+j]], :]

        ''' Feed Forward '''
        o1 = feedforward(x, weights[1], biases[1])
        a1 = tanh(o1)

        o2 = feedforward(a1, weights[2], biases[2])
        a2 = softmax(o2)
        predicted_index = np.argmax(a2)
        for word,index in word_dict.items():
            if index == predicted_index:
                three_word.append(word)
                break
        print three_word

def get_distance(word_dict, embedding_weights, word_a, word_b):
    '''
        input:
                word_dict: store word index mapping
        embedding_weights: word vector (V*D)
                   word_a: word String
                   word_b: word String
        output: the Euclidean distance between the two words
    '''
    index_a = word_dict[word_a]
    index_b = word_dict[word_b]

    vec_a = embedding_weights[index_a]
    vec_b = embedding_weights[index_b]

    dist = np.linalg.norm(vec_a - vec_b)
    return dist

if __name__ == "__main__":
    best_weights = {}
    best_biases = {}
    # NOTE:get the best weights from the model(if params not stored)
    # best_weights, best_biases = get_best_params(epochs)
    # load weights from files if stored
    with open(weights_file, 'rb') as f:
        best_weights = pickle.load(f)

    with open(biases_file, 'rb') as f:
        best_biases = pickle.load(f)

    word_dict = get_word_mapping(voc_file_name)
    similar_words = [['monday', 'tuesday'],['man', 'woman'],['8.50', '8.53']]
    unsimilar_words = [['wolf', 'greatly'], ['document', 'foam'], ['ounces', 'early']]
    print 'similar words distance: '
    for words in similar_words:
        dist = get_distance(word_dict, best_weights[0], words[0], words[1])
        print dist

    print 'unsimilar words distance:'
    for words in unsimilar_words:
        dist = get_distance(word_dict, best_weights[0], words[0], words[1])
        print dist
    # randomly get 500 indices
    indices500 = random.sample(range(1, 8000), 500)
    similar_words_indices = []
    for words in similar_words:
        index0 = word_dict[words[0]]
        index1 = word_dict[words[1]]
        indices_list = [index0,index1]

        indices500.append(index0)
        indices500.append(index1)
        similar_words_indices.append(indices_list)

    print similar_words_indices

    ''' Visualize Weights V * D (8000 * 2)'''
    x = best_weights[0][:,0]
    y = best_weights[0][:,1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x,y)
    plt.show()
