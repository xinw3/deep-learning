import sys
import numpy as np


# tunable parameters
epochs = 100
eta = 0.01
num_dim = 16
num_hid = 128
batch_size = 64

train_file = 'train_ngram.txt'
val_file = 'val_ngram.txt'

voc_file_name = 'output8000'
voc_size = 8000
N = 4   # n-grams

def p32():

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

    for e in range(epochs):
        training_error = 0
        valid_error = 0
        i_train = 0
        i_valid = 0
        ''' Traninig '''
        while i_train < num_training_example
            j_train = i_train + batch_size
            if j_train < num_training_example:
                x = 



def init_weights(n_in, n_out):
    '''
        Xavier initialization of weights
    '''
    a = np.sqrt(6. / (n_in + n_out))
    return a * np.random.uniform(-1., 1., (n_in, n_out))

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
