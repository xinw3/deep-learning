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

def p32():
    # create a look up table for vocabulary

    word_dict = dict()  # ('word': index)
    index = 0
    # build word vocabulary
    # print 'building vocabulary...'
    with open(voc_file_name) as f:
        for line in f:
            word = line.strip()
            word_dict[word] = index
            index += 1

    # TODO: process input
    x_train, y_train = load_data(train_file)
    print x_train, y_train
    #


def load_data(data_file):
    data_array = np.loadtxt(data_file, delimiter=' ')
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
