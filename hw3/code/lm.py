import sys
import numpy as np


# tunable parameters
epochs = 100
eta = 0.01
num_dim = 16
num_hid = 128
batch_size = 64

def p32():
    # TODO: create a look up table for vocabulary
    voc_file_name = 'output8000'
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

    #

func_arg = {"-p32": p32}

if __name__ == "__main__":
    func_arg[sys.argv[1]]()
