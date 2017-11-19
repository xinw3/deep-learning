#!/usr/bin/python

import sys
import re
import matplotlib.pyplot as plt


def mapper():
    '''
        This file is used for process the output from terminal,
        in which case, when we run the script on a virtual machine
        and there is not a proper UI to show the plots.

        We write the output from the terminal to a file and directly process
        the file.

        Input: output in terminal
        Output: plots for training error, validation error and perplexities.
    '''
    train_error_list = []
    val_error_list = []
    val_ppl_list = []
    eta = 0.01
    num_hid = 128

    for line in sys.stdin:
        fields = line.strip().split(',')
        if not fields[0].startswith('training'):
            continue
        for field in fields:
            values = field.split('=')
            if values[0].strip() == 'training_error':
                train_error_list.append(float(values[1].strip()))
            elif values[0].strip() == 'valid_error':
                val_error_list.append(float(values[1].strip()))
            elif values[0].strip() == 'val_ppl':
                val_ppl_list.append(float(values[1].strip()))


    ''' Visualization '''
    # Cross Entropy
    plt.figure(1)
    plt.xlabel("# epochs")
    plt.ylabel("error")
    plt.plot(train_error_list, label='training error')
    plt.plot(val_error_list, label='valid error')
    plt.title('Cross Entropy\n (learning rate=%s, hidden=%s)'\
            % (eta, num_hid))
    plt.legend()
    # Perplexity
    plt.figure(2)
    plt.xlabel("# epochs")
    plt.ylabel("perplexity")
    plt.plot(val_ppl_list, label='validation perplexity')
    plt.title('Val Perplexity\n (learning_rate=%s, hidden=%s)'\
            % (eta, num_hid))

    plt.show()

if __name__ == "__main__":
    mapper()
