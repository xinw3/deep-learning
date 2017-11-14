#!/usr/bin/python

import sys
import re
import matplotlib.pyplot as plt
import numpy as np

def plot_ngram():
    '''
        Input: output_ngram
               build list for 4-grams
    '''
    voc_file_name = 'output_ngram'
    count_list = []

    with open(voc_file_name) as f:
        for line in f:
            word, count = line.split('\t')
            count = int(count)
            count_list.append(count)

    plt.plot(count_list)
    plt.xlabel('id')
    plt.ylabel('# of ngrams')
    plt.title('ngrams distribution')
    plt.show()


if __name__ == "__main__":
    plot_ngram()
