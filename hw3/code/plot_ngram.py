#!/usr/bin/python

import sys
import re
import matplotlib.pyplot as plt
import numpy as np

def plot_ngram():
    '''
        Build 4-grams for train.txt

        Input: train.txt
               top 7997 words in the training file to build vocabulary
    '''
    voc_file_name = 'output_ngram'
    count_list = []


    with open(voc_file_name) as f:
        for line in f:
            word, count = line.split(',')
            count_list.append(int(count))


    print count_list
    x_axis = np.arange(len(count_list))

    plt.plot(x_axis,count_list)
    plt.xlabel('id')
    plt.ylabel('# of ngrams')
    plt.title('ngrams distribution')
    plt.show()


if __name__ == "__main__":
    plot_ngram()
