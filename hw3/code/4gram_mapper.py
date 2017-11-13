#!/usr/bin/python

import sys
import re

def mapper():
    '''
        Build 4-grams for train.txt

        Input: train.txt
               top 7997 words in the training file to build vocabulary
    '''
    voc_file_name = 'output7997'
    word_dict = dict()
    tags = ['UNK', 'START', 'END']
    # build word vocabulary
    print 'building vocabulary...'
    index = 0
    with open(voc_file_name) as f:
        for line in f:
            word, count = line.split(',')
            word_dict[index] = word
            index += 1

    for i in range(len(tags)):
        word_dict[index + i] = tags[i]

    print word_dict
    print 'done'

if __name__ == "__main__":
    mapper()
