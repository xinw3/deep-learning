#!/usr/bin/python

import sys
import re

def mapper():
    '''
        Input: output7997   (words,count)
               top 7997 words in the training file to build vocabulary
    '''
    word_dict = dict()
    tags = ['UNK', 'START', 'END']
    # build word vocabulary
    index = 0
    for line in sys.stdin:
        word, count = line.split(',')
        word_dict[index] = word
        index += 1

    for i in range(len(tags)):
        word_dict[index + i] = tags[i]

    print word_dict

if __name__ == "__main__":
    mapper()
