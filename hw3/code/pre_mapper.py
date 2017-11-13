#!/usr/bin/python

import sys
import re

def mapper():
    '''
        Input: train
    '''
    word_count = dict()
    tags = ['UNK', 'START', 'END']
    for line in sys.stdin:
        words = line.split()
        for word in words:
            word_count[word] = word_count.get(word,0) + 1

    sorted_word_count = sorted(word_count, key=word_count.get, reverse=True)    # list
    word_set = set(sorted_word_count[:7997])

    for i in range(len(tags)):
        word_set.add(tags[i])
    print len(word_set)

if __name__ == "__main__":
    mapper()
