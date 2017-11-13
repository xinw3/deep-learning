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
    N = 4       # N is the number of grams
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

    print 'done'


    print 'generating 4 gram from streaming input....'

    for line in sys.stdin:
        line = 'START ' + line
        words = tokenize_doc(line)
        words.append('END')
        # print words_added

def tokenize_doc(doc):
    return re.findall('\\w+', doc)

if __name__ == "__main__":
    mapper()
