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
    word_set = set()
    tags = ['UNK', 'START', 'END']
    N = 4       # N is the number of grams
    # build word vocabulary
    # print 'building vocabulary...'
    index = 0
    total_ngrams = 0
    with open(voc_file_name) as f:
        for line in f:
            word, count = line.split(',')
            word_set.add(word)

    for i in range(len(tags)):
        word_set.add(tags[i])

    # print 'done'
    # print 'generating 4 gram from streaming input....'

    for line in sys.stdin:
        if (len(line) < 2):
            break
        line = 'START ' + line
        words = line.split(' ')
        words.append('END')

        stop_index = len(words) - N
        for i in range(stop_index + 1):
            if words[i] not in word_set:
                words[i] = 'UNK'
            ngram = words[i]
            j = i + 1
            # count = N - 1
            while j < i + N:
                if j >= len(words):
                    break
                if words[j] not in word_set:
                    words[j] = 'UNK'
                if j < i + N:
                    ngram += " "
                ngram += words[j]
                j += 1


            if len(ngram.split()) < 4:
                continue
            print '%s,%s' % (ngram,1)
            total_ngrams += 1

    print total_ngrams

def tokenize_doc(doc):
    return re.findall('\\w+', doc)

if __name__ == "__main__":
    mapper()