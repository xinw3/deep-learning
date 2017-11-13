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
    print 'building vocabulary...'
    index = 0
    with open(voc_file_name) as f:
        for line in f:
            word, count = line.split(',')
            word_set.add(word)

    for i in range(len(tags)):
        word_set.add(tags[i])

    print 'done'
    print 'generating 4 gram from streaming input....'

    for line in sys.stdin:
        if (len(line) < 2):
            break
        line = 'START ' + line
        words = line.split(' ')
        words.append('END')

        stop_index = len(words) - N
        for i in range(stop_index + 1):
            if words[i] not in word_set:
                continue
            ngram = words[i]
            j = i
            count = N - 1
            while count > 0:
                j += 1
                if j >= len(words):
                    break
                if words[j] not in word_set:
                    continue
                if count >= 1:
                    ngram += " "
                ngram += words[j]
                count -= 1
            if len(ngram) < 4:
                continue
            print '%s,%s' % (ngram,1)


def tokenize_doc(doc):
    return re.findall('\\w+', doc)

if __name__ == "__main__":
    mapper()
