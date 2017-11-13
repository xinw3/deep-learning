#!/usr/bin/python

import sys
import re

def mapper():
    '''
        Input: output7997   (words,count)
               top 7997 words in the training file to build vocabulary
    '''
    wordList = []
    # build word vocabulary
    for line in sys.stdin:
        word, count = line.split(',')
        wordList.append(word)

    wordList.append('UNK')
    wordList.append('START')
    wordList.append('END')
    print wordList

# tokenize the documents
def tokenize_doc(doc):
    return re.findall('\\w+', doc)

if __name__ == "__main__":
    mapper()
