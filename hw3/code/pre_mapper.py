#!/usr/bin/python

import sys
import re

def mapper():
    '''
        Input: train
    '''
    for line in sys.stdin:
        words = line.split()
        for word in words:
            print '%s\t%s' % (word.lower(),1)


# tokenize the documents
def tokenize_doc(doc):
    return re.findall('\\w+', doc)

if __name__ == "__main__":
    mapper()
