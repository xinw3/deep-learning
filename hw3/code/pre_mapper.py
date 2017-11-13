#!/usr/bin/python

import sys
import re

# TODO: make the data lower case

# TODO:

def mapper():
    for line in sys.stdin:
        words = tokenize_doc(line)
        for word in words:
            print '%s,%s' % (word.lower(),1)


# tokenize the documents
def tokenize_doc(doc):
    return re.findall('\\w+', doc)

if __name__ == "__main__":
    mapper()
