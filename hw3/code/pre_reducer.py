#!/usr/bin/python

import sys
import re

def reducer():
    current_word = None
    current_count = 0
    word = None

    for line in sys.stdin:
        line = line.strip()
        word,count = re.split('\t', line)
        count = int(count)

        if current_word == word:
            current_count += count
        else:
            if current_word != None:
                print '%s\t%s' % (current_word, current_count)
            current_word = word
            current_count = count

    # print the first word
    if current_word == word:
        print '%s\t%s' % (current_word, current_count)


if __name__ == "__main__":
    reducer()
