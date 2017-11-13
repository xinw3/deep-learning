#!/usr/bin/python

import sys
import re

def reducer():
    current_word = None
    current_count = 0
    word = None

    for line in sys.stdin:
        line_splited = line.split(',')
        if len(line_splited) < 2:
            continue
        word = line_splited[0]
        count = line_splited[1]
        count = int(count)

        if current_word == word:
            current_count += count
        else:
            if current_word != None:
                print '%s,%s' % (current_word, current_count)
            current_word = word
            current_count = count

    # print the first word
    if current_word == word:
        print '%s,%s' % (current_word, current_count)


if __name__ == "__main__":
    reducer()
