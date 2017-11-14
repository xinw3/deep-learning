import sys

def mapper():
    '''
        process n-grams files get lower case ngrams without counts
    '''
    for line in sys.stdin:
        ngram,count = line.strip().split('\t')
        print ngram.lower()

if __name__ == "__main__":
    mapper()
