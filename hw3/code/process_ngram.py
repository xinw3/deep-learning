import sys

def mapper():
    '''
        process n-grams files get the indices for each word in the n-grams
    '''
    # Build the word-index mapping
    voc_file_name = 'output8000'
    word_dict = dict()
    index = 0
    with open(voc_file_name) as f:
        for line in f:
            word = line.strip()
            word_dict[word] = index
            index += 1

    for line in sys.stdin:
        ngram,count = line.strip().split('\t')
        words = ngram.split()
        for i in range(len(words)):
            if words[i] not in word_dict:
                words[i] = 'UNK'

            index = word_dict[words[i]]
            sys.stdout.write(str(index))
            if i < len(words) - 1:
                sys.stdout.write(' ')
            else:
                sys.stdout.write('\n')

if __name__ == "__main__":
    mapper()
