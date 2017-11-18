from collections import defaultdict
import operator
from collections import Counter
import matplotlib.pyplot as plt


def get_vocabulary(filename):
    fd = open(filename, "r")
    content = fd.read()
    vocabulary = content.lower().split()

    d = Counter(vocabulary)
    sorted_vocabulary = sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:7997]
    truncated_vocabulary = [f[0] for f in sorted_vocabulary]
    truncated_vocabulary.extend(['START', 'END', 'UNK'])  # list
    truncated_vocabulary_set = set(truncated_vocabulary)
    lookup_table = {}
    # print sorted_vocabulary[:10]
    for i in range(len(truncated_vocabulary)):
        lookup_table[truncated_vocabulary[i]] = i

    return lookup_table, truncated_vocabulary_set, truncated_vocabulary


def get_grams(filename, lookup_table, truncated_vocabulary_set):
    fd = open(filename, "r")
    lines = fd.readlines()
    replaced_text = [['START'] + line.lower().split() + ['END'] for line in lines]

    for line in replaced_text:
        for i in range(len(line)):
            if line[i] not in truncated_vocabulary_set:
                line[i] = 'UNK'

    # count 4-grams and generate training grams
    grams = []
    gram_dict = defaultdict(int)
    for line in replaced_text:
        line = [lookup_table[word] for word in line]  #  mapped to word_id
        for i in range(len(line) - 3):
            # gram = ' '.join(line[i:i+4])
            # gram_dict[gram] += 1
            grams.append(line[i:i+4])

    # counts = sorted(gram_dict.values())
    # plt.plot(range(1, len(counts)+1), counts)
    # plt.show()

    # sorted_grams = sorted(gram_dict.items(), key=operator.itemgetter(1),reverse=True)[0:50]
    # top_50grams = [f[0] for f in sorted_grams]
    # print top_50grams
    # print gram_dict['0 *t*-1 . END']

    return grams

# get_vocabulary("train.txt")