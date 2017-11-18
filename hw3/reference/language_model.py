import numpy as np
import math
import matplotlib.pyplot as plt
import random
from datetime import datetime
from preprocessing import get_vocabulary, get_grams
random.seed(datetime.now())


def sigmoid(z):
    return 1/(1 + np.exp(- z))


def relu(z):
    z[z < 0] = 0
    return z


def activate(z, method):
    if method == 'sig':
        return sigmoid(z)
    if method == 'relu':
        return relu(z)
    if method == 'tanh':
        return np.tanh(z)
    if method == 'linear':
        return z


def derivative(z, method):
    if method == 'sig':
        return sigmoid(z) * (1 - sigmoid(z))
    if method == 'relu':
        result = np.zeros(z.shape)
        result[z <= 0] = 0
        result[z > 0] = 1
        return result
    if method == 'tanh':
        return 1 - (np.tanh(z) ** 2)
    if method == 'linear':
        return np.ones(z.shape)


def softmax(z):
    result = np.exp(z)
    result /= np.sum(result, 0)
    return result


def get_mini_batches(x, y, batch_size):
    random_idxs = np.random.choice(len(y), len(y), replace=False)
    x_shuffled = x[random_idxs,:]
    y_shuffled = y[random_idxs]
    mini_batches = [(x_shuffled[i:i+batch_size, :], y_shuffled[i:i+batch_size]) for
                    i in range(0, len(y), batch_size)]
    return mini_batches


def forward_pass(x, w, method, no_hidden_layer, batch_size):
    a = [np.array([0])] * (no_hidden_layer + 2)
    h = [np.array([0])] * (no_hidden_layer + 2)
    h[0] = x.T
    for k in range(1, no_hidden_layer + 1):
        a[k] = np.dot(w[k], h[k - 1])
        h[k] = activate(a[k], method)
        h[k] = np.concatenate((np.ones((1, batch_size)), h[k]))
        a[k] = np.concatenate((np.ones((1, batch_size)), a[k]))

    a[no_hidden_layer + 1] = np.dot(w[no_hidden_layer + 1], h[no_hidden_layer])
    f = softmax(a[no_hidden_layer + 1])

    return a, h, f


def lookup(word_id, lookup_table, C):
    no = word_id.shape[0]
    dim = C.shape[1]
    x = np.zeros((no, 3 * dim))
    for i in range(no):
        for j in range(3):
            x[i, j * dim: (j+1) * dim] = C[word_id[i, j], :]
    return x


def language_model_train(grams_train, grams_val, dim, lookup_table, no_hidden_units, no_hidden_layer,
                         batch_size, alpha, num_epoch, method):
    no_train = len(grams_train)  # list
    no_val = len(grams_val)
    no_words = len(lookup_table.keys())
    no_units = [3 * dim, no_hidden_units, no_words]
    C = np.random.normal(0, 0.1, (no_words, dim))
    delta_C = np.array([0])
    # w = np.random.normal(0, 0.1, (no_hidden_units, no_units))

    w = [np.array([0])] * (no_hidden_layer + 2)
    delta_w = [np.array([0])] * (no_hidden_layer + 2)
    delta_a = [np.array([0])] * (no_hidden_layer + 2)
    delta_h = [np.array([0])] * (no_hidden_layer + 2)

    for l in range(1, no_hidden_layer + 2):
        w_lb = math.sqrt(6) / math.sqrt(no_units[l - 1] + no_units[l])
        w_ub = - w_lb
        w[l] = np.random.uniform(w_lb, w_ub, (no_units[l], no_units[l - 1] + 1))
        w[l][:, 0] = np.zeros(no_units[l])

    y_train = np.array([f[-1] for f in grams_train])
    y_val = np.array([f[-1] for f in grams_val])
    x_id_train = np.array([f[:-1] for f in grams_train])  # should be grams
    x_id_val = np.array([f[:-1] for f in grams_val])

    loss_train = np.zeros(num_epoch)
    loss_val = np.zeros(num_epoch)
    perplexity_train = np.zeros(num_epoch)
    perplexity_val = np.zeros(num_epoch)

    for s in range(0, num_epoch):
        print s

        # lookup
        x_train = lookup(x_id_train, lookup_table, C)
        x_val = lookup(x_id_val, lookup_table, C)

        x_train = np.concatenate((np.ones((no_train, 1)), x_train), 1)
        x_val = np.concatenate((np.ones((no_val, 1)), x_val), 1)

        _, _, f_train = forward_pass(x_train, w, method, no_hidden_layer, no_train)
        for i in range(0, no_train):
            loss_train[s] -= math.log(f_train[y_train[i], i])

        _, _, f_val = forward_pass(x_val, w, method, no_hidden_layer, no_val)
        for i in range(0, no_val):
            loss_val[s] -= math.log(f_val[y_val[i], i])

        loss_train[s] /= no_train
        loss_val[s] /= no_val
        perplexity_val[s] = math.exp(loss_val[s])
        perplexity_train[s] = math.exp(loss_train[s])
        print 'perplexity_val: ', perplexity_val[s]

        # Backpropagation, mini-batch update
        mini_batches = get_mini_batches(x_id_train, y_train, batch_size)

        for batch_no in range(len(mini_batches)):
            batch = mini_batches[batch_no]
            x_id_b = batch[0]
            x_b = lookup(x_id_b, lookup_table, C)
            y_b = batch[1]
            bs = y_b.shape[0]
            # concat
            x_b = np.concatenate((np.ones((bs, 1)), x_b), 1)

            a, h, f = forward_pass(x_b, w, method, no_hidden_layer, bs)
            e = np.zeros(f.shape)
            for i in range(0, bs):
                e[y_b[i], i] = 1
            delta_a[no_hidden_layer + 1] = (f - e) / bs  # average

            for l in range(no_hidden_layer + 1, 0, -1):
                delta_w[l] = np.dot(delta_a[l], h[l - 1].T)

                if l > 1:
                    delta_h[l - 1] = np.dot(w[l].T, delta_a[l])
                    delta_a[l - 1] = delta_h[l - 1] * derivative(a[l - 1], method)
                    delta_a[l - 1] = delta_a[l - 1][1:, :]

            for l in range(no_hidden_layer + 1, 0, -1):
                w[l] = w[l] - alpha * delta_w[l]

            # update representation matrix C
            delta_C = w[1][:, 1:].T.dot(delta_a[1])
            for i in range(bs):
                for j in range(3):
                    C[x_id_b[i, j], :] -= delta_C[j * dim: (j+1) * dim, i].T

    return w, C, loss_train, loss_val, perplexity_train, perplexity_val


def neural_network_predict(x_id, lookup_table, C, w, no_hidden_layer, method):
    n = x_id.shape[0]
    x = lookup(x_id, lookup_table, C)
    x = np.concatenate((np.ones((n, 1)), x), 1)
    a, h, f = forward_pass(x, w, method, no_hidden_layer, n)
    prediction = np.argmax(f, axis=0)
    return prediction


def compute_dist_between_words(word1, word2, lookup_table, C):
    id1 = lookup_table[word1]
    id2 = lookup_table[word2]
    return np.linalg.norm(C[id1, :] - C[id2, :])


def main():
    lookup_table, truncated_vocabulary_set, truncated_vocabulary = get_vocabulary("train.txt")
    grams_train = get_grams("train.txt", lookup_table, truncated_vocabulary_set)
    grams_val = get_grams("val.txt", lookup_table, truncated_vocabulary_set)
    print('number of training grams: ', len(grams_train))
    print('number of validation grams: ', len(grams_val))
    dim = 16

    num_epoch = 100
    alpha = 0.01  # learning rate
    no_hidden_layer = 1
    no_hidden_units = 128
    method = 'linear'
    batch_size = 128

    w, C, loss_train, loss_val, perplexity_train, perplexity_val = language_model_train(grams_train, grams_val, dim, lookup_table,
                                                                   no_hidden_units, no_hidden_layer, batch_size, alpha,
                                                                   num_epoch, method)

    print ' min validation entropy:', np.min(loss_val)
    # print ' min train entropy:', np.min(loss_train)

    plt.plot(range(1, num_epoch + 1), loss_train, 'r', range(1, num_epoch + 1), loss_val, 'g')
    plt.xlabel('number of epoch')
    plt.ylabel('cross-entropy loss')
    plt.legend(['loss_train', 'loss_val'])
    plt.show()

    plt.plot(range(1, num_epoch + 1), perplexity_train, 'r', range(1, num_epoch + 1), perplexity_val, 'g')
    plt.xlabel('number of epoch')
    plt.ylabel('perplexity')
    plt.legend(['perplexity_train', 'perplexity_val'])
    plt.show()
    #
    start_words = [['city', 'of', 'new'], ['he', 'is', 'the'], ['life', 'in', 'the'],
                   ['government', 'of', 'united'], ['this', 'will', 'be']]
    result = []
    for grams in start_words:
        predicted_words = [lookup_table[word] for word in grams]
        for i in range(10):
            x_id = np.array([predicted_words[-3:]])
            new_word = neural_network_predict(x_id, lookup_table, C, w, no_hidden_layer, method)
            predicted_words.append(new_word)
        predicted_words = [truncated_vocabulary[word_id] for word_id in predicted_words]
        print ' '.join(predicted_words)
        result.append(' '.join(predicted_words))

    # save results
    w = np.array(w)
    saved_var = np.array([w, C, loss_train, loss_val, perplexity_train, perplexity_val])
    np.save('saved_var_q3_2_teng', saved_var)

    # for i in range(0, no_hidden_units):
    #     plt.subplot(10, 10, i + 1)
    #     plt.imshow(np.reshape((w[i, :] - np.min(w[i, :]))/(np.max(w[i, :]) - np.min(w[i, :])), (28, 28)),
    #                aspect='auto', interpolation='none', extent=[1, 28, 1, 28], origin='lower', cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()


if __name__ == '__main__':
    main()