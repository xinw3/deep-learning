import numpy as np
import re
import matplotlib.pyplot as plt

fdist = {}

train_file = "train.txt"
val_file = "val.txt"
f = open(train_file, 'r')
text = f.read()
f.close()
words = text.lower().replace('\n', ' ').split(' ')
print len(words)
for word in words:
    if word in fdist:
        fdist[word] += 1
    else:
        fdist[word] = 1
list_tuple = sorted(fdist.items(), key=lambda kv: kv[1], reverse=True)
vocab = [i[0] for i in list_tuple]
vocab = vocab[0:7997]
vocab = vocab + ["START", "END", "UNK"]


def build_gram_list(file_name, vocab):
    gram4dict = {}
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_lc = line.lower().strip()
            words_list = re.split(' ', line_lc)
            words_list.insert(0, "START")
            words_list.append("END")
            for i in range(len(words_list)):
                if words_list[i] not in vocab:
                    words_list[i] = "UNK"
            for i in range(len(words_list)-3):
                gram4 = tuple(words_list[i:i+4])
                if gram4 in gram4dict:
                    gram4dict[gram4] += 1
                else:
                    gram4dict[gram4] = 1
    gram4list = sorted(gram4dict.items(), key=lambda kv: kv[1], reverse=True)
    print(len(gram4list))
    return gram4list

gram4list_train = build_gram_list(train_file, vocab)
gram4list_val = build_gram_list(val_file, vocab)
with open(val_file, 'r') as f:
    lines = f.readlines()
    val_lines_num = len(lines)


# with open("most_common.txt", 'w') as f:
#     for i in range(50):
#         f.write(' '.join(gram4list[i][0])+'\n')

# frequency = [i[1] for i in gram4list]
# indices = np.arange(len(gram4list))
# plt.bar(indices, frequency, color='r')
# plt.tight_layout()
# plt.show()


def word_to_index(gram4list, vocab):
    indexlist = []
    for i in range(len(gram4list)):
        temp = []
        for j in range(4):
            temp.append(vocab.index(gram4list[i][0][j]))
        for k in range(gram4list[i][1]):
            indexlist.append(temp)
    print(len(indexlist))
    return indexlist

indexlist_train = word_to_index(gram4list_train, vocab)
indexlist_val = word_to_index(gram4list_val, vocab)

V = 8000
D = 16
H = 128
bL = np.sqrt(6)/np.sqrt(2*V)
L = np.random.uniform(-bL, bL, [D, V]).astype(np.float32)
bK = np.sqrt(6)/np.sqrt(V + D)
K = np.random.uniform(-bK, bK, [H, D, 3]).astype(np.float32)
bh = np.zeros(H, dtype=np.float32)
bM = np.sqrt(6)/np.sqrt(H + V)
M = np.random.uniform(-bM, bM, [V, H]).astype(np.float32)
bo = np.zeros(V, dtype=np.float32)

alpha = 0.1
epochs = 100
train_num = len(indexlist_train)
val_num = len(indexlist_val)
batch_size = 200
batch_num = int(np.floor(train_num/batch_size))
train_data = np.array(indexlist_train)  # train_num * 4
val_data = np.array(indexlist_val)


def one_hot_coding(data, data_num):
    vec = np.reshape(data, -1)
    result = np.zeros([data_num, V, 4], dtype=np.float32)
    result[tuple(np.repeat(range(data_num), 4)), vec, tuple(np.tile(range(4), data_num))] = 1
    return result

train_xy = one_hot_coding(train_data, train_num)  # train_num * V * 4
val_xy = one_hot_coding(val_data, val_num)  # val_num * V * 4
val_x = val_xy[:, :, 0:3]  # val_num * V * 3
val_y = val_xy[:, :, 3]  # val_num * V
train_xy_shuffled = train_xy

for i in range(epochs):
    print i
    np.random.shuffle(train_xy)
    train_x_shuffled = train_xy_shuffled[:, :, 0:3]  # train_num * V * 3
    train_y_shuffled = train_xy_shuffled[:, :, 3]  # train_num * V

    # training
    print 'start training'
    for j in range(batch_num):
        print j
        # forward propagation
        x = train_x_shuffled[np.arange(j*batch_size, (j+1)*batch_size), :, :]  # batch_size * V * 3
        y = train_y_shuffled[np.arange(j*batch_size, (j+1)*batch_size), :]  # batch_size * V
        w = np.einsum('dv,bvs->bds', L, x)  # batch_size * D * 3
        a = np.einsum('hds,bds->bh', K, w) + bh  # batch_size * H
        h = a
        t = np.einsum('vh,bh->bv', M, h) + bo  # batch_size * V
        f = np.exp(t) / np.sum(np.exp(t), axis=1)[:, None]  # batch_size * V

        # backward propagation
        fy = np.sum(np.multiply(y, f), axis=1)  # batch_size
        gradient_f = -y / fy[:, None]  # batch_size * V
        gradient_t = f - y  # batch_size * V
        gradient_h = np.matmul(gradient_t, M)  # batch_size * H
        gradient_a = gradient_h  # batch_size * H
        gradient_w = np.einsum('bh,hds->bds', gradient_a, K)  # batch_size * D * 3

        gradient_M = np.einsum('bv,bh->vh', gradient_t, h) / batch_size
        gradient_bo = np.mean(gradient_t, axis=0)
        gradient_K = np.einsum('bh,bds->hds', gradient_a, w) / batch_size
        gradient_bh = np.mean(gradient_a, axis=0)
        gradient_L = np.einsum('bds,bvs->dv', gradient_w, x) / batch_size

        # update weights and bias
        M = M - alpha * gradient_M
        bo = bo - alpha * gradient_bo
        K = K - alpha * gradient_K
        bh = bh - alpha * gradient_bh
        L = L - alpha * gradient_L

    # prediction
    w = np.einsum('dv,bvs->bds', L, val_x)  # val_num * D * 3
    a = np.einsum('hds,bds->bh', K, w) + bh  # val_num * H
    h = a
    t = np.einsum('vh,bh->bv', M, h) + bo  # val_num * V
    f = np.exp(t) / np.sum(np.exp(t), axis=1)[:, None]  # val_num * V
    p_val = np.sum(np.multiply(val_y, f), axis=1)
    val_loss = -np.sum(np.log(p_val)) / val_num
    l = np.sum(np.log2(p_val)) / val_lines_num
    perplexity = np.power(2, -l)

    w = np.einsum('dv,bvs->bds', L, train_x_shuffled)  # train_num * D * 3
    a = np.einsum('hds,bds->bh', K, w) + bh  # train_num * H
    h = a
    t = np.einsum('vh,bh->bv', M, h) + bo  # train_num * V
    f = np.exp(t) / np.sum(np.exp(t), axis=1)[:, None]  # train_num * V
    p_train = np.sum(np.multiply(train_y_shuffled, f), axis=1)
    train_loss = -np.sum(np.log(p_train)) / train_num
    print "train loss: {} | val loss: {} | val perplexity: {}".format(train_loss, val_loss, perplexity)
