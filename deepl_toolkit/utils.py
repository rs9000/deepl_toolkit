import torch
import json
from torch import nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_tensor_lengths(x, null_char):
    N, T = x.shape
    idx = torch.LongTensor(N).fill_(T - 1)

    # Find the last non-null element in each sequence
    x_cpu = x.data.cpu()
    for i in range(N):
        for t in range(T - 1):
            if x_cpu[i, t] != null_char and x_cpu[i, t + 1] == null_char:
                idx[i] = t
                break
    idx = idx.type_as(x.data).long()
    idx.requires_grad = False
    return idx

def load_json(filename):
    with open(filename) as f:
        vocab = json.load(f)
    return vocab


def save_json(filename, data):
    with open(filename, "w") as text_file:
        text_file.write(json.dumps(data))
    return


def restrict_w2v(w2v, restricted_word_set, filename):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        new_vectors_norm = None
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = new_index2entity
    w2v.index2word = new_index2entity
    w2v.vectors_norm = new_vectors_norm
    w2v.init_sims()
    w2v.save_word2vec_format(filename)

def update_learning_rate(optimizer, epoch, init_lr=0.001, decay=15):
    learning_rate =  init_lr * 0.5**(float(epoch) / decay)
    for param_group in optimizer.param_groups: param_group['lr'] = learning_rate

    return learning_rate