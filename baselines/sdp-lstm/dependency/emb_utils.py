import os
import sys
import numpy as np
import cPickle as pickle
from collections import Counter
import argparse

import data_utils

DATA_ROOT = "../data/"
EMB_ROOT = "../../glove"

np.random.seed(1234)

def _load_glove_vec(fname, vocab, dim):
    """
    Loads vectors from Glove pre-trained dataset.
    """
    word_vecs = {}
    with open(fname, 'r') as f:
        for line in f:
            array = line.strip().split()
            assert(len(array) == dim + 1)
            w = array[0]
            if w in vocab:
                v = [float(x) for x in array[1:]]
                word_vecs[w] = np.array(v)
    return word_vecs

def _load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return (word_vecs, layer1_size)

def _add_random_vec(word_vecs, vocab, dim=300, scale=1.0):
    """
    The scale is 0.25 for word2vec, and 1.0 for glove? 
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-scale,scale,dim)
    return word_vecs

def prepare_pretrained_embedding(fname, word2id, dim):
    print 'Reading pretrained word vectors from file ...'
    word_vecs = _load_glove_vec(fname, word2id, dim)
    num_loaded = len(word_vecs)
    word_vecs = _add_random_vec(word_vecs, word2id, dim, 1.0)
    num_generated = len(word_vecs) - num_loaded
    print "Reading finished. %d vectors loaded, and %d generated." % (num_loaded, num_generated)
    embedding = np.zeros([len(word2id), dim])
    for w,idx in word2id.iteritems():
        embedding[idx,:] = word_vecs[w]
    print 'Generated embeddings with shape ' + str(embedding.shape)
    return embedding

def main():
    parser = argparse.ArgumentParser(description="Create initial word embedding matrix from pretrained word vectors.")
    parser.add_argument('--vocab_size', default=36002, type=int, help='The vocabulary size for the embedding matrix.')
    parser.add_argument('--dim', default=300, type=int, help='The dimension of embeddings.')
    args = parser.parse_args()

    dim = args.dim
    vocab_size = args.vocab_size
    print "Creating embedding matrix of size %d x %d" % (vocab_size, dim)
    
    emb_file = EMB_ROOT + "/glove.6B.%dd.txt" % dim
    print "Creating embeddings from file " + emb_file
    word2id = data_utils.load_from_dump(os.path.join(DATA_ROOT, "dependency/%d.vocab" % vocab_size))
    embedding = prepare_pretrained_embedding(emb_file, word2id, dim)
    np.save(DATA_ROOT + 'dependency/emb-v%d-d%d.npy' % (vocab_size, dim), embedding)
    print "Embedding matrix of size %d x %d has been created and saved!" % (embedding.shape[0], embedding.shape[1])

if __name__ == '__main__':
    main()
