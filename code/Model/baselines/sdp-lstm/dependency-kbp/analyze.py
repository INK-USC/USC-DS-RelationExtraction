import time
import os
import sys
import random
import tensorflow as tf
import numpy as np

import data_utils

tf.app.flags.DEFINE_string('data_dir', '../data/dependency/', 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_integer('num_class', '42', 'Number of total classes')
tf.app.flags.DEFINE_integer('vocab_size', 11893, 'Vocabulary size')
tf.app.flags.DEFINE_integer('sent_len', 32, 'Input sentence length. This is after the padding is performed.')

FLAGS = tf.app.flags.FLAGS

def get_unk_count_in_dataset(loader):
    total_count, unk_count = 0, 0
    padded_sentences = []
    for i in range(loader.batch_size):
        batch = loader.next_batch()
        padded_sentences += batch[0][data_utils.WORD_FIELD]
    padded_sentences += loader.get_residual()[0][data_utils.WORD_FIELD]
    print len(padded_sentences)
    for sent in padded_sentences:
        for t in sent:
            if t == data_utils.UNK_ID:
                unk_count += 1
            if t != data_utils.PAD_ID:
                total_count += 1
    return unk_count, total_count

def analyze_unk():
    corruption_prob = 0.06
    print "Loading data using vocab size %d..." % FLAGS.vocab_size
    word2id = data_utils.load_from_dump(FLAGS.data_dir + '%d.vocab' % FLAGS.vocab_size)
    train_loader = data_utils.DataLoader(FLAGS.data_dir + 'train.vocab%d.id' % FLAGS.vocab_size, 50, FLAGS.sent_len, unk_prob=corruption_prob)
    dev_loader = data_utils.DataLoader(FLAGS.data_dir + 'dev.vocab%d.id' % FLAGS.vocab_size, 50, FLAGS.sent_len)
    test_loader = data_utils.DataLoader(FLAGS.data_dir + 'test.vocab%d.id' % FLAGS.vocab_size, 50, FLAGS.sent_len)

    print "Counting..."
    train_unk, train_total = get_unk_count_in_dataset(train_loader)
    dev_unk, dev_total = get_unk_count_in_dataset(dev_loader)
    test_unk, test_total = get_unk_count_in_dataset(test_loader)

    print "Training token count:"
    print "\tunk:%d\ttotal:%d\tratio:%g" % (train_unk, train_total, 1.0*train_unk/train_total)
    print "Dev token count:"
    print "\tunk:%d\ttotal:%d\tratio:%g" % (dev_unk, dev_total, 1.0*dev_unk/dev_total)
    print "Test token count:"
    print "\tunk:%d\ttotal:%d\tratio:%g" % (test_unk, test_total, 1.0*test_unk/test_total)

def main():
    analyze_unk()

if __name__ == '__main__':
    main()
