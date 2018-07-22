import csv
import datetime
import json
import time
from os.path import expanduser, exists
from pprint import pprint
from zipfile import ZipFile

import numpy as np

SEED = np.random.randint(1,1001)
print('Using Random Seed: '+str(SEED))
np.random.seed(SEED)

from keras.utils.np_utils import to_categorical
from keras import backend as K, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Dense, Merge, BatchNormalization, TimeDistributed, Lambda, LSTM, SimpleRNN, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import get_file
from helper import sequence_from_dic, get_none_id, evaluate_rm_neg, get_class_num
from collections import defaultdict
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET = 'TACRED'

if DATASET == 'TACRED':
    USE_PROVIDED_DEV = True
    DEV_JSON_FILE = 'data/' + DATASET + '/dev.json'
    DEV_LABEL_JSON_FILE = 'data/' + DATASET + '/dev_label.json'
else:
    USE_PROVIDED_DEV = False

TEST_MODE = 0
TRAIN_JSON_FILE = 'data/' + DATASET + '/train.json'
TRAIN_LABEL_JSON_FILE = 'data/' + DATASET + '/train_label.json'
TEST_JSON_FILE = 'data/' + DATASET + '/test.json'
TEST_LABEL_JSON_FILE = 'data/' + DATASET + '/test_label.json'
GLOVE_FILE = 'data/glove.6B.100d.txt'
MODEL_WEIGHTS_FILE = 'model.h5'

none_id = get_none_id('data/' + DATASET + '/relation2id.txt')
print('None id: ', none_id)
num_classes = get_class_num('data/' + DATASET + '/relation2id.txt')
print('num_classes: ', num_classes)


MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 50
DEP_EMBEDDING_DIM = 50
DIR_EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 50
np.random.seed(RNG_SEED)

train_LEM = []
train_POS = []
train_DEP = []
train_DIR = []

train_labels = []

test_LEM = []
test_POS = []
test_DEP = []
test_DIR = []

test_labels = []

dev_LEM = []
dev_POS = []
dev_DEP = []
dev_DIR = []

dev_labels = []

train_LeftW = []
train_RightW = []
test_LeftW = []
test_RightW = []
dev_LeftW = []
dev_RightW = []

with open(TRAIN_JSON_FILE, encoding='utf-8') as train_file, \
        open(TRAIN_LABEL_JSON_FILE, encoding='utf-8') as train_label_file, \
        open(TEST_JSON_FILE, encoding='utf-8') as test_file, \
        open(TEST_LABEL_JSON_FILE, encoding='utf-8') as test_label_file:
    train_data = json.load(train_file)
    train_label_data = json.load(train_label_file)
    test_data = json.load(test_file)
    test_label_data = json.load(test_label_file)

for idx, data in enumerate(train_data):
    train_LEM.append(data[0])
    train_POS.append(data[1])
    train_DEP.append(data[2])
    train_DIR.append(data[3])
    train_labels.append(train_label_data[idx])

for idx, data in enumerate(test_data):
    test_LEM.append(data[0])
    test_POS.append(data[1])
    test_DEP.append(data[2])
    test_DIR.append(data[3])
    test_labels.append(test_label_data[idx])

print('Train numbers: %d' % len(train_data))
print('Test numbers: %d' % len(test_data))

if USE_PROVIDED_DEV:
    with open(DEV_JSON_FILE, encoding='utf-8') as dev_file,\
            open(DEV_LABEL_JSON_FILE, encoding='utf-8') as dev_label_file:
        dev_data = json.load(dev_file)
        dev_label_data = json.load(dev_label_file)

    for idx, data in enumerate(dev_data):
        dev_LEM.append(data[0])
        dev_POS.append(data[1])
        dev_DEP.append(data[2])
        dev_DIR.append(data[3])
        dev_labels.append(dev_label_data[idx])
    print('Dev numbers: %d' % len(dev_data))


LEM_dic = dict()
POS_dic = dict()
DEP_dic = dict()
DIR_dic = dict()

for LEMlist in train_LEM + test_LEM + dev_LEM:
    for LEM in LEMlist:
        if LEM not in LEM_dic:
            LEM_dic[LEM] = len(LEM_dic)

for POSlist in train_POS + test_POS + dev_POS:
    for POS in POSlist:
        if POS not in POS_dic:
            POS_dic[POS] = len(POS_dic)

for DEPlist in train_DEP + test_DEP + dev_DEP:
    for DEP in DEPlist:
        if DEP not in DEP_dic:
            DEP_dic[DEP] = len(DEP_dic)

for DIRlist in train_DIR + test_DIR + dev_DEP:
    for DIR in DIRlist:
        if DIR not in DIR_dic:
            DIR_dic[DIR] = len(DIR_dic)

nb_LEM = len(LEM_dic)
nb_POS = len(POS_dic)
nb_DEP = len(DEP_dic)
nb_DIR = len(DIR_dic)

train_LEM_word_sequences = sequence_from_dic(train_LEM, LEM_dic)
test_LEM_word_sequences = sequence_from_dic(test_LEM, LEM_dic)

train_POS_word_sequences = sequence_from_dic(train_POS, POS_dic)
test_POS_word_sequences = sequence_from_dic(test_POS, POS_dic)

train_DEP_word_sequences = sequence_from_dic(train_DEP, DEP_dic)
test_DEP_word_sequences = sequence_from_dic(test_DEP, DEP_dic)

train_DIR_word_sequences = sequence_from_dic(train_DIR, DIR_dic)
test_DIR_word_sequences = sequence_from_dic(test_DIR, DIR_dic)

if USE_PROVIDED_DEV:
    dev_LEM_word_sequence = sequence_from_dic(dev_LEM, LEM_dic)
    dev_POS_word_sequence = sequence_from_dic(dev_POS, POS_dic)
    dev_DEP_word_sequence = sequence_from_dic(dev_DEP, DEP_dic)
    dev_DIR_word_sequence = sequence_from_dic(dev_DIR, DIR_dic)

print("Processing", GLOVE_FILE)

embeddings_index = {}
with open(GLOVE_FILE, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings: %d' % len(embeddings_index))

word_embedding_matrix = np.zeros((nb_LEM + 1, EMBEDDING_DIM))
for word in LEM_dic:
    i = LEM_dic[word]
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector

train_LEM_data = pad_sequences(train_LEM_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_LEM_data = pad_sequences(test_LEM_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_POS_data = pad_sequences(train_POS_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_POS_data = pad_sequences(test_POS_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_DEP_data = pad_sequences(train_DEP_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_DEP_data = pad_sequences(test_DEP_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_DIR_data = pad_sequences(train_DIR_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_DIR_data = pad_sequences(test_DIR_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# print(test_rightWN_data[0])

train_label_data = np.array(train_labels, dtype=int)
test_label_data = np.array(test_labels, dtype=int)

train_label_cat = to_categorical(train_label_data, num_classes=num_classes)
test_label_cat = to_categorical(test_label_data, num_classes=num_classes)

if USE_PROVIDED_DEV:
    dev_LEM_data = pad_sequences(dev_LEM_word_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    dev_POS_data = pad_sequences(dev_POS_word_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    dev_DEP_data = pad_sequences(dev_DEP_word_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    dev_DIR_data = pad_sequences(dev_DIR_word_sequence, maxlen=MAX_SEQUENCE_LENGTH)

    dev_label_data = np.array(dev_labels, dtype=int)
    dev_label_cat = to_categorical(dev_label_data, num_classes=num_classes)

for idx, data in enumerate(train_data):
    train_LeftW.append([LEM_dic[data[0][0]]])
    train_RightW.append([LEM_dic[data[0][-1]]])

for idx, data in enumerate(test_data):
    test_LeftW.append([LEM_dic[data[0][0]]])
    test_RightW.append([LEM_dic[data[0][-1]]])

if USE_PROVIDED_DEV:
    for idx, data in enumerate(dev_data):
        dev_LeftW.append([LEM_dic[data[0][0]]])
        dev_RightW.append([LEM_dic[data[0][-1]]])

# print('Shape of train_leftW data tensor:', train_leftW_data.shape)
# print('Shape of train_rightPOS data tensor:', train_rightPOS_data.shape)
# print('Shape of label tensor:', train_label_data.shape)

# np.save('', word_embedding_matrix)

LEM = Sequential()
LEM.add(Embedding(nb_LEM + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                  trainable=True, mask_zero=True))

POS = Sequential()
POS.add(Embedding(nb_POS + 1, POS_EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True, mask_zero=True))

DEP = Sequential()
DEP.add(Embedding(nb_DEP + 1, DEP_EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True, mask_zero=True))

DIR = Sequential()
DIR.add(Embedding(nb_DIR + 1, DIR_EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True, mask_zero=True))

LeftWord = Sequential()
LeftWord.add(Embedding(nb_LEM + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=1,
                       trainable=True, mask_zero=True))
LeftWord.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,)))

RightWord = Sequential()
RightWord.add(Embedding(nb_LEM + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=1,
                        trainable=True, mask_zero=True))
RightWord.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,)))

SUM4 = Sequential()
# SUM4.add(Concatenate([LEM.output, POS.output, DEP.output, DIR.output]))
SUM4.add(Merge([LEM, POS, DEP, DIR], mode='concat'))
SUM4.add(LSTM(100, activation='relu', return_sequences=False, dropout=0.3, recurrent_dropout=0.5))

model = Sequential()
# model.add(Concatenate([LeftWord, SUM4, RightWord]))
# model.add(Concatenate([LeftWord.output, SUM4.output, RightWord.output]))
model.add(Merge([LeftWord, SUM4, RightWord], mode='concat'))
model.add(BatchNormalization())
# model.add(Dropout(0.15))
model.add(Dense(num_classes, activation='softmax'))
sgd = optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

if USE_PROVIDED_DEV:
    validation_temp = [np.asarray(dev_LeftW), np.asarray(dev_LEM_data), np.asarray(dev_POS_data), \
            np.asarray(dev_DEP_data), np.asarray(dev_DIR_data), np.asarray(dev_RightW)]
    validation_data = (validation_temp, dev_label_cat)
else:
    validation_data = None

if not TEST_MODE:
    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True),
                 EarlyStopping(monitor='val_acc', patience=7)]

    print("Starting training at", datetime.datetime.now())

    t0 = time.time()
    history = model.fit(
        [np.asarray(train_LeftW), np.asarray(train_LEM_data), np.asarray(train_POS_data), np.asarray(train_DEP_data),
         np.asarray(train_DIR_data), np.asarray(train_RightW)],
        train_label_cat,
        epochs=NB_EPOCHS,
        validation_split=VALIDATION_SPLIT,
        validation_data=validation_data,
        verbose=1,
        callbacks=callbacks)
    t1 = time.time()

    print("Training ended at", datetime.datetime.now())

    print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
else:
    model.load_weights(MODEL_WEIGHTS_FILE)
# y_pred = model.predict_classes([test_leftW_data, test_rightW_data, test_leftPOS_data, test_rightPOS_data,
#                                 test_leftGR_data, test_rightGR_data, test_leftWN_data, test_rightWN_data],
#                                batch_size=500)

# y_pred_score = model.predict_proba(
#     [np.asarray(test_LeftW), np.asarray(test_LEM_data), np.asarray(test_POS_data), np.asarray(test_DEP_data),
#      np.asarray(test_DIR_data), np.asarray(test_RightW)],
#     batch_size=100)


y_pred = model.predict_classes(
    [np.asarray(test_LeftW), np.asarray(test_LEM_data), np.asarray(test_POS_data), np.asarray(test_DEP_data),
     np.asarray(test_DIR_data), np.asarray(test_RightW)],
    batch_size=100)


assert len(y_pred) == len(test_label_data)
precision, recall, f1 = evaluate_rm_neg(y_pred, test_labels, none_id)
print("\nP, R, F1:")
print(precision, recall, f1)
