__author__ = 'xiang'

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('code/Classifier/')

import json
import os
from collections import defaultdict
from math import sqrt
import operator
from DataIO import *
from Classifier import casestudy

def sim_func(v1, v2, _MODE):
    val = 0.0
    if _MODE == 'dot':
        ### dot product:
        val = sum( [v1[i]*v2[i] for i in range(len(v1))] )
    elif _MODE == 'cosine':
        ### cosine sim:
        norm1 = sqrt(sum( [v1[i]*v1[i] for i in range(len(v1))] ))
        norm2 = sqrt(sum( [v2[i]*v2[i] for i in range(len(v1))] ))
        val = sum( [v1[i]*v2[i]/norm1/norm2 for i in range(len(v1))] )
    return val

# Embedding of different nodes
class Embedding:
    def __init__(self, file_name):
        self._embs = []
        self._node_size = 0
        self._vector_size = 0
        # load file to embedding array
        with open(file_name) as f:
            seg = f.readline().split(' ')
            self._node_size = int(seg[0])
            self._vector_size = int(seg[1])
            # self._embs = [np.zeros(self._vector_size) for i in range(self._node_size)]
            self._embs = [[] for i in range(self._node_size)]
            for line in f:
                seg = line.strip().split('\t')
                idx = int(seg[1])
                _emb = seg[2].split(' ')
                # self._embs[idx] = np.array([float(x) for x in _emb])
                self._embs[idx] = [float(x) for x in _emb]
        print 'emb:', self._node_size, self._vector_size


    def get_embedding(self, index):
        return self._embs[index]

class Network:
    def __init__(self, file_name):
        self._network = defaultdict(list)
        # load file to network dictionary
        cnt = 0
        with open(file_name) as f:
            for line in f:
                seg = line.strip('\r\n').split('\t')
                self._network[int(seg[0])].append(int(seg[1]))
                cnt += 1
        # print 'edges:', cnt

    # return list of features
    def get_neighbors(self, idx):
        return self._network[idx]


# Predict types from feature embeddings
class Predicter_useFeatureEmb:
    def __init__(self, embs_feature, embs_type, network_mention_feature, typefile, sim_func):
        self._embs_feature = Embedding(embs_feature)
        self._embs_type = Embedding(embs_type)
        assert self._embs_feature._vector_size == self._embs_type._vector_size
        self._network_mention_feature = Network(network_mention_feature)
        self._sim_func = sim_func

    # get embedding vector for a mention
    def get_mention_embedding(self, mention_id):
        # from _network_mention_feature & _emb_feature
        feature_list = self._network_mention_feature.get_neighbors(mention_id)
        if len(feature_list) == 0:
            print 'No feature for this test mention!'
        _emb_mention = [0.0 for i in range(self._embs_feature._vector_size)]
        for feature_id in feature_list:
            for i in range(self._embs_feature._vector_size):
                _emb_mention[i] += self._embs_feature.get_embedding(feature_id)[i] / float(len(feature_list))
        return _emb_mention

    # predict types given a mention embedding
    def predict_types_for_rm_maximum(self, mention_id, candidate):
        _type_size = self._embs_type._node_size
        _emb_mention = self.get_mention_embedding(mention_id)
        # calculate scores and find maximum score
        max_index = -1
        max_score = -sys.maxint
        for i in candidate:
            _emb_type = self._embs_type.get_embedding(i)
            score = sim_func(_emb_mention, _emb_type, self._sim_func)
            if  score > max_score:
                    max_index = i
                    max_score = score

        return max_index, max_score


def predict(indir, outdir, _method, _sim_func, _threshold, output, none_label_index):

    predicter = Predicter_useFeatureEmb(\
        embs_feature=os.path.join(outdir + '/emb_' + _method + '_feature.txt'), \
        embs_type=os.path.join(outdir + '/emb_' + _method + '_type.txt'), \
        network_mention_feature=os.path.join(indir + '/mention_feature_test.txt'), \
        typefile=os.path.join(indir + '/type.txt'), \
        sim_func=_sim_func)

    with open(os.path.join(indir + '/mention_feature_test.txt')) as f,\
         open(output, 'w') as g:
        mentions_ids = load_mentionids(os.path.join(indir + '/mention_feature_test.txt'))
        all_candidates = load_all_candidates(os.path.join(indir + '/type.txt'), mentions_ids)
        cnt = 0
        pos_cnt = 0
        mentions_tested = set()
        labels = []
        scores = []
        mentions = []
        for line in f:
            seg = line.strip('\r\n').split('\t')
            mention_id = int(seg[0])
            if mention_id not in mentions_tested:
                mentions_tested.add(mention_id)
                label, score = predicter.predict_types_for_rm_maximum(mention_id, all_candidates[mention_id])
                if none_label_index != None and score == 0.0:
                    label = none_label_index
                    # print 'No Feature!'
                mentions.append(mention_id)
                labels.append(label)
                scores.append(score)
                cnt += 1

        scores_normalized = min_max_normalization(scores)
        # print scores_normalized
        for i in range(len(mentions)):
            if scores_normalized[i] > _threshold:
                g.write(str(mentions[i])+'\t'+str(labels[i])+'\t'+ str(scores_normalized[i]) + '\n')
                pos_cnt += 1

        f.close()
        g.close()
    # print pos_cnt, '/', cnt, 'are detected as mentions'

def min_max_normalization(scores):
    min_score = 0.0
    max_score = 0.0
    for score in scores:
        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score
    scores_normalized = []
    for score in scores:
        score_normalized = (score - min_score) / (max_score - min_score + 1e-8)
        scores_normalized.append(score_normalized)
    return scores_normalized

def load_mentionids(filename):
    """
    Load mention id as a set.
    e.g.[[0,1,2],[1,2]]
    """
    with open(filename) as f:
        indexes = set()
        for line in f:
            seg = line.strip('\r\n').split('\t')
            indexes.add(int(seg[0]))
        return indexes

def load_candidates(filename, indexes):
    """
    Load data as a dict of list.
    e.g.{0:[0,1,2],1:[1,2]}
    """
    with open(filename) as f:
        data = defaultdict(list)
        for line in f:
            seg = line.strip('\r\n').split('\t')
            index = int(seg[0])
            if index in indexes:
                data[index].append(int(seg[1]))
        return data

def load_all_candidates(filename, indexes):
    """
    Load data as a dict of list.
    e.g.{0:[0,1,2],1:[1,2]}
    """
    type_list = []
    with open(filename) as f:
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if len(seg) == 3:
                tid = int(seg[1])
                type_list.append(tid)
    # print 'all tid: ', type_list

    data = defaultdict(list)
    for index in indexes:
        data[index] = type_list
    return data




if __name__ == "__main__":

    if len(sys.argv) != 5:
        print 'Usage: emb_prediction.py -DATA(nyt_candidates) -METHOD(retypeRM) -SIM(cosine/dot) -THRESHOLD'
        exit(-1)

    # do prediction here
    _data = sys.argv[1]
    _method = sys.argv[2]
    _sim_func = sys.argv[3]
    _threshold = float(sys.argv[4])

    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'

    ### Prediction
    type_file = indir + '/type.txt'
    mention_file = indir + '/mention.txt'
    json_file = indir + '/test_new.json'
    output = outdir +'/prediction_emb_' + _method + '_' + _sim_func + '.txt'
    # intext_output = outdir +'/predictionInText_emb_' + _method + '_' + _sim_func + '.txt'

    if '_neg' in _data:
        none_label_index = find_none_index(indir + '/type.txt')
        predict(indir, outdir, _method, _sim_func, _threshold, output, none_label_index)
    else:
        predict(indir, outdir, _method, _sim_func, _threshold, output, None)

    ### Write inText Results
    # mention_mapping = load_map(mention_file, 'mention')
    # label_mapping = load_map(type_file, 'label')
    # clean_mentions = load_mention_type(output)
    # casestudy(json_file, intext_output, mention_mapping, label_mapping, clean_mentions)
