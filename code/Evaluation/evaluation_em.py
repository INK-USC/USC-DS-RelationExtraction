__author__ = 'xiang'

import sys
from collections import  defaultdict
import os

def find_none_index(file_name):
    with open(file_name) as f:
        for line in f:
            entry = line.strip('\r\n').split('\t')
            if entry[0] == 'None':
                return int(entry[1])
        print 'No None label!!!'
        return

def load_labels(file_name):
    ### To Do: "None" RMs should NOT in ground_truth (double check whether we will have that)
    labels = defaultdict(set)
    with open(file_name) as f:
        for line in f:
            seg = line.strip('\r\n').split('\t')
            try:
                labels[int(seg[0])].add(int(seg[1]))
            except:
                labels[int(seg[0])].add(int(float(seg[1])))
        f.close()
    return labels

def load_labels_gt(file_name, _data):
    gt_file = 'data/source/' + _data + '/em_gt.txt'
    map_file = 'data/intermediate/' + _data + '/em/mention.txt'
    gt = set()
    if os.path.isfile(gt_file):
        map = {}
        with open(gt_file) as gt_f, open(map_file) as map_f:
            for line in map_f:
                seg = line.strip('\r\n').split('\t')
                map[seg[0]] = seg[1]
            for line in gt_f:
                gt.add(map[line.strip('\r\n')])
    print '# test EMs: ', len(gt)
    labels = defaultdict(set)
    with open(file_name) as f:
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if len(gt) > 0 and seg[0] in gt:
                try:
                    labels[int(seg[0])].add(int(seg[1]))
                except:
                    labels[int(seg[0])].add(int(float(seg[1])))
        f.close()
    #print labels
    return labels

def load_raw_labels(file_name, ground_truth):
    labels = defaultdict(set)
    with open(file_name) as f:
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if int(seg[0]) in ground_truth:
                labels[int(seg[0])].add(int(seg[1]))
        f.close()
    return labels

def load_label_score(file_name):
    labels = defaultdict(tuple)
    with open(file_name) as f:
        for line in f:
            seg = line.strip('\r\n').split('\t')
            try:
                if seg[2] == '-Infinity':
                    labels[int(seg[0])] = (int(float(seg[1])), 0.0)
                else:
                    labels[int(seg[0])] = (int(seg[1]), float(seg[2]))
            except:
                if seg[2] == '-Infinity':
                    labels[int(seg[0])] = (int(float(seg[1])), 0.0)
                else:
                    labels[int(seg[0])] = (int(float(seg[1])), float(seg[2]))
        f.close()
    return labels


def evaluate_em(prediction, ground_truth):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:z
    """
    print "prediction:%d, ground:%d"%(len(prediction),len(ground_truth))
    # assert len(prediction) == len(ground_truth)
    count = len(prediction)
    same = 0
    macro_precision = 0.0
    macro_recall = 0.0
    micro_n = 0.0
    micro_precision = 0.0
    micro_recall = 0.0

    for i in ground_truth:
        p = prediction[i]
        g = ground_truth[i]
        if p == g:
            same += 1
        same_count = len(p&g)
        macro_precision += float(same_count)/float(len(p) + 1e-8)
        macro_recall += float(same_count)/float(len(g) + 1e-8)
        micro_n += same_count
        micro_precision += len(p)
        micro_recall += len(g)

    accuracy = float(same) / float(count)
    macro_precision /= count
    macro_recall /= count
    macro_f1 = 2*macro_precision*macro_recall/(macro_precision + macro_recall + 1e-8)
    micro_precision = micro_n/micro_precision
    micro_recall = micro_n/micro_recall
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    return accuracy,macro_precision,macro_recall,macro_f1,micro_precision,micro_recall,micro_f1


### evaluation break-down by types (to-do)
# def evaluation_by_type()


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print 'Usage: evaluation.py -DATA(nyt_candidates) -MODE(classifier/emb) -METHOD(retypeRM) -SIM(cosine/dot)'
        exit(-1)

    # do prediction here
    _data = sys.argv[1]
    _mode = sys.argv[2] # emb or classifier/method name
    _method = sys.argv[3] # emb method or null
    _sim_func = sys.argv[4] # similarity functin or null

    indir = 'data/intermediate/' + _data + '/em'
    outdir = 'data/results/' + _data + '/em'
    output = outdir +'/prediction_' + _mode + '_' + _method + '_' + _sim_func + '.txt'

    if _data == 'kbp_candidates':
        ground_truth = load_labels_gt(indir + '/mention_type_test.txt', _data)
        predictions = load_labels_gt(output, _data)
    else:
        ground_truth = load_labels(indir + '/mention_type_test.txt')
        predictions = load_labels(output)

    print 'Predicted labels (embedding):'
    if '_neg' in _data:
        none_label_index = find_none_index(indir + '/type.txt')
        if none_label_index != None:
            raise Exception('entity mention type should not be none.')
        #print '%f\t%f\t%f\t' % evaluate_rm_neg(predictions, ground_truth, none_label_index)
    else:
        print '%f\t%f\t%f\t%f\t%f\t%f\t%f' % evaluate_em(predictions, ground_truth)




