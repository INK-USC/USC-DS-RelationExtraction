__author__ = 'xiang'
import sys
from collections import  defaultdict

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

###
def evaluate_rm(prediction, ground_truth):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:
    """
    pos_pred = 0.0
    pos_gt = len(ground_truth) + 0.0
    true_pos = 0.0

    for i in prediction:
        # classified as pos example (Is-A-Relation)
        pos_pred += 1.0
        if i in ground_truth and prediction[i] == ground_truth[i]:
            true_pos += 1.0

    precision = true_pos / (pos_pred + 1e-8)
    recall = true_pos / (pos_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # print "predicted # Pos RMs:%d, ground-truth #Pos RMs:%d"%(int(pos_pred), int(pos_gt))

    return precision,recall,f1

###
def evaluate_rm_neg(prediction, ground_truth, none_label_index):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:
    """
    # print '[None] label index:', none_label_index

    pos_pred = 0.0
    pos_gt = 0.0
    true_pos = 0.0
    for i in ground_truth:
        if ground_truth[i] != set([none_label_index]):
            pos_gt += 1.0

    for i in prediction:
        if prediction[i] != set([none_label_index]):
            # classified as pos example (Is-A-Relation)
            pos_pred += 1
            if prediction[i] == ground_truth[i]:
                true_pos += 1.0

    precision = true_pos / (pos_pred + 1e-8)
    recall = true_pos / (pos_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # print "predicted # Pos RMs:%d, ground-truth #Pos RMs:%d"%(int(pos_pred), int(pos_gt))

    return precision,recall,f1


if __name__ == "__main__":

    if len(sys.argv) != 6:
        print 'Usage: evaluation.py  -TASK (classify/extract) -DATA(nyt_candidates) -MODE(classifier/emb) -METHOD(retypeRM) -SIM(cosine/dot)'
        exit(-1)

    # do prediction here
    _task = sys.argv[1] # classifer / extract
    _data = sys.argv[2]
    _mode = sys.argv[3] # emb or classifier/method name
    _method = sys.argv[4] # emb method or null
    _sim_func = sys.argv[5] # similarity functin or null

    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'

    output = outdir +'/prediction_' + _mode + '_' + _method + '_' + _sim_func + '.txt'
    ground_truth = load_labels(indir + '/mention_type_test.txt')
    predictions = load_labels(output)

    if _task == 'extract':
        none_label_index = find_none_index(indir + '/type.txt')
        prec, rec, f1 = evaluate_rm_neg(predictions, ground_truth, none_label_index)
        print 'precision:', prec
        print 'recall:', rec
        print 'f1:', f1
    elif _task == 'classify':
        prec, rec, f1 = evaluate_rm(predictions, ground_truth)
        print 'accuracy:', prec
    else:
        print 'wrong TASK argument.'
        exit(1)
 




