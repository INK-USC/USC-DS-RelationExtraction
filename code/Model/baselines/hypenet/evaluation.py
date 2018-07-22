import sys
from collections import defaultdict


def find_none_index(file_name):
    with open(file_name) as f:
        for line in f:
            entry = line.strip('\r\n').split('\t')
            if entry[0] == 'None':
                return int(entry[1])
        print('No None label!!!')
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


def evaluate_em(prediction, ground_truth):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:z
    """
    print("prediction:%d, ground:%d" % (len(prediction), len(ground_truth)))
    assert len(prediction) == len(ground_truth)
    count = len(prediction)
    # print 'Test', count, 'mentions'
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
        same_count = len(p & g)
        macro_precision += float(same_count) / float(len(p))
        macro_recall += float(same_count) / float(len(g))
        micro_n += same_count
        micro_precision += len(p)
        micro_recall += len(g)

    accuracy = float(same) / float(count)
    macro_precision /= count
    macro_recall /= count
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-8)
    micro_precision = micro_n / micro_precision
    micro_recall = micro_n / micro_recall
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    return accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1


###
def evaluate_rm(prediction, ground_truth):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:
    """
    pos_pred = 0.0
    pos_gt = 0.0 + len(ground_truth)
    true_pos = 0.0

    for i in prediction:
        # classified as pos example (Is-A-Relation)
        pos_pred += 1
        if i in ground_truth and prediction[i] == ground_truth[i]:
            true_pos += 1.0

    precision = true_pos / (pos_pred + 1e-8)
    recall = true_pos / (pos_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # print "predicted # Pos RMs:%d, ground-truth #Pos RMs:%d"%(int(pos_pred), int(pos_gt))

    return precision, recall, f1


def evaluate_rm_gold(prediction, ground_truth):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:
    """
    pos_gt = 0.0
    pos_pred = 0.0
    true_pos = 0.0

    for i in ground_truth:
        pos_gt += 1
        if i in prediction:
            pos_pred += 1
            if prediction[i] == ground_truth[i]:
                true_pos += 1.0

    precision = true_pos / (pos_pred + 1e-8)
    recall = true_pos / (pos_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("%d / %d gold RMs tested" % (int(pos_pred), int(pos_gt)))
    return precision, recall, f1


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
        if ground_truth[i] != {none_label_index}:
            pos_gt += 1

    for i in prediction:
        if prediction[i] != {none_label_index}:
            # classified as pos example (Is-A-Relation)
            pos_pred += 1
            if prediction[i] == ground_truth[i]:
                true_pos += 1.0

    precision = true_pos / (pos_pred + 1e-8)
    recall = true_pos / (pos_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # print "predicted # Pos RMs:%d, ground-truth #Pos RMs:%d"%(int(pos_pred), int(pos_gt))

    return precision, recall, f1


def min_max_nomalization(prediction):
    min_val = sys.maxsize
    max_val = -sys.maxsize
    prediction_normalized = defaultdict(tuple)
    for i in prediction:
        if prediction[i][1] < min_val:
            min_val = prediction[i][1]
        if prediction[i][1] > max_val:
            max_val = prediction[i][1]
    for i in prediction:
        score_normalized = (prediction[i][1] - min_val) / (max_val - min_val + 1e-8)
        prediction_normalized[i] = (prediction[i][0], score_normalized)
    return prediction_normalized


def evaluate_threshold(_threshold, ground_truth, prediction):
    # print 'threshold = ', _threshold
    prediction_cutoff = defaultdict(set)
    for i in prediction:
        if prediction[i][1] > _threshold:
            prediction_cutoff[i] = {prediction[i][0]}
    result = evaluate_rm(prediction_cutoff, ground_truth)
    # print result
    return result


def evaluate_threshold_neg(_threshold, ground_truth, none_label_index, prediction):
    # print 'threshold = ', _threshold
    prediction_cutoff = defaultdict(set)
    for i in prediction:
        if prediction[i][1] > _threshold:
            prediction_cutoff[i] = {prediction[i][0]}
    result = evaluate_rm_neg(prediction_cutoff, ground_truth, none_label_index)
    # print result
    return result


def tune_threshold(_threshold_list, ground_truth, none_label_index, prediction):
    result = defaultdict(tuple)
    for _threshold in _threshold_list:
        if none_label_index == None:
            result[_threshold] = evaluate_threshold(_threshold, ground_truth, prediction)
        else:
            result[_threshold] = evaluate_threshold_neg(_threshold, ground_truth, none_label_index, prediction)
    return result


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print('Usage: evaluation.py -DATA(nyt_candidates) -MODE(classifier/emb) -METHOD(retypeRM) -SIM(cosine/dot)')
        exit(-1)

    # do prediction here
    _data = sys.argv[1]
    _mode = sys.argv[2]  # emb or classifier/method name
    _method = sys.argv[3]  # emb method or null
    _sim_func = sys.argv[4]  # similarity functin or null

    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'

    output = outdir + '/prediction_' + _mode + '_' + _method + '_' + _sim_func + '.txt'
    ground_truth = load_labels(indir + '/mention_type_test.txt')
    predictions = load_labels(output)

    print('Predicted labels (embedding):')
    if '_neg' in _data:
        none_label_index = find_none_index(indir + '/type.txt')
        print('%f\t%f\t%f\t' % evaluate_rm_neg(predictions, ground_truth, none_label_index))
    else:
        print('%f\t%f\t%f\t' % evaluate_rm(predictions, ground_truth))
