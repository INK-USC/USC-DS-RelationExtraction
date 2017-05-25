# Script to predict and evaluate in a pipeline
__author__ = 'xiang'

import sys
from collections import  defaultdict
from evaluation import *
from emb_prediction import *

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print 'Usage: emb_test.py -TASK (classify/extract) \
        -DATA(BioInfer/NYT/Wiki) -METHOD(retype) -SIM(cosine/dot) -THRESHOLD'
        exit(-1)

    # do prediction here
    _task = sys.argv[1]
    _data = sys.argv[2]
    _method = sys.argv[3]
    _sim_func = sys.argv[4]
    _threshold = float(sys.argv[5])

    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'

    output = outdir +'/prediction_emb_' + _method + '_' + _sim_func + '.txt'
    ground_truth = load_labels(indir + '/mention_type_test.txt')

    ### Prediction
    if _task == 'extract':
        none_label_index = find_none_index(indir + '/type.txt')
        predict(indir, outdir, _method, _sim_func, _threshold, output, none_label_index)
    elif _task == 'classify':
        predict(indir, outdir, _method, _sim_func, _threshold, output, None)
    else:
        print 'wrong TASK argument!'
        exit(1)

    ### Evluate embedding predictions
    predictions = load_labels(output)
    print 'Evalaution:'
    if _task == 'extract':
        none_label_index = find_none_index(indir + '/type.txt')
        prec, rec, f1 = evaluate_rm_neg(predictions, ground_truth, none_label_index)
        # print 'precision:', prec
        # print 'recall:', rec
        # print 'f1:', f1
    elif _task == 'classify':
        prec, rec, f1 = evaluate_rm(predictions, ground_truth)
        # print 'accuracy:', prec
    else:
        print 'wrong TASK argument.'
        exit(1)
