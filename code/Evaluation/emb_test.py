# Script to predict and evaluate in a pipeline
__author__ = 'xiang'

import sys
from collections import  defaultdict
from evaluation import *
from emb_prediction import *

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print 'Usage: emb_test.py -DATA(nyt_candidates) -METHOD(retypeRM) -SIM(cosine/dot) -THRESHOLD'
        exit(-1)

    # do prediction here
    _data = sys.argv[1]
    _method = sys.argv[2]
    _sim_func = sys.argv[3]
    _threshold = float(sys.argv[4])

    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'

    output = outdir +'/prediction_emb_' + _method + '_' + _sim_func + '.txt'
    ground_truth = load_labels(indir + '/mention_type_test.txt')

    ### Prediction
    if '_neg' in _data:
        none_label_index = find_none_index(indir + '/type.txt')
        predict(indir, outdir, _method, _sim_func, _threshold, output, none_label_index)
    else:
        predict(indir, outdir, _method, _sim_func, _threshold, output, None)

    ### Evluate embedding predictions
    predictions = load_labels(output)
    print 'Predicted labels (embedding):'
    if '_neg' in _data:
        none_label_index = find_none_index(indir + '/type.txt')
        print '%f\t%f\t%f\t' % evaluate_rm_neg(predictions, ground_truth, none_label_index)
    else:
        print '%f\t%f\t%f\t' % evaluate_rm(predictions, ground_truth)
