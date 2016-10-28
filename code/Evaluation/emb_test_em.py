# Script to predict and evaluate in a pipeline
__author__ = 'xiang'

import sys
from collections import  defaultdict
from evaluation_em import *
from emb_prediction_em import *

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print 'Usage: emb_test.py -DATA(nyt_candidates) -METHOD(retypeRM) -SIM(cosine/dot) -THRESHOLD'
        exit(-1)

    # do prediction here
    _data = sys.argv[1]
    _method = sys.argv[2]
    _sim_func = sys.argv[3]
    _threshold = float(sys.argv[4])

    indir = 'data/intermediate/' + _data + '/em'
    outdir = 'data/results/' + _data + '/em'

    output = outdir +'/prediction_emb_' + _method + '_' + _sim_func + '.txt'
    ground_truth = load_labels_gt(indir + '/mention_type_test.txt', _data)

    ### Prediction
    if '_neg' in _data:
        none_label_index = find_none_index(indir + '/type.txt')
        if none_label_index is not None:
            raise Exception('entity mention type should not be none.')
        predict(indir, outdir, _method, _sim_func, _threshold, output, none_label_index)
    else:
        predict(indir, outdir, _method, _sim_func, _threshold, output, None)

    ### Evluate embedding predictions
    predictions = load_labels(output)
    print 'Predicted labels (embedding):'
    print '%f\t%f\t%f\t%f\t%f\t%f\t%f' % evaluate_em(predictions, ground_truth)
