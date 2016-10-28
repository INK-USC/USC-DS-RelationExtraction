__author__ = 'ZeqiuWu'
import sys
import os
import math
from multiprocessing import Process, Lock
from nlp_parse import parse
#from postagger_parse import parse
from ner_feature import pipeline, filter, pipeline_test
from pruning_heuristics import prune
from statistic import supertype

def get_number(filename):
    with open(filename) as f:
        count = 0
        for line in f:
            count += 1
        return count

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print 'Usage:feature_generation.py -DATA -numOfProcesses -emtypeFlag(0 or 1) -negWeight (1.0)'
        exit(1)
    indir = 'data/source/%s' % sys.argv[1]
    if int(sys.argv[3]) == 1:
        outdir = 'data/intermediate/%s_emtype/rm' % sys.argv[1]
        requireEmType = True
    elif int(sys.argv[3]) == 0:
        outdir = 'data/intermediate/%s/rm' % sys.argv[1]
        requireEmType = False
    else:
        print 'Usage:feature_generation.py -DATA -numOfProcesses -emtypeFlag(0 or 1)'
        exit(1)
    outdir_em = 'data/intermediate/%s/em' % sys.argv[1]
    # NLP parse
    raw_train_json = indir + '/train.json'
    raw_test_json = indir + '/test.json'
    train_json = outdir + '/train_new.json'
    test_json = outdir + '/test_new.json'

    ### Generate features using Python wrapper (disabled if using run_nlp.sh)
    print 'Start nlp parsing'

    file = open(raw_train_json, 'r')
    sentences = file.readlines()
    numOfProcesses = int(sys.argv[2])
    sentsPerProc = int(math.floor(len(sentences)*1.0/numOfProcesses))
    lock = Lock()
    processes = []
    train_json_file = open(train_json, 'w', 0)

    for i in range(numOfProcesses):
        if i == numOfProcesses - 1:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:], train_json_file, lock, i, True))
        else:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:(i+1)*sentsPerProc], train_json_file, lock, i, True))
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join()
    train_json_file.close()

    print 'Train set parsing done'

    file = open(raw_test_json, 'r')
    numOfProcesses = int(sys.argv[2])
    sentences = file.readlines()
    sentsPerProc = int(math.floor(len(sentences)*1.0/numOfProcesses))
    processes = []
    lock = Lock()
    test_json_file = open(test_json, 'w', 0)
    for i in range(numOfProcesses):
        if i == numOfProcesses - 1:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:], test_json_file, lock, i, False))
        else:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:(i+1)*sentsPerProc], test_json_file, lock, i, False))
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join()

    test_json_file.close()
    print 'Test set parsing done'

    print 'Start em feature extraction'
    pipeline(train_json, indir + '/brown', outdir_em, requireEmType=requireEmType, isEntityMention=True)

    filter(outdir_em+'/feature.map', outdir_em+'/train_x.txt', outdir_em+'/feature.txt', outdir_em+'/train_x_new.txt')

    pipeline_test(test_json, indir + '/brown', outdir_em+'/feature.txt',outdir_em+'/type.txt', outdir_em, requireEmType=requireEmType, isEntityMention=True)
    supertype(outdir_em)

    ### Perform no pruning to generate training data
    print 'Start em training and test data generation'
    feature_number = get_number(outdir_em + '/feature.txt')
    type_number = get_number(outdir_em + '/type.txt')
    prune(outdir_em, outdir_em, 'no', feature_number, type_number, neg_label_weight=float(sys.argv[4]), isRelationMention=False, emDir='')

    print 'Start rm feature extraction'
    pipeline(train_json, indir + '/brown', outdir, requireEmType=requireEmType, isEntityMention=False)
    filter(outdir+'/feature.map', outdir+'/train_x.txt', outdir+'/feature.txt', outdir+'/train_x_new.txt')

    pipeline_test(test_json, indir + '/brown', outdir+'/feature.txt',outdir+'/type.txt', outdir, requireEmType=requireEmType, isEntityMention=False)

    ### Perform no pruning to generate training data
    print 'Start rm training and test data generation'
    feature_number = get_number(outdir + '/feature.txt')
    type_number = get_number(outdir + '/type.txt')
    prune(outdir, outdir, 'no', feature_number, type_number, neg_label_weight=float(sys.argv[4]), isRelationMention=True, emDir=outdir_em)

