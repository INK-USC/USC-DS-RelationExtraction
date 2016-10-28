__author__ = 'wenqihe'

import os
import operator
import sys
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf8')

class PruneStrategy:
    def __init__(self, strategy):
        self._strategy = strategy
        self.pruner = self.no_prune

    def no_prune(self, fileid, is_ground, labels):
        new_labels = set(labels)
        return list(new_labels)

def prune(indir, outdir, strategy, feature_number, type_number, neg_label_weight, isRelationMention, emDir):
    prune_strategy = PruneStrategy(strategy=strategy)

    type_file = open((os.path.join(indir+'/type.txt')), 'r')
    negLabelIndex = -1
    for line in type_file:
        seg = line.strip('\r\n').split('\t')
        if seg[0] == "None":
            negLabelIndex = int(seg[1])
            print "neg label : ", negLabelIndex
            break

    mids = {}
    ground_truth = set()
    count = 0
    train_y = os.path.join(indir+'/train_y.txt')
    train_x = os.path.join(indir+'/train_x_new.txt')
    test_x = os.path.join(indir+'/test_x.txt')
    test_y = os.path.join(indir+ '/test_y.txt')
    mention_file = os.path.join(outdir+ '/mention.txt')
    mention_type = os.path.join(outdir+ '/mention_type.txt')
    mention_feature = os.path.join(outdir+ '/mention_feature.txt')
    mention_type_test = os.path.join(outdir+'/mention_type_test.txt')
    mention_feature_test = os.path.join(outdir+ '/mention_feature_test.txt')
    feature_type = os.path.join(outdir+ '/feature_type.txt')
    # generate mention_type, and mention_feature for the training corpus
    with open(train_x) as fx, open(train_y) as fy, open(test_y) as ft, \
        open(mention_type,'w') as gt, open(mention_feature,'w') as gf:
        for line in ft:
            seg = line.strip('\r\n').split('\t')
            ground_truth.add(seg[0])
        # generate mention_type and mention_feature
        for line in fy:
            line2 = fx.readline()
            seg = line.strip('\r\n').split('\t')
            seg_split = seg[0].split('_')
            fileid = '_'.join(seg_split[:-3])
            labels = [int(x) for x in seg[1].split(',')]
            new_labels = prune_strategy.pruner(fileid=fileid, is_ground=(seg[0] in ground_truth), labels=labels)
            if new_labels is not None:
                seg2 = line2.strip('\r\n').split('\t')
                if len(seg2) != 2:
                    print seg2
                features = seg2[1].split(',')
                if seg[0] in mids:
                    continue
                for l in new_labels:
                    if l == negLabelIndex:  # discount weight for None label (index is 1)
                        gt.write(str(count)+'\t'+str(l)+'\t' + str(neg_label_weight) + '\n')
                    else:
                        gt.write(str(count)+'\t'+str(l)+'\t1\n')
                for f in features:
                    gf.write(str(count)+'\t'+f+'\t1\n')
                mids[seg[0]] = count
                count += 1
                if count%200000==0:
                    print count
    # generate mention_type_test, and mention_feature_test for the test corpus
    print count
    print 'start test'
    with open(test_x) as fx, open(test_y) as fy,\
        open(mention_type_test,'w') as gt, open(mention_feature_test, 'w') as gf:
        # generate mention_type and mention_feature
        for line in fy:
            line2 = fx.readline()
            seg = line.strip('\r\n').split('\t')
            try:
                labels = [int(x) for x in seg[1].split(',')]
            except:
                labels = [] ### if it's negative example (no type label), make it a []
            seg2 = line2.strip('\r\n').split('\t')
            features = seg2[1].split(',')
            if seg[0] in mids:
                mid = mids[seg[0]]
            else:
                mid = count
               # print line2
                mids[seg[0]] = count
                count += 1
            for l in labels:
                gt.write(str(mid)+'\t'+str(l)+'\t1\n')
            for f in features:
                gf.write(str(mid)+'\t'+f+'\t1\n')
    print count
    print 'start mention part'
    # generate mention.txt
    with open(mention_file,'w') as m:
        sorted_mentions = sorted(mids.items(), key=operator.itemgetter(1))
        for tup in sorted_mentions:
            m.write(tup[0]+'\t'+str(tup[1])+'\n')
    if isRelationMention:
        entity_mention_file = os.path.join(emDir+ '/mention.txt')
        triples_file = os.path.join(outdir+ '/triples.txt')
        with open(entity_mention_file, 'r') as emFile, open(triples_file, 'w') as triplesFile:
            emIdByString ={}
            for line in emFile.readlines():
                seg = line.strip('\r\n').split('\t')
                emIdByString[seg[0]] = seg[1]
            for tup in sorted_mentions:
                seg = tup[0].split('_')
                em1id = emIdByString['_'.join(seg[:-2])]
                em2id = emIdByString['_'.join(seg[:2]+seg[-2:])]
                rmid = tup[1]
                triplesFile.write(em1id+'\t'+em2id+'\t'+str(rmid)+'\n')

    print 'start feature_type part'
    with open(mention_feature) as f1, open(mention_type) as f2,\
        open(feature_type,'w') as g:
        fm = defaultdict(set)
        tm = defaultdict(set)
        for line in f1:
            seg = line.strip('\r\n').split('\t')
            i = int(seg[0])
            j = int(seg[1])
            fm[j].add(i)
        for line in f2:
            seg = line.strip('\r\n').split('\t')
            i = int(seg[0])
            j = int(seg[1])
            tm[j].add(i)
        for i in xrange(feature_number):
            for j in xrange(type_number):
                if j == negLabelIndex:  ### discount weight for None label "1"
                    temp = len(fm[i]&tm[j]) * neg_label_weight
                else:
                    temp = len(fm[i]&tm[j])
                if temp > 0:
                    g.write(str(i)+'\t'+str(j)+'\t'+str(temp)+'\n')
