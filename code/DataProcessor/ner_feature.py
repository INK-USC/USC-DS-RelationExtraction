__author__ = 'wenqihe'

from Feature import *
import sys
from mention_reader import MentionReader
reload(sys)
sys.setdefaultencoding('utf8')

class NERFeature(object):

    def __init__(self, is_train, brown_file, requireEmType, isEntityMention, feature_mapping={}, label_mapping={}):
        self.is_train = is_train
        self.feature_count = 0
        self.label_count = 0
        self.feature_list = []
        self.feature_mapping = feature_mapping # {feature_name: [feature_id, feature_frequency]}
        self.label_mapping = label_mapping # {label_name: [label_id, label_frequency]}
        if isEntityMention:
            # head feature
            self.feature_list.append(EMHeadFeature())
            # token feature
            self.feature_list.append(EMTokenFeature())
            # context unigram
            self.feature_list.append(EMContextFeature(window_size=3))
            # context bigram
            self.feature_list.append(EMContextGramFeature(window_size=3))
            # pos feature
            self.feature_list.append(EMPosFeature())
            # word shape feature
            self.feature_list.append(EMWordShapeFeature())
            # length feature
            self.feature_list.append(EMLengthFeature())
            # character feature
            self.feature_list.append(EMCharacterFeature())
            # brown clusters
            self.feature_list.append(EMBrownFeature(brown_file))
            # dependency feature
            #self.feature_list.append(EMDependencyFeature())
        else:
            # head feature
            self.feature_list.append(HeadFeature())
            # token feature
            self.feature_list.append(EntityMentionTokenFeature())
            self.feature_list.append(BetweenEntityMentionTokenFeature())
            # context unigram
            self.feature_list.append(ContextFeature(window_size=3))
            # context bigram
            self.feature_list.append(ContextGramFeature(window_size=3))
            # pos feature
            self.feature_list.append(PosFeature())
            # word shape feature
            self.feature_list.append(EntityMentionOrderFeature())
            # length feature
            self.feature_list.append(DistanceFeature())
            # character feature
            self.feature_list.append(NumOfEMBetweenFeature())
            self.feature_list.append(SpecialPatternFeature())
            # brown clusters
            self.feature_list.append(BrownFeature(brown_file))
            if requireEmType:
                self.feature_list.append(EMTypeFeature())


    def extract(self, sentence, mention):
        # extract feature strings
        feature_str = []
        for f in self.feature_list:
            f.apply(sentence, mention, feature_str)
        #print ' '.join(sentence.tokens), feature_str
            # print f
        # map feature_names and label_names
        feature_ids = set()
        label_ids = set()
        for s in feature_str:
            if s in self.feature_mapping:
                feature_ids.add(self.feature_mapping[s][0])
                self.feature_mapping[s][1] += 1  # add frequency
            elif self.is_train:
                feature_ids.add(self.feature_count)
                self.feature_mapping[s] = [self.feature_count, 1]
                self.feature_count += 1
        #if (mention.labels) > 1:
            #print sentence.articleId, sentence.sentId
        for l in mention.labels:
            if l in self.label_mapping:
                label_ids.add(self.label_mapping[l][0])
                self.label_mapping[l][1] += 1  # add frequency
            elif self.is_train:
                label_ids.add(self.label_count)
                self.label_mapping[l] = [self.label_count, 1]
                self.label_count += 1

        return feature_ids, label_ids


def pipeline(json_file, brown_file, outdir, requireEmType, isEntityMention):
    reader = MentionReader(json_file)
    ner_feature = NERFeature(is_train=True, brown_file=brown_file, requireEmType=requireEmType, isEntityMention=isEntityMention, feature_mapping={}, label_mapping={})
    count = 0
    gx = open(outdir+'/train_x.txt', 'w')
    gy = open(outdir+'/train_y.txt', 'w')
    f = open(outdir+'/feature.map', 'w')
    t = open(outdir+'/type.txt', 'w')
    label_counts_file = open(outdir+'/label_counts.txt', 'w')
    print 'start train feature generation'
    mention_count = 0
    mentionCountByNumOfLabels = {}
    while reader.has_next():
        if count%10000 == 0:
            sys.stdout.write('process ' + str(count) + ' lines\r')
            sys.stdout.flush()
        sentence = reader.next()
        if isEntityMention:
            mentions = sentence.entityMentions
        else:
            mentions = sentence.relationMentions
        for mention in mentions:
            try:
                if isEntityMention:
                    m_id = '%s_%s_%d_%d'%(sentence.articleId, sentence.sentId, mention.start, mention.end)
                else:
                    m_id = '%s_%d_%d_%d_%d_%d'%(sentence.articleId, sentence.sentId, mention.em1Start, mention.em1End, mention.em2Start, mention.em2End)
                feature_ids, label_ids = ner_feature.extract(sentence, mention)
                if len(label_ids) not in mentionCountByNumOfLabels:
                    mentionCountByNumOfLabels[len(label_ids)] = 1
                else:
                    mentionCountByNumOfLabels[len(label_ids)] += 1
                gx.write(m_id+'\t'+','.join([str(x) for x in feature_ids])+'\n')
                gy.write(m_id+'\t'+','.join([str(x) for x in label_ids])+'\n')
                mention_count += 1
                count += 1
            except Exception as e:
                print e.message, e.args
                print sentence.articleId, sentence.sentId, len(sentence.tokens)
                print mention
                raise
    print '\n'
    print 'mention :%d'%mention_count
    print 'feature :%d'%len(ner_feature.feature_mapping)
    print 'label :%d'%len(ner_feature.label_mapping)
    sorted_map = sorted(mentionCountByNumOfLabels.items(),cmp=lambda x,y:x[0]-y[0])
    for item in sorted_map:
        label_counts_file.write(str(item[0])+'\t'+str(item[1])+'\n')
    write_map(ner_feature.feature_mapping, f)
    write_map(ner_feature.label_mapping, t)
    reader.close()
    gx.close()
    gy.close()
    f.close()
    t.close()


def pipeline_test(json_file, brown_file, featurefile, labelfile, outdir, requireEmType, isEntityMention):
    #  load feature mapping and label mapping
    feature_map = load_map(featurefile)
    label_map = load_map(labelfile)

    reader = MentionReader(json_file)
    ner_feature = NERFeature(is_train=False, brown_file=brown_file, requireEmType=requireEmType, isEntityMention=isEntityMention, feature_mapping=feature_map, label_mapping=label_map)
    count = 0
    gx = open(outdir+'/test_x.txt', 'w')
    gy = open(outdir+'/test_y.txt', 'w')

    print 'start test feature generation'
    while reader.has_next():
        if count%10000 == 0 and count != 0:
            sys.stdout.write('process ' + str(count) + ' lines\r')
            sys.stdout.flush()
        sentence = reader.next()
        if isEntityMention:
            mentions = sentence.entityMentions
        else:
            mentions = sentence.relationMentions
        for mention in mentions:
            try:
                if isEntityMention:
                    m_id = '%s_%s_%d_%d'%(sentence.articleId, sentence.sentId, mention.start, mention.end)
                else:
                    m_id = '%s_%d_%d_%d_%d_%d'%(sentence.articleId, sentence.sentId, mention.em1Start, mention.em1End, mention.em2Start, mention.em2End)
                #print mention.em1Start, mention.em1End, mention.em2Start, mention.em2End
                feature_ids, label_ids = ner_feature.extract(sentence, mention)
                gx.write(m_id+'\t'+','.join([str(x) for x in feature_ids])+'\n')
                gy.write(m_id+'\t'+','.join([str(x) for x in label_ids])+'\n')
                count += 1
            except Exception as e:
                print e.message, e.args
                print sentence.articleId, sentence.sentId
                print mention
                continue
    type_test = open(outdir+'/type_test.txt', 'w')
    write_map(ner_feature.label_mapping, type_test)
    print '\n'
    reader.close()
    gx.close()
    gy.close()


def load_map(input):
    f = open(input)
    mapping = {}
    for line in f:
        seg = line.strip('\r\n').split('\t')
        mapping[seg[0]] = [int(seg[1]), 0]
    f.close()
    return mapping


def write_map(mapping, output):
    sorted_map = sorted(mapping.items(),cmp=lambda x,y:x[1][0]-y[1][0])
    for tup in sorted_map:
        output.write(tup[0]+'\t'+str(tup[1][0])+'\t'+str(tup[1][1])+'\n')


def filter(featurefile, trainfile, featureout,trainout):
    f = open(featurefile)
    featuremap = {}
    old2new = {}
    count = 0
    for line in f:
        seg = line.strip('\r\n').split('\t')
        frequency = int(seg[2])
        if frequency>=1:
            featuremap[seg[0]] = (count,seg[2])
            old2new[seg[1]] = count
            count+=1
    print 'Feature after filter: %d'%count
    f.close()
    g = open(featureout,'w')
    write_map2(featuremap, g)
    g.close()

    # scan the training set and filter features
    f = open(trainfile)
    g = open(trainout,'w')
    for line in f:
        seg = line.strip('\r\n').split('\t')
        # features = line.strip('\r\n').split(',')
        features = seg[1].split(',')
        newfeatures = set()
        for feature in features:
            if feature in old2new:
                newfeatures.add(old2new[feature])
        g.write(seg[0]+'\t'+','.join([str(x) for x in newfeatures])+'\n')
        # g.write(','.join([str(x) for x in newfeatures])+'\n')

    f.close()
    g.close()


def write_map2(mapping, output):
    sorted_map = sorted(mapping.items(),cmp=lambda x,y:x[1][0]-y[1][0])
    for tup in sorted_map:
        output.write(tup[0]+'\t'+str(tup[1][0])+'\n')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print 'Usage:ner_feature.py -TRAIN_JSON -TEST_JSON -BROWN_FILE -OUTDIR'
        exit(1)
    train_json = sys.argv[1]
    test_json = sys.argv[2]
    brown_file = sys.argv[3]
    outdir = sys.argv[4]
    pipeline(train_json, brown_file, outdir)
    filter(featurefile=outdir+'/feature.map', trainfile=outdir+'/train_x.txt', featureout=outdir+'/feature.txt',trainout=outdir+'/train_x_new.txt')
    pipeline_test(test_json, brown_file, outdir+'/feature.txt',outdir+'/type.txt', outdir)
