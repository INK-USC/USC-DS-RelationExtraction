__author__ = 'wenqihe'

from abstract_feature import AbstractFeature
from em_token_feature import get_lemma


class EMBrownFeature(AbstractFeature):

    def __init__(self, brown_file):
        with open(brown_file) as f:
            self.len = [4, 8, 12, 20]
            self.mapping = {}
            for line in f:
                items = line.strip('\r\n').split('\t')
                self.mapping[items[1]] = items[0]

    def apply(self, sentence, mention, features):
        for i in xrange(mention.start,mention.end):
            word = get_lemma(sentence.tokens[i], sentence.pos[i])
            if word in self.mapping:
                cluster = self.mapping[word]
                for l in self.len:
                    if len(cluster) >= l:
                        features.append('BROWN_%d_%s' % (l, cluster[0:l]))
                features.append('BROWN_ALL_%s' % cluster)
