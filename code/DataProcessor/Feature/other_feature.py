__author__ = 'wenqihe'

import re
from abstract_feature import AbstractFeature
from token_feature import HeadFeature

class PosFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        start = mention.em1End
        end = mention.em2Start
        if mention.em1Start > mention.em2Start:
            start = mention.em2End
            end = mention.em1Start
        for i in xrange(start, end):
            features.append('POS_%s' % sentence.pos[i])


class DistanceFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        dist = mention.em2Start - mention.em1End
        if mention.em2Start < mention.em1Start:
            dist = mention.em1Start - mention.em2End
        features.append('DISTANCE_%d' % dist)

class EntityMentionOrderFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        if mention.em1Start < mention.em2Start:
            features.append('EM1_BEFORE_EM2')
        elif mention.em1Start > mention.em2Start:
            features.append('EM2_BEFORE_EM1')

class NumOfEMBetweenFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        numOfEMBetween = mention.numOfEMBetween
        features.append('NUM_EMS_BTWEEN_%d' % numOfEMBetween)

class EMTypeFeature(AbstractFeature):
    def apply(self, sentence, mention, features):
        for em in sentence.entityMentions:
            if em.start == mention.em1Start and em.end == mention.em1End:
                for l in em.labels:
                    features.append('EM1_TYPE_%s' % l)
            if em.start == mention.em2Start and em.end == mention.em2End:
                for l in em.labels:
                    features.append('EM2_TYPE_%s' % l)

class SpecialPatternFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        if mention.em1End + 1 == mention.em2Start:
            if sentence.tokens[mention.em1End] == 'in':
                features.append('EM1_IN_EM2')
        if mention.em2End + 1 == mention.em1Start:
            if sentence.tokens[mention.em2End] == 'in':
                features.append('EM2_IN_EM1')

