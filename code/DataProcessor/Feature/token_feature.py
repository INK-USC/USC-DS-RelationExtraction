__author__ = 'wenqihe'

import re
from nltk.stem.wordnet import WordNetLemmatizer
from abstract_feature import AbstractFeature


cached = {}
lmtzr = WordNetLemmatizer()


def get_lemma(word, pos):
    key = word + '_' + pos
    if key in cached:
        return cached[key]
    if re.match('[a-zA-Z]+$', word) is None:
        cached[key] = word
        return word
    lemma = word
    if pos.startswith('N'):
        lemma = lmtzr.lemmatize(word, 'n')
    elif pos.startswith('V'):
        lemma = lmtzr.lemmatize(word, 'v')
    cached[key] = lemma
    return lemma


class HeadFeature(AbstractFeature):


    @staticmethod
    def get_head(sentence, start, end):
        head = end - 1
        for i in xrange(start, end):
            pt = sentence.pos[i]
            if pt.startswith('N'):
                head = i
            elif pt == 'IN' or pt == ',':
                break
        return head

    def apply(self, sentence, mention, features):
        em1index = HeadFeature.get_head(sentence, mention.em1Start, mention.em1End)
        em1head = sentence.tokens[em1index]
        em1pos = sentence.pos[em1index]
        features.append('HEAD_EM1_%s' % get_lemma(em1head, em1pos))
        em2index = HeadFeature.get_head(sentence, mention.em2Start, mention.em2End)
        em2head = sentence.tokens[em2index]
        em2pos = sentence.pos[em2index]
        features.append('HEAD_EM2_%s' % get_lemma(em2head, em2pos))



class EntityMentionTokenFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        for i in xrange(mention.em1Start, mention.em1End):
            features.append('TKN_EM1_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))
        for i in xrange(mention.em2Start, mention.em2End):
            features.append('TKN_EM2_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))

class BetweenEntityMentionTokenFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        start = mention.em1End
        end = mention.em2Start
        if mention.em1Start > mention.em2Start:
            start = mention.em2End
            end = mention.em1Start
        for i in xrange(start, end):
            if i == start:
                features.append('FIRST_TKN_BTWN_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))
            if i == (end - 1):
                features.append('LAST_TKN_BTWN_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))
            features.append('TKN_BTWN_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))

class ContextFeature(AbstractFeature):

    def __init__(self, window_size=1):
        self.window_size = window_size

    def apply(self, sentence, mention, features):
        # left
        for i in xrange(max(0, mention.em1Start-self.window_size), mention.em1Start):
            features.append('CTXT_EM1_LEFT_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))
        # right
        for i in xrange(mention.em1End, min(sentence.size(), mention.em1End+self.window_size)):
            features.append('CTXT_EM1_RIGHT_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))

        # left
        for i in xrange(max(0, mention.em2Start-self.window_size), mention.em2Start):
            features.append('CTXT_EM2_LEFT_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))
        # right
        for i in xrange(mention.em2End, min(sentence.size(), mention.em2End+self.window_size)):
            features.append('CTXT_EM2_RIGHT_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))


class ContextGramFeature(AbstractFeature):

    def __init__(self, window_size=1):
        self.window_size = window_size

    def apply(self, sentence, mention, features):
        start = max(0, mention.em1Start-self.window_size)
        end = min(sentence.size()-1, mention.em1End - 1 + self.window_size)
        for i in xrange(start, end):
            token1 = get_lemma(sentence.tokens[i], sentence.pos[i])
            token2 = get_lemma(sentence.tokens[i+1], sentence.pos[i+1])
            if mention.em1Start <= i < mention.em1End - 1:
                features.append('GRM_EM1_%s_%s'%(token1, token2))
            elif i < mention.em1Start:
                features.append('CTXT_EM1_LEFT_GRM_%s_%s' % (token1, token2))
            else:
                features.append('CTXT_EM1_RIGHT_GRM_%s_%s' % (token1, token2))

        start = max(0, mention.em2Start-self.window_size)
        end = min(sentence.size()-1, mention.em2End - 1 + self.window_size)
        for i in xrange(start, end):
            token1 = get_lemma(sentence.tokens[i], sentence.pos[i])
            token2 = get_lemma(sentence.tokens[i+1], sentence.pos[i+1])
            if mention.em2Start <= i < mention.em2End - 1:
                features.append('GRM_EM2_%s_%s'%(token1, token2))
            elif i < mention.em2Start:
                features.append('CTXT_EM2_LEFT_GRM_%s_%s' % (token1, token2))
            else:
                features.append('CTXT_EM2_RIGHT_GRM_%s_%s' % (token1, token2))
        # left
        # if mention.start-2 >= 0:
        #     token1 = get_lemma(sentence.tokens[mention.start-2], sentence.pos[mention.start-2])
        #     token2 = get_lemma(sentence.tokens[mention.start-1], sentence.pos[mention.start-1])
        #     features.append('CTXT_LEFT_GRM_%s_%s' % (token1, token2))
        # # right
        # if mention.end + 1 < len(sentence.tokens):
        #     token1 = get_lemma(sentence.tokens[mention.end], sentence.pos[mention.end])
        #     token2 = get_lemma(sentence.tokens[mention.end+1], sentence.pos[mention.end+1])
        #     features.append('CTXT_RIGHT_GRM_%s_%s' % (token1, token2))
