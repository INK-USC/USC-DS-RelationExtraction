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


class EMHeadFeature(AbstractFeature):

    @staticmethod
    def get_head(sentence, mention):
        head = mention.end - 1
        for i in xrange(mention.start, mention.end):
            pt = sentence.pos[i]
            if pt.startswith('N'):
                head = i
            elif pt == 'IN' or pt == ',':
                break
        return head

    def apply(self, sentence, mention, features):
        index = EMHeadFeature.get_head(sentence, mention)
        head = sentence.tokens[index]
        pos = sentence.pos[index]
        features.append('HEAD_%s' % get_lemma(head, pos))


class EMTokenFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        for i in xrange(mention.start, mention.end):
            features.append('TKN_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))


class EMContextFeature(AbstractFeature):

    def __init__(self, window_size=1):
        self.window_size = window_size

    def apply(self, sentence, mention, features):
        # left
        for i in xrange(max(0, mention.start-self.window_size), mention.start):
            features.append('CTXT_LEFT_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))
        # right
        for i in xrange(mention.end, min(sentence.size(), mention.end+self.window_size)):
            features.append('CTXT_RIGHT_%s' % get_lemma(sentence.tokens[i], sentence.pos[i]))


class EMContextGramFeature(AbstractFeature):

    def __init__(self, window_size=1):
        self.window_size = window_size

    def apply(self, sentence, mention, features):
        start = max(0, mention.start-self.window_size)
        end = min(sentence.size()-1, mention.end - 1 + self.window_size)
        for i in xrange(start, end):
            token1 = get_lemma(sentence.tokens[i], sentence.pos[i])
            token2 = get_lemma(sentence.tokens[i+1], sentence.pos[i+1])
            if mention.start <= i < mention.end - 1:
                features.append('GRM_%s_%s'%(token1, token2))
            elif i < mention.start:
                features.append('CTXT_LEFT_GRM_%s_%s' % (token1, token2))
            else:
                features.append('CTXT_RIGHT_GRM_%s_%s' % (token1, token2))
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
