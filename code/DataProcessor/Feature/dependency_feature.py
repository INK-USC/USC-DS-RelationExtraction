__author__ = 'wenqihe'

from abstract_feature import AbstractFeature
from token_feature import HeadFeature, get_lemma


class DependencyFeature(AbstractFeature):
    accepted_deps=[ "nn", "agent", "dobj", "nsubj", "amod", "nsubjpass", "poss", "appos"]

    """
    Universal Dependencies
    """
    def apply(self, sentence, mention, features):
        # head_index = HeadFeature.get_head(sentence, mention)
        # for dep_type, gov, dep in sentence.dep:
        #     if head_index == gov:
        #         token = 'root'
        #         if dep >= 0:
        #             token = get_lemma(sentence.tokens[dep], sentence.pos[dep])
        #         features.append('ROLE_gov:%s' % dep_type)
        #         features.append('PARENT_%s' % token)
        #     if head_index == dep:
        #         token = 'root'
        #         if gov >= 0:
        #             token = get_lemma(sentence.tokens[dep], sentence.pos[gov])
        #         features.append('ROLE_dep:%s' % dep_type)
        #         features.append('PARENT_%s' % token)
        start = mention.start
        end = mention.end
        for dep_type, gov, dep in sentence.dep:
            if start <= gov < end:
                if 0 <= dep <sentence.size():
                    token = get_lemma(sentence.tokens[dep], sentence.pos[dep])
                    pos = sentence.pos[dep]
                    if self.accept_pos(pos) and self.accept_dep(dep_type):
                        key = "gov:" + dep_type + ":" + token + "=" + pos[0]
                        features.append(("DEP_" + key))
            if start <= dep < end:
                if 0 <= gov < sentence.size():
                    token = get_lemma(sentence.tokens[gov], sentence.pos[gov])
                    pos = sentence.pos[gov]
                    if self.accept_pos(pos) and self.accept_dep(dep_type):
                        key = "dep:" + dep_type + ":" + token + "=" + pos[0]
                        features.append(("DEP_" + key))

    def accept_pos(self, pos):
        return pos[0] == 'N' or pos[0] == 'V'

    def accept_dep(self, dep):
        return dep.startswith('prep') or dep in self.accepted_deps

