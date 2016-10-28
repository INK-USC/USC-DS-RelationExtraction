__author__ = 'wenqihe'


class AbstractFeature(object):
    def apply(self, sentence, mention, features):
        raise NotImplementedError('Should have implemented this')
