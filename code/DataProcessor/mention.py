__author__ = 'ZeqiuWu'


class RelationMention(object):
    """
    Wrap a relation mention. Each entity mention text of the relation mention is sentence.tokens[start:end].
    Attributes
    ==========
    em1Start : int
        The start index of the first entity mention.
    em1nd : int
        The end index of the first entity mention.
    em2Start : int
        The start index of the second entity mention.
    em2End : int
        The end index of the second entity mention.
    labels : string
        The label.
    """
    def __init__(self, em1Start, em1End, em2Start, em2End, numOfEMBetween, labels):
        self.em1Start = em1Start
        self.em1End = em1End
        self.em2Start = em2Start
        self.em2End = em2End
        self.numOfEMBetween = numOfEMBetween
        self.labels = labels

    def __str__(self):
        result = 'EM1 : start: %d, end: %d ; EM2 : start: %d, end: %d\n' % (self.em1Start, self.em1End, self.em2Start, self.em2End)
        for label in self.labels:
            result += label
        return result

class EntityMention(object):
    """
    Wrap a relation mention. Each entity mention text of the relation mention is sentence.tokens[start:end].
    Attributes
    ==========
    em1Start : int
        The start index of the first entity mention.
    em1nd : int
        The end index of the first entity mention.
    em2Start : int
        The start index of the second entity mention.
    em2End : int
        The end index of the second entity mention.
    labels : string
        The label.
    """
    def __init__(self, start, end, labels):
        self.start = start
        self.end = end
        self.labels = labels

    def __str__(self):
        result = 'start: %d, end: %d\n' % (self.start, self.end)
        for label in self.labels:
            result += label
        return result


class Sentence(object):
    """
    Wrap a sentence.
    Attributes
    ==========
    articleId : string
        The article id.
    sentid : string
        The sentence id.
    tokens : list
        The token list of this sentence.
    """
    def __init__(self, articleId, sentId, tokens):
        self.articleId = articleId
        self.sentId = sentId
        self.tokens = tokens
        self.entityMentions = []
        self.relationMentions = []
        self.pos = []

    def __str__(self):
        result = 'articleId: %s, sentId: %s\n'%(self.articleId, self.sentId)
        for token in self.tokens:
            result += token + ' '
        result += '\n'
        for m in self.mentions:
            result += m.__str__() + '\n'
        return result

    def add_relationMention(self, relationMention):
        assert isinstance(relationMention, RelationMention)
        self.relationMentions.append(relationMention)

    def add_entityMention(self, entityMention):
        assert isinstance(entityMention, EntityMention)
        self.entityMentions.append(entityMention)

    def size(self):
        return min(len(self.tokens),len(self.pos))


