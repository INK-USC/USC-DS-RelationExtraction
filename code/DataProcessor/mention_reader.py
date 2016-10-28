__author__ = 'wenqihe'

import json
from mention import RelationMention, EntityMention, Sentence


class MentionReader:
    """
    Mention reader. Cache one sentence in advance.
    Attributes
    ==========
    mention_file : string
        mention file.
    current : Sentence
        current sentence.
    input : File
        input stream.
    """
    def __init__(self, mention_file):
        self.mention_file = mention_file
        self.input = open(mention_file, 'rb')
        self.current = self._decode(self.input.readline())

    def close(self):
        self.input.close()

    def has_next(self):
        """
        Check if there is more sentence to read.
        :return: true if there is more sentence to read
        """
        return self.current is not None

    def next(self):
        """
        :return: the next sentence object
        """
        result = self.current
        self.current = self._decode(self.input.readline())
        return result

    @staticmethod
    def _decode(mention_json):
        """
        Decode a json string of a sentence.
        e.g.,  {"senid":40,
                "mentions":[{"start":0,"end":2,"labels":["/person"]},
                            {"start":6,"end":8,"labels":["/location/city","/location"]}],
                "tokens":["Raymond","Jung",",","51",",","of","Federal","Way",";",
                         "accused","of","leasing","apartments","where","the","women",
                         "were","housed","."],
                "fileid":""}
        :param mention_json: string
        :return: a sentence instance with all mentions appearing in this sentence
        """
        if mention_json == '':
            return None
        # try:
        decoded = json.loads(mention_json)
        sentence = Sentence(decoded['articleId'], int(decoded['sentId']), decoded['tokens'])
        for rm in decoded['relationMentions']:
            #if len(rm['labels']) > 1:
                #print decoded['articleId'], decoded['sentId']
            sentence.add_relationMention(RelationMention(int(rm['em1Start']), int(rm['em1End']), int(rm['em2Start']), int(rm['em2End']), rm['numOfEMBetween'], rm['labels']))
        for em in decoded['entityMentions']:
            sentence.add_entityMention(EntityMention(int(em['start']), int(em['end']), em['labels']))
        if 'pos' in decoded:
            sentence.pos = decoded['pos']
        """
        if 'dep' in decoded:
            for dep in decoded['dep']:
                sentence.dep.append((dep['type'], dep['gov'], dep['dep']))
        """
        # except ValueError:
        #     print 'error in decodig JSON'
        #     print mention_json
        #     return None
        return sentence
