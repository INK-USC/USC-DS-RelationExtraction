from pycorenlp import StanfordCoreNLP
from pprint import pprint
import json

nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))


def get_stanford_annotations(text, port=9000,
                             annotators='tokenize,ssplit,pos,lemma,depparse,parse'):
    output = nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.isOneSentence": "true",
        'annotators': annotators,
    })
    return output

document = 'dadsa'
print('document: {0}'.format(document))

# Parse the text
annotations = get_stanford_annotations(document, port=9000,
                                       annotators='tokenize,ssplit,pos,lemma,depparse')

annotations = json.loads(annotations, encoding="utf-8", strict=False)
tokens = annotations['sentences'][0]["tokens"]
tokens_text = [token['originalText'] for token in tokens]
lemmas_text = [token['lemma'] for token in tokens]

# print(tokens_text)
tokens_line = " ".join(tokens_text)
lemmas_line = " ".join(lemmas_text)
print(tokens_line)
print(lemmas_line)
