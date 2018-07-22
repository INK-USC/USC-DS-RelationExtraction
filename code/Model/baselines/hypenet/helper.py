from collections import defaultdict
import json
import numpy as np
from sklearn.model_selection import train_test_split

WordNet_44_categories = [["0"],
                         ["B-adj.all", "B-adj.pert", "B-adj.ppl", 
                          "B-adv.all", "I-adj.all", "I-adv.all"],
                         ["B-noun.Tops", "I-noun.Tops"],
                         ["B-noun.act", "I-noun.act"],
                         ["B-noun.animal", "I-noun.animal"],
                         ["B-noun.artifact", "I-noun.artifact"],
                         ["B-noun.attribute", "I-noun.attribute"],
                         ["B-noun.body", "I-noun.body"],
                         ["B-noun.cognition", "I-noun.cognition"],
                         ["B-noun.communication", "I-noun.communication"],
                         ["B-noun.event", "I-noun.event"],
                         ["B-noun.feeling", "I-noun.feeling"],
                         ["B-noun.food", "I-noun.food"],
                         ["B-noun.group", "I-noun.group"],
                         ["B-noun.location", "I-noun.location"],
                         ["B-noun.motive", "I-noun.motive"],
                         ["B-noun.object", "I-noun.object"],
                         ["B-noun.other", "I-noun.other"],
                         ["B-noun.person", "I-noun.person"],
                         ["B-noun.phenomenon", "I-noun.phenomenon"],
                         ["B-noun.plant", "I-noun.plant"],
                         ["B-noun.possession", "I-noun.possession"],
                         ["B-noun.process", "I-noun.process"],
                         ["B-noun.quantity", "I-noun.quantity"],
                         ["B-noun.relation", "I-noun.relation"],
                         ["B-noun.shape", "I-noun.shape"],
                         ["B-noun.state", "I-noun.state"],
                         ["B-noun.substance", "I-noun.substance"],
                         ["B-noun.time", "I-noun.time"],
                         ["B-verb.body", "I-verb.body"],
                         ["B-verb.change", "I-verb.change"],
                         ["B-verb.cognition", "I-verb.cognition"],
                         ["B-verb.communication", "I-verb.communication"],
                         ["B-verb.competition", "I-verb.competition"],
                         ["B-verb.consumption", "I-verb.consumption"],
                         ["B-verb.contact", "I-verb.contact"],
                         ["B-verb.creation", "I-verb.creation"],
                         ["B-verb.emotion", "I-verb.emotion"],
                         ["B-verb.motion", "I-verb.motion"],
                         ["B-verb.perception", "I-verb.perception"],
                         ["B-verb.possession", "I-verb.possession"],
                         ["B-verb.social", "I-verb.social"],
                         ["B-verb.stative", "I-verb.stative"],
                         ["B-verb.weather", "I-verb.weather"]]

POS_15_categories = [["NN", "NNS", "NNP", "NNPS"],
                     ["IN"],  
                     ["VBN"], 
                     ["VBD"], 
                     ["VBZ"],
                     ["VBG"], 
                     ["VBP"], 
                     ["VB"],  
                     ["TO"],  
                     ["JJ", "JJR", "JJS"],  
                     ["RB", "RBR", "RBS"],  
                     ["CD"],  
                     ["DT", "PDT"],  
                     ["PRP"], 
                     ["RP"]]

GR_19_categories = [["dep"],
                    ["aux"],
                    ["auxpass"], 
                    ["cop"],
                    ["comp", "acomp", "attr", "ccomp", "xcomp", "pcomp",
                             "compl", "complm", "mark", "rel"],
                    ["dobj"], 
                    ["iobj"], 
                    ["pobj"],
                    ["nsubj"],
                    ["nsubjpass"],
                    ["csubj"],
                    ["csubjpass"],
                    ["cc"],
                    ["conj"],
                    ["expl"],
                    ["mod", "abbrev", "amod", "appos", "advcl", "purpcl",
                            "det", "predet", "pred", "preconj", "infmod", 
                            "partmod", "advmod", "neg", "rcmod", "quantmod", 
                            "tmod", "measure", "nn", "num", "number", "prep", 
                            "poss", "possessive", "prt"],
                    ["parataxis"],
                    ["ref"],
                    ["sdep", "xsubj"]]


def lst_2_dic(lst):
    """
    Input list：
        GR_19_categories = [["dep"],
                            ["aux"],
                            ["auxpass"], 
                            ["cop"],
                            ["comp", "acomp", "attr", "ccomp", "xcomp", "pcomp",
                             "compl", "complm", "mark", "rel"],
                             ...]
    Output dic:
        dic = { "dep" : 1,
                "aux" : 2,
                "auxpass" : 3,
                "cop" : 4,
                "comp" : 5,
                "acomp" : 5,
                "attr" : 5,
                ...}
    """
    dic = {}
    for i in range(len(lst)):
        for ele in lst[i]:
            dic[ele] = i+1
    return dic


def sequence_from_dic(lst, dic):
    res = []
    dic = defaultdict(int, **dic)
    for row in lst:
        seq = [dic[w] for w in row]
        res.append(seq)
    return res


def train_val_test_split(X, id_file=None):
    with open(id_file) as json_file:
        id_list = json.load(json_file)
        train_id_list = id_list[0]
        val_id_list = id_list[1]
        test_id_list = id_list[2]
        X_train = np.array([x for i, x in enumerate(X) if i in train_id_list])
        X_tuning = np.array([x for i, x in enumerate(X) if i in val_id_list])
        X_test = np.array([x for i, x in enumerate(X) if i in test_id_list])
        return X_train, X_tuning, X_test


def train_val_test_split_auto(X, y):
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, train_id_list, test_id_list = train_test_split(X, y, indices, stratify=y, test_size=1000,  random_state=123123)
    X_train, X_tuning, y_train, y_tuning,train_id_list, tuning_id_list = train_test_split(X_train, y_train,train_id_list, stratify=y_train, test_size=1000, random_state=123123)


def get_none_id(type_filename):
    with open(type_filename, encoding='utf-8') as type_file:
        for line in type_file:
            ls = line.strip().split()
            if ls[0] == "None":
                return int(ls[1])


def get_class_num(type_filename):
    num_classes = 0
    with open(type_filename, encoding='utf-8') as type_file:
        for line in type_file:
            ls = line.strip().split()
            if len(ls) == 2:
                num_classes += 1
    return num_classes


def evaluate_rm_neg(prediction, ground_truth, none_label_index):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:
    """
    # print '[None] label index:', none_label_index

    pos_pred = 0.0
    pos_gt = 0.0
    true_pos = 0.0
    for i in range(len(ground_truth)):
        if ground_truth[i] != none_label_index:
            pos_gt += 1.0

    for i in range(len(prediction)):
        if prediction[i] != none_label_index:
            # classified as pos example (Is-A-Relation)
            pos_pred += 1.0
            if prediction[i] == ground_truth[i]:
                true_pos += 1.0

    precision = true_pos / (pos_pred + 1e-8)
    recall = true_pos / (pos_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1

