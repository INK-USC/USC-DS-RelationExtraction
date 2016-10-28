__author__ = 'xiang'

from collections import defaultdict

class TypeSet:
    def __init__(self, file_name, number_of_types):
        self._type_hierarchy = {} # type -> [parent type]
        self._subtype_mapping = defaultdict(list) # type -> [subtype]
        self._root = set() # root types (on 1-level)
        with open(file_name) as f:
            for line in f:
                try:
                    type, tid, freq = line.strip('\r\n').split('\t')
                    self._root.add(int(tid))
                except Exception as e:
                    print e
                    pass
        #self._root = list(set(range(0, number_of_types)).difference(self._root))

    def get_type_path(self, label):
        if label in self._type_hierarchy:  # label has super type
            path = [label]
            while label in self._type_hierarchy:
                path.append(self._type_hierarchy[label])
                label = self._type_hierarchy[label]
            path.reverse()
            return path
        else:  # label is the root type
            return [label]

    def get_subtypes(self, label):
        if label in self._subtype_mapping:
            return self._subtype_mapping[label]
        else:
            return None

class TypeHierarchy:
    def __init__(self, file_name, number_of_types):
        self._type_hierarchy = {} # type -> [parent type]
        self._subtype_mapping = defaultdict(list) # type -> [subtype]
        self._root = set() # root types (on 1-level)
        with open(file_name) as f:
            for line in f:
                t = line.strip('\r\n').split('\t')
                self._type_hierarchy[int(t[0])] = int(t[1])
                self._subtype_mapping[int(t[1])].append(int(t[0]))
                self._root.add(int(t[0]))
        self._root = list(set(range(0, number_of_types)).difference(self._root))

    def get_type_path(self, label):
        if label in self._type_hierarchy:  # label has super type
            path = [label]
            while label in self._type_hierarchy:
                path.append(self._type_hierarchy[label])
                label = self._type_hierarchy[label]
            path.reverse()
            return path
        else:  # label is the root type
            return [label]

    def get_subtypes(self, label):
        if label in self._subtype_mapping:
            return self._subtype_mapping[label]
        else:
            return None
