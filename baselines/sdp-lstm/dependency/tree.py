# A tree structure that can read from conll-style data.
from collections import Counter
from collections import OrderedDict

WORD_FIELD = 'token'
DEP_HEAD_FIELD = 'stanford_head'
DEP_REL_FIELD = 'stanford_deprel'
SUBJECT_FIELD = 'subj'
OBJECT_FIELD = 'obj'
LABEL_FIELD = 'label'

SUBJECT_NAME = 'SUBJECT'
OBJECT_NAME = 'OBJECT'

# after tacred v1, root node becomes 0 instead of -1
ROOT_HEAD_ID = 0

class Node:  # a node in the tree
    def __init__(self, idx):
        self.index = idx
        #self.fields = {}
        self.parent = None  # reference to parent
        self.children = []  # reference to children
        self.rel = None # dependency relation to parenti, or ROOT
        self.level = None

class Tree:
    def __init__(self, conll_data):
        ''' Input conll_data should be a dictionary that maps fields to lists/sequences. '''
        assert(WORD_FIELD in conll_data)
        assert(DEP_HEAD_FIELD in conll_data)
        assert(DEP_REL_FIELD in conll_data)
        self.all_fields = conll_data

        # construct the tree from conll fields
        self.root, self.idx2node = self.parse_from_conll(conll_data)
        if self.root is None:
            raise Exception('No root found for tree: ' + str(conll_data))
        self.num_nodes = len(self.idx2node)
        assign_node_levels(self.root, 0)

        # find subjects and objects
        self.subjects_by_idx = []
        self.objects_by_idx = []
        for i in range(self.num_nodes):
            if self.all_fields[SUBJECT_FIELD][i] == 'SUBJECT':
                self.subjects_by_idx.append(i)
            if self.all_fields[OBJECT_FIELD][i] == 'OBJECT':
                self.objects_by_idx.append(i)
        self.subject_head_idx = get_entity_head(self.subjects_by_idx, self.idx2node) # 0-based idx of head of subject
        self.object_head_idx = get_entity_head(self.objects_by_idx, self.idx2node) # 0-based idx of the head of object

    def parse_from_conll(self, conll_fields):
        ''' Recover the tree structure from the conll field map. Return the root node.'''
        dep_head_seq = [int(x) for x in conll_fields[DEP_HEAD_FIELD]]
        dep_rel_seq = conll_fields[DEP_REL_FIELD]
        num_nodes = len(dep_head_seq)

        # create nodes and connect them
        idx2node = [Node(x) for x in range(0, num_nodes)]
        root = None
        for i, n in enumerate(idx2node):
            head = dep_head_seq[i] # note that head is 1-based indexing, except for ROOT, which is 0
            if head > num_nodes: # invalid head numbers
                raise Exception('Invalid dependency head: head id %d, total node %d.' % (head, num_nodes))
            if head == ROOT_HEAD_ID:
                n.parent = None
                root = n
            else:
                n.parent = idx2node[head-1] # set parent of current node
                idx2node[head-1].children.append(n) # set children of parent node
            n.rel = dep_rel_seq[i]
        return root, idx2node

    def get_shortest_path_through_root(self):
        ''' We define shortest path through root as always going from subject past the root to object.
        '''
        subject_to_root, root_idx1 = get_path_to_root(self.idx2node[self.subject_head_idx])
        object_to_root, root_idx2 = get_path_to_root(self.idx2node[self.object_head_idx])
        # sanity check
        if len(subject_to_root) == 0 or len(object_to_root) == 0:
            raise Exception('Entity to root path is empty.')
        assert(root_idx1 == root_idx2)
        # reverse object_to_root and append to subject_to_path
        shortest_path_through_root = subject_to_root + object_to_root[:-1][::-1] # the second seq should not include root
        return shortest_path_through_root, root_index

    def get_shortest_path_through_ancestor(self):
        ''' This is the shortest path between two nodes through common ancestor. '''
        subject_to_root, root_idx1 = get_path_to_root(self.idx2node[self.subject_head_idx])
        object_to_root, root_idx2 = get_path_to_root(self.idx2node[self.object_head_idx])
        # sanity check
        if len(subject_to_root) == 0 or len(object_to_root) == 0:
            raise Exception('Entity to root path is empty.')
        assert(root_idx1 == root_idx2)
        # find common ancestor by comparing path to root
        #print 'Root: %d' % root_idx1
        #print 'Subj: %d' % self.subject_head_idx
        #print 'Obj: %d' % self.object_head_idx
        #print subject_to_root[::-1]
        #print object_to_root[::-1]
        ancestor_idx = get_common_ancestor(subject_to_root[::-1], object_to_root[::-1])
        #print ancestor_idx
        subject_to_ancestor = get_path_to_node(self.idx2node[self.subject_head_idx], ancestor_idx)
        object_to_ancestor = get_path_to_node(self.idx2node[self.object_head_idx], ancestor_idx)
        assert(subject_to_ancestor[-1] == object_to_ancestor[-1])
        shortest_path = subject_to_ancestor + object_to_ancestor[:-1][::-1]
        return shortest_path, ancestor_idx

    def copy_fields_at_index(self, idx, target_fields):
        for k,v in self.all_fields.iteritems():
            if k == LABEL_FIELD:
                continue
            target_fields[k].append(v[idx])
        return

    def __repr__(self):
        root_word = self.all_fields[WORD_FIELD][self.root.index]
        s = "Tree: Root at %s(%d), " % (root_word, self.root.index)
        s += "with children: "
        for c in self.root.children:
            s += "%s(%d), " % (self.all_fields[WORD_FIELD][c.index], c.index)
        s = s[:-2]
        return s

def assign_node_levels(node, level):
    node.level = level
    for c in node.children:
        assign_node_levels(c, level+1)
    return

def get_entity_head(entity_idx_seq, idx2node):
    if len(entity_idx_seq) == 0:
        raise Exception('Input entity index sequence is empty when searching for entity head!')
    elif len(entity_idx_seq) == 1:
        return entity_idx_seq[0]
    else:
        levels = []
        top_level = 999
        top_level_node = 0
        for idx in entity_idx_seq:
            cur_level = idx2node[idx].level
            levels.append(cur_level)
            if top_level > cur_level:
                top_level = cur_level
                top_level_node = idx
        # check if there are two nodes at the same top level
        # ctr = Counter(levels)
        # if ctr[top_level] > 1:
            # print "Mulitple (%d) entity heads found." % ctr[top_level]
        return top_level_node

def get_path_to_root(node):
    ''' Return path to root by idx as a list, and root index. '''
    path_to_root = []
    n = node
    while n.parent is not None:
        path_to_root.append(n.index)
        n = n.parent
    path_to_root.append(n.index)
    return path_to_root, n.index

def get_path_to_node(current_node, target_node_idx):
    n = current_node
    if n.index == target_node_idx:
        return [n.index]
    path = []
    while n is not None and n.index != target_node_idx:
        path.append(n.index)
        n = n.parent
    # either n.index == target_node_idx or n == None
    if n.index == target_node_idx:
        path.append(n.index) # add the target
    else:
        raise Exception('Target node is not an ancestor of current node.')
    return path

def get_common_ancestor(root_to_node1, root_to_node2):
    i = 0
    min_len = min(len(root_to_node1), len(root_to_node2))
    while i < min_len and root_to_node1[i] == root_to_node2[i]:
        i += 1
    if i == 0:
        raise Exception('Error: the root of two root-to-node sequence does not match!')
    return root_to_node1[i-1] # return common ancestor
