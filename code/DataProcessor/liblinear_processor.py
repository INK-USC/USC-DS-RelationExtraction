__author__ = 'xiang'
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def load_as_list(filename):
    """
    Load data as a list of list.
    e.g.[[0,1,2],[1,2]]
    """
    with open(filename) as f:
        data = []
        indexes = []
        line = f.readline()
        seg = line.strip('\r\n').split('\t')
        index = int(seg[0])
        features = [int(seg[1])]
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if index == int(seg[0]):  # Still in the same mention
                features.append(int(seg[1]))
            else:
                # Append to train_x
                data.append(sorted(features))
                indexes.append(index)
                features = [int(seg[1])]
                index = int(seg[0])
        if len(features) > 0:
            data.append(sorted(features))
            indexes.append(index)
        return indexes, data

def write_train_as_liblinear(train_x, train_y, filename):
    with open(filename, 'w') as f:
        for i in range(len(train_x)):
            label = str(train_y[i][0])
            f.write(label + ' ')
            tmp = []
            for feature in train_x[i]:
                tmp.append(str(feature + 1) + ':1.0')
            f.write(' '.join(tmp) + '\n')

def write_test_as_liblinear(test_x, filename):
    with open(filename, 'w') as f:
        for i in range(len(test_x)):
            f.write('-1 ')
            tmp = []
            for feature in test_x[i]:
                tmp.append(str(feature + 1) + ':1.0')
            f.write(' '.join(tmp) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print 'Usage: liblinear_processor.py -DATA(nyt_candidates)'
        exit(-1)

    indir = 'data/intermediate/' + sys.argv[1] + '/rm'

    train_x_file = indir + '/mention_feature.txt'
    train_y_file = indir + '/mention_type.txt'
    test_x_file = indir + '/mention_feature_test.txt'

    lib_train_file = indir + '/liblinear_train.txt'
    lib_test_file = indir + '/liblinear_test.txt'

    ### Train
    train_x = load_as_list(train_x_file)
    train_y = load_as_list(train_y_file)

    write_train_as_liblinear(train_x[1], train_y[1], lib_train_file)

    ### Test
    indexes, test_x = load_as_list(test_x_file)
    write_test_as_liblinear(test_x, lib_test_file)




