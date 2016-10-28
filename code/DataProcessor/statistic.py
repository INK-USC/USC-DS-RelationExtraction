__author__ = 'wenqihe'
import json
import sys
from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf8')


def distribution(indir):
    with open(indir+'/train_y.txt') as f, \
         open(indir+'/distribution_per_doc.txt','w') as g:
        d = defaultdict(dict)
        for line in f:
            sent = line.strip('\r\n').split('\t')
            fileid = sent[0].split('_')
            fileid = '_'.join(fileid[:-3])
            labels = sent[1].split(',')
            for index in labels:
                if index in d[fileid]:
                    d[fileid][index] +=1
                else:
                    d[fileid][index] =1
        for key in d:
            labels = [i for i in d[key] if d[key][i] >=2]
            if len(labels)>0:
                g.write(key+'\t'+",".join(labels)+'\n') 


def supertype(indir):
    with open(indir+'/type.txt') as f, \
         open(indir+'/supertype.txt','w') as g:
        mm = {}
        for line in f:
            seg = line.strip('\r\n').split('\t')
            mm[seg[0]] = seg[1] 

        for key1 in mm:
            for key2 in mm:
                if key1!=key2:
                    seg1 = key1[1:].split('/')
                    seg2 = key2[1:].split('/')
                    if len(seg1)==len(seg2)+1:
                        flag = True
                        for k in xrange(len(seg2)):
                            if seg1[k]!=seg2[k]:
                                flag = False
                                break
                        if flag:
                            g.write(mm[key1]+'\t'+mm[key2]+'\n')


    
