import sys
import os

s2l = {}
fi = open(sys.argv[3], 'r')
for line in fi:
	s = line.split('\t')[0]
	l = line.split('\t')[1]
	s2l[s] = l
fi.close()

#print s2l

fi = open(sys.argv[1], 'r')
fo = open(sys.argv[2], 'w')
for line in fi:
	s = line.split('::')[1]
	l = s2l.get(s, '-1')
	if l == '-1':
		print s
	fo.write(l + '\n')
fi.close()
fo.close()

