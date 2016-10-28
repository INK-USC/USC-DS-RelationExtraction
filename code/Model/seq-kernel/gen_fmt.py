import sys
import os

s2i = {}
fi = open(sys.argv[1], 'r')
for line in fi:
	s = line.strip().split()[0]
	i = line.strip().split()[1]
	s2i[s] = i
fi.close()
print len(s2i)

fis = open(sys.argv[2], 'r')
fil = open(sys.argv[3], 'r')
fo = open(sys.argv[4], 'w')
line = fil.readline()
lst = line.strip().split()
dic = {}
for i in range(len(lst)):
	if i == 0:
		continue
	dic[lst[i]] = i
for line in fis:
	s = line.split('::')[0]
	i = s2i.get(s, '-1')
	if i == '-1':
		print s
	lst = fil.readline().split()
	if len(lst) > 1:
		l = lst[0]
		k = dic.get(l, -1)
		fo.write(i + '\t' + l + '\t' + lst[k] + '\n')
fis.close()
fil.close()
fo.close()
