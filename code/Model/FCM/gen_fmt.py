import sys
import os

sid2idx = {}
fi = open(sys.argv[1], 'r')
for line in fi:
	sid = line.strip().split()[0]
	idx = line.strip().split()[1]
	sid2idx[sid] = idx
fi.close()

label2idx = {}
fi = open(sys.argv[2], 'r')
for line in fi:
	label = line.split()[0]
	idx = line.split()[1]
	label2idx[label] = idx
fi.close()

fis = open(sys.argv[3], 'r')
fip = open(sys.argv[4], 'r')
fo = open(sys.argv[5], 'w')
for sid in fis:
	try:
		label = fip.readline().strip().split('\t')[1]
		sidx = sid2idx.get(sid.strip(), -1)
		lidx = label2idx.get(label, -1)
		if sidx == -1 or lidx == -1:
			print sid, label
			exit(0)
	except:
		print 'line parsing fail!'
		
	fo.write(str(sidx) + '\t' + str(lidx) + '\t' + '1' + '\n')
fis.close()
fip.close()
fo.close()
