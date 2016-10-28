import sys
import os

label = {}
fi = open(sys.argv[1], 'r')
while True:
	l1 = fi.readline()
	l2 = fi.readline()
	l3 = fi.readline()
	l4 = fi.readline()
	if not l1:
		break
	lb = l1.strip().split()[0]
	label[lb] = 1
fi.close()

default_lb = label.keys()[0]

fi = open(sys.argv[2], 'r')
fo = open(sys.argv[3], 'w')
while True:
	l1 = fi.readline()
        l2 = fi.readline()
        l3 = fi.readline()
        l4 = fi.readline()
	if not l1:
                break
	pst = l1.find('\t')
	lb = l1[0:pst]
	if label.get(lb, None) == None:
		lb = default_lb
	fo.write(lb + l1[pst:])
	fo.write(l2)
	fo.write(l3)
	fo.write(l4)
fi.close()
fo.close()
	
