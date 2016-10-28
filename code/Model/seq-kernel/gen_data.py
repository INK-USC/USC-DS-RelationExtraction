import sys
import os

fil = open(sys.argv[1], 'r')
fiv = open(sys.argv[2], 'r')
fo = open(sys.argv[3], 'w')

line = fiv.readline()
n = int(line.split()[0])
d = int(line.split()[1])
for i in range(n):
	if i % 100 == 0:
		print i
	lb = fil.readline().strip()
	line = fiv.readline()
	lst = line.strip().split()
	fo.write(lb + ' 0:' + str(i + 1))
	for j in range(d):
		f = float(lst[j])
		fo.write(' ' + str(j + 1) + ':' + str(f))
	fo.write('\n')
	#exit(0)
fil.close()
fiv.close()
fo.close()
