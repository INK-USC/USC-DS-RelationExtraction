import sys
import os

fi = open(sys.argv[1], 'r')
fo = open(sys.argv[2], 'w')

for line in fi:
	lst = line.split('::')
	fo.write(lst[2])

fi.close()
fo.close()
