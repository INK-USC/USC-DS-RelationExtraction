import sys
import os
import json

fi = open(sys.argv[1], 'r')
fo = open(sys.argv[2], 'w')

for line in fi:
	dic = json.loads(line.strip())
	tk_lst = dic["tokens"]

	for tk in tk_lst:
		fo.write(tk + ' ')
	fo.write('\n')
fi.close()
fo.close()
