import string
import json

fin = open('result.txt')
fout = open('pr.txt', 'w')

A = dict()

tot = 0
for line in fin:
	js = json.loads(line)
	rel = js['relation']
	scl = js['score']
	maxs = -1
	for idx, sc in enumerate(scl):
		if sc > maxs:
			maxs = sc
			maxid = idx
	A[str(rel)+'\t'+str(maxid)+'\t'+str(tot)] = maxs
	tot += 1

tp = 0.0
A_r = sorted(A.iteritems(), key=lambda d:d[1], reverse = True)
for idx, a_r in enumerate(A_r):
	tmp = a_r[0].split('\t')
	if tmp[0] == tmp[1]:
		tp += 1
	fout.write(str(tp/(idx+1))+'\t'+str(tp/tot)+'\n')

fin.close()
fout.close()

	