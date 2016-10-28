import sys
import os
import json

fi = open(sys.argv[1], 'r')
fo = open(sys.argv[2], 'w')
foid = open(sys.argv[3], 'w')

for line in fi:
	dic = json.loads(line.strip())
	pos_lst = dic["pos"]
	tk_lst = dic["tokens"]
	rlt_lst = dic["relationMentions"]

	s = ''
	for tk in tk_lst:
		s = s + tk + ' '
	if len(s) > 990:
		continue

	aid = dic["articleId"]
	sid = dic["sentId"]

	for rlt in rlt_lst:

		e1p = int(rlt["em1Start"])
		e1q = int(rlt["em1End"]) - 1
		e2p = int(rlt["em2Start"])
		e2q = int(rlt["em2End"]) - 1

		#if e1p != e1q or e2p != e2q:
                #        continue

		strid = str(aid) + '_' + str(sid) + '_' + str(e1p) + '_' + str(e1q + 1) + '_' + str(e2p) + '_' + str(e2q + 1)
		foid.write(strid + '\n')

		e1s = ''
		for i in range(e1p, e1q + 1):
			e1s += tk_lst[i] + ' '
		e1s = e1s.strip()

		e2s = ''
		for i in range(e2p, e2q + 1):
			e2s += tk_lst[i] + ' '
		e2s = e2s.strip()
		
		lb = rlt["labels"][0]

		s = lb + '\t' + str(e1p) + '\t' + str(e1q) + '\t' + e1s + '\t' + str(e2p) + '\t' + str(e2q) + '\t' + e2s
		fo.write(s + '\n')
		s = ''
		for i in range(len(tk_lst)):
			tk = tk_lst[i]
			pos = pos_lst[i]
			s += tk + '\t' + pos + '\t0\t0\t0\t'
		fo.write(s.strip() + '\n')
		len1 = e1q - e1p + 1
		s = str(len1) + '\t'
		for i in range(len1):
			s += '0\t0\t'
		fo.write(s.strip() + '\n')
		len2 = e2q - e2p + 1
		s = str(len2) + '\t'
		for i in range(len2):
			s += '0\t0\t'
		fo.write(s.strip() + '\n')

	#exit(0)
	
fi.close()
fo.close()
foid.close()
