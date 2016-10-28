import sys
import os
import json

fi = open(sys.argv[1], 'r')
fo = open(sys.argv[2], 'w')

for line in fi:
	dic = json.loads(line.strip())
	pos_lst = dic["pos"]
	tk_lst = dic["tokens"]
	rlt_lst = dic["relationMentions"]

	aid = dic["articleId"]
	sid = dic["sentId"]

	for rlt in rlt_lst:
		e1p = rlt["em1Start"]
		e1q = rlt["em1End"]
		e2p = rlt["em2Start"]
		e2q = rlt["em2End"]
		
		if e1p < e2q:
			p = e1p - 3
			q = e2q + 3
		elif e2p < e1q:
			p = e2p - 3
			q = e1q + 3

		s = ''
		for i in range(p,q):
			if i < 0 or i >= len(pos_lst):
				continue
			pos = pos_lst[i]
                	tk = tk_lst[i]
                	s = s + tk + '/' + pos + ' '

		strid = str(aid) + '_' + str(sid) + '_' + str(e1p) + '_' + str(e1q) + '_' + str(e2p) + '_' + str(e2q)
		
		lb = rlt["labels"][0]

		fo.write(strid + '::' + lb + '::' + s + '\n')
	
fi.close()
fo.close()
