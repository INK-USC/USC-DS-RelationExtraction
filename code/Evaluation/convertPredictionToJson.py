import sys
import json

data = sys.argv[1]
predictionFile = 'data/results/'+data+'/rm/prediction_emb_retypeRm_cosine.txt'
testJson = 'data/intermediate/'+data+'/rm/test_new.json'
predictionJson = 'data/results/'+data+'/rm/prediction.json'
mentionMapFile = 'data/intermediate/'+data+'/rm/mention.txt'
typeMapFile = 'data/intermediate/'+data+'/rm/type.txt'
threshold = float(sys.argv[2])

tid2Name = {}
with open(typeMapFile) as typeF:
  for line in typeF:
    seg = line.strip('\r\n').split('\t')
    tid2Name[seg[1]] = seg[0]
    if seg[0] == 'None':
      noneid = seg[1]

mention2id = {}
with open(mentionMapFile) as menF:
  for line in menF:
    seg = line.strip('\r\n').split('\t')
    mention2id[seg[0]] = seg[1]

mid2tid = {}
with open(predictionFile, 'r') as predF:
  for line in predF:
    seg = line.strip('\r\n').split('\t')
    mid = seg[0]
    tid = seg[1]
    if float(seg[2]) < threshold:
      tid = noneid
    mid2tid[mid] = tid

with open(testJson) as testJ, open(predictionJson, 'w') as predJ:
  for line in testJ:
    sentDic = json.loads(line.strip('\r\n'))
    aid = str(sentDic['articleId'])
    sid = str(sentDic['sentId'])
    new_rms = []
    for rm in sentDic['relationMentions']:
      mention = '_'.join([aid, sid, str(rm['em1Start']), str(rm['em1End']), str(rm['em2Start']), str(rm['em2End'])])
      mid = mention2id[mention]
      if mid not in mid2tid:
        tid = noneid
      else:
        tid = mid2tid[mid]
      predicted_type = tid2Name[tid]
      rm['labels'] = [predicted_type]
    predJ.write(json.dumps(sentDic)+'\n')




