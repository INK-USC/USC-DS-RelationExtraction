import json
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']

fin = open('result.txt') 

y_score = []
y_label = []

for line in fin:
	js = json.loads(line)
	label = js['relation']
	
	tmplist = [0]*10
	tmplist[label] = 1
	y_label.append(tmplist)

	tmplist = [js['score'][i] for i in range(10)]
	y_score.append(tmplist)

fin.close()

y_label = np.asarray(y_label)
y_score = np.asarray(y_score)

precision = dict()
recall = dict()
average_precision = dict()

# y_label_m = []
# y_score_m = []

# for i in range(10):
# 	if 1 not in y_label[:, i]:
# 		print(i)
# 		continue
# 	precision[i], recall[i], _ = precision_recall_curve(y_label[:, i], y_score[:, i])
# 	average_precision[i] = average_precision_score(y_label[:, i], y_score[:, i])
# 	y_label_m += y_label[:, i]
# 	y_score_m += y_score[:, i]

precision['hype'], recall['hype'], _ = precision_recall_curve(y_label.ravel(), y_score.ravel())

print(precision['hype'])
print(recall['hype'])

average_precision['hype'] = average_precision_score(y_label.ravel(), y_score.ravel(), average="micro")

plt.clf()
plt.plot(recall['hype'], precision['hype'], color='navy', label='HypeNET')

fin = open('CNN+ATT.txt')
precision['cnn'] = []
recall['cnn'] = []
for line in fin:
	tmplist = line.strip().split('\t')
	precision['cnn'].append((float)(tmplist[0]))
	recall['cnn'].append((float)(tmplist[1]))
fin.close()

plt.plot(recall['cnn'], precision['cnn'], color='turquoise', label='CNN+ATT')

fin = open('PCNN+ATT.txt')
precision['pcnn'] = []
recall['pcnn'] = []
for line in fin:
	tmplist = line.strip().split('\t')
	precision['pcnn'].append((float)(tmplist[0]))
	recall['pcnn'].append((float)(tmplist[1]))
fin.close()

plt.plot(recall['pcnn'], precision['pcnn'], color='darkorange', label='PCNN+ATT')

fin = open('CNN+max.txt')
precision['cmax'] = []
recall['cmax'] = []
for line in fin:
	tmplist = line.strip().split('\t')
	precision['cmax'].append((float)(tmplist[0]))
	recall['cmax'].append((float)(tmplist[1]))
fin.close()

plt.plot(recall['cmax'], precision['cmax'], color='cornflowerblue', label='CNN+max')

fin = open('Path+max.txt')
precision['pmax'] = []
recall['pmax'] = []
for line in fin:
	tmplist = line.strip().split('\t')
	precision['pmax'].append((float)(tmplist[0]))
	recall['pmax'].append((float)(tmplist[1]))
fin.close()

plt.plot(recall['pmax'], precision['pmax'], color='teal', label='Path+max')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('PR-curve.png')