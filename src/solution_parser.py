import csv
import numpy as np
import matplotlib.pyplot as plt
import MLP

sol_id = []
sol_weight = []
sol_label = []
with open("../Data/atlas-higgs-challenge-2014-v2.csv","r") as file:
	reader = csv.reader(file,delimiter=",")
	next(reader)
	for row in reader:
		if row[-2] == 'v':
			sol_id.append(int(row[0]))
			sol_weight.append(float(row[-4]))
			sol_label.append(row[-3])

pred_id = []
pred_rank = []
pred_label = []
pred = []
with open("../Data/preds_0_175.csv","r") as file:
	reader = csv.reader(file,delimiter=",")
	next(reader)
	for row in reader:
		pred_id.append(int(row[0]))
		pred_rank.append(float(row[1]))
		pred_label.append(row[2])
		pred.append([int(row[0]),float(row[1])/55000,row[2]])
pred = sorted(pred, key=lambda x:x[0])

total = []
k=0
for i in range(0,len(sol_id)):
	for j in range(k,len(pred)):
		if sol_id[i] == pred[j][0]:
			if pred[j][2] == 's':
				print(i,j,len(total))
				if sol_label[i] == 's':
					total.append([pred[j][0],sol_weight[i]/0.005720682,pred[j][1],1.0])
				else:
					total.append([pred[j][0],sol_weight[i]/2.3861992,pred[j][1],-1.0])
				if i % 1000 == 0:
					print(len(total),i,j)
					k=j
				break

with open('salimans-submission-only-signal.csv',"w") as file:
	writer = csv.writer(file, delimiter=",")
	writer.writerows(total)



weights = []
answer = []

sig_weights = [691.988*sol_weight[i]/380.088 for i in range(len(sol_weight)) if sol_label[i] == 's']
back_weights = [410999.847*sol_weight[i]/226044.057 for i in range(len(sol_weight)) if sol_label[i] == 'b']
max_s = max(sig_weights)
max_b = max(back_weights)
print(max_s,max_b)
for i in range(len(sol_weight)):
	if sol_label[i] == 's':
		weights.append(691.988*sol_weight[i]/(380.088*max_s))
		answer.append([1.0])
	else:
		weights.append(410999.847*sol_weight[i]/(226044.057*max_b))
		answer.append([-1.0])

max_rank = max([pred[i][1] for i in range(len(pred))])
scaled_rank = [[float(pred[i][1])/max_rank] for i in range(len(pred))]

b_fraction = (380.088)/(226044.057)



ams_c = []
cutoff = [0.01*i for i in range(1,100)]
for k in range(99):
	true_pos = []
	true_pos_weight = []
	false_pos = []
	false_pos_weight = []
	for i in range(len(pred)):
		if pred[i][2] == 's':
			if pred[i][1] >= (cutoff[k]*550000):
				if pred[i][2] == sol_label[i]:
					true_pos.append([pred[i][0]])
					true_pos_weight.append(sol_weight[i]*691.88/380.088)
				else:
					false_pos.append([pred[i][0]])
					false_pos_weight.append(sol_weight[i]*410999.847/226044.057)

	s = sum(true_pos_weight)
	b = sum(false_pos_weight)
	b_0 = .5 * (b - 0.1*b + np.sqrt((b-0.1*b*0.1*b)**2 + 4*(s+b)*0.1*b*0.1*b))
	sigma_b = 0.1*b
	term = np.log((s+b)/b_0)
	ams = np.sqrt(2*((s+b)*term - s - b + b_0) + (b-b_0)*2/(sigma_b**2))
	ams_c.append(np.sqrt(2*((s+b+10)*np.log(1+(s)/(b+10))-s)))
	print(ams_c[-1])
fig, axs = plt.subplots()
axs.plot(cutoff,ams_c)
plt.show()

