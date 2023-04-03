import csv
import matplotlib.pyplot as plt
import random
import numpy as np
import MLP


# max signal weight in private data - 0.00572
# max background weight in private data - 2.3862
# ratio - .70137
# ratio of signal guesses - .152
# rati oof background guesses - .848
# new ratio = 5.578

sig_weight = []
back_weight = []
sig_rank = []
back_rank = []
sig_answer = []
back_answer = []
with open("../Data/melis-submission.csv","r") as file:
	reader = csv.reader(file,delimiter=",")
	for row in reader:
		if float(row[3]) == 1.0:
			sig_weight.append(float(row[1]))
			sig_rank.append([100*float(row[2])])
			sig_answer.append([float(row[3])])
		else:
			back_weight.append(float(row[1]))
			back_rank.append([100*float(row[2])])
			back_answer.append([float(row[3])])
print(max(sig_weight),max(back_weight))
max_r_b = max(back_rank[:][0])
max_r_s = max(sig_rank[:][0])
max_r = 0
if max_r_b > max_r_s:
	max_r = max_r_b
else:
	max_r = max_r_s
print(max(back_rank[:][0]),max(sig_rank[:][0]))

# for i in range(len(back_rank)):
# 	if back_rank[i][0] >= .848*max_r:
# 		back_rank[i][0] *= 5.578

# for i in range(len(sig_rank)):
# 	if sig_rank[i][0] >= .848*max_r:
# 		sig_rank[i][0] *= 5.578
print(max(back_rank[:][0]),max(sig_rank[:][0]))

print(sum(sig_weight),sum(back_weight))
ratio = sum(sig_weight)/sum(back_weight)
data = np.concatenate((sig_rank,back_rank)).tolist()
answer = np.concatenate((sig_answer,back_answer)).tolist()
weights = np.concatenate((sig_weight,back_weight)).tolist()

sig = [sig_rank[i][0] for i in range(len(sig_rank))]
back = [back_rank[i][0] for i in range(len(back_rank))]


k = 1
ks = [i for i in range(1,25)]
mis = []
for i in range(1,25):
	mis.append(MLP.mi_binary_weights(data,answer,weights,ratio,k=i))
fig, axs = plt.subplots()
axs.plot(ks,mis)
plt.show()


# signal = []
# background = []
# answer = []
# sig_weights2 = []
# back_weights2 = []
# for i in range(len(sorted_events)):
# 	if labels[i] == 's':
# 		signal.append(sorted_events[i][1])
# 		sig_weights.append(weights[i])
# 	else:
# 		background.append(sorted_events[i][1])
# 		back_weights.append(weights[i])
# 	continue


# fig, axs = plt.subplots()
# sig = axs.hist(signal,bins=500,alpha=.25,color='b',density=True,weights=sig_weights)
# back = axs.hist(background,bins=500,alpha=.25,color='r',density=True,weights=back_weights)

# axs.set_xlabel("Rank")


# sig_norm = sum(sig[0])
# sig_pdf = [sig[0][i]/sig_norm for i in range(len(sig[0]))]
# back_norm = sum(back[0])
# back_pdf = [back[0][i]/back_norm for i in range(len(back[0]))]
# joint = [sig[0][i] + back[0][i] for i in range(len(sig[0]))]
# joint_norm = sum(joint)
# joint = [joint[i]/joint_norm for i in range(len(joint))]

# p_s = len(signal)/(len(signal) + len(background))
# p_b = len(background)/(len(signal) + len(background))

# sig_ent = 0.0
# back_ent = 0.0
# joint_ent = 0.0
# for i in range(len(joint)):
# 	if joint[i] > 0.0:
# 		joint_ent -= joint[i] * np.log2(joint[i])
# 	if sig_pdf[i] > 0.0:
# 		sig_ent -= sig_pdf[i] * np.log2(sig_pdf[i])
# 	if back_pdf[i] > 0.0:
# 		back_ent -= back_pdf[i] * np.log2(back_pdf[i])
# print(joint_ent,sig_ent,back_ent,p_s,p_b)
# mi_val = joint_ent - p_s*sig_ent - p_b*back_ent
# print(mi_val)
# axs.set_ylabel("MI: %s" % mi_val)
# axs.set_title("Histogram of Rank for Melis Predictions")
# plt.savefig("Melis.png")
# plt.show()

# ids = []
# labels = []
# with open("id_label.csv","r") as file:
# 	reader = csv.reader(file,delimiter=",")
# 	next(reader)
# 	for row in reader:
# 		ids.append(int(float(row[0])))
# 		labels.append(row[1])
# labels = [labels[i] for i in range(249999,800002)]
# ids = [ids[i] for i in range(249999,800002)]

# ids2 = []
# labels2 = []
# events = []
# with open("final-submission-1.csv","r") as file:
# 	reader = csv.reader(file,delimiter=",")
# 	next(reader)
# 	for row in reader:
# 		ids2.append(int(row[0]))
# 		events.append([int(row[0]),float(row[1])/550000.0,row[2]])
# 		labels2.append(row[2])
# sorted_events = sorted(events,key=lambda x:x[0])


# signal = []
# background = []
# answer = []
# sig_weights = []
# back_weights = []
# for i in range(len(sorted_events)):
# 	#print(ids[i],labels[i],sorted_events[i],weights[i])
# 	if labels[i] == 's':
# 		signal.append(sorted_events[i][1])
# 		sig_weights.append(weights[i])
# 	else:
# 		background.append(sorted_events[i][1])
# 		back_weights.append(weights[i])
# 	continue
# event = []
# answer = []
# for i in range(len(signal)):
# 	event.append(signal[i])
# 	answer.append([1.0])
# for i in range(len(background)):
# 	event.append(background[i])
# 	answer.append([-1.0])
# #mi_val = ee.mi_binary(event,answer,k=3)

# #background = [background[i] for i in range(len(signal))]
# fig, axs = plt.subplots()
# sig = axs.hist(signal,bins=1000,range=(0.0,1.0),alpha=.25,color='b',density=True,stacked=True,weights=sig_weights)
# back = axs.hist(background,bins=1000,range=(0.0,1.0),alpha=.25,color='r',density=True,stacked=True,weights=back_weights)

# axs.set_xlabel("Rank")


# p_s = len(signal)/(len(signal) + len(background))
# p_b = len(background)/(len(signal) + len(background))
# # find mi from bins
# sig_norm = sum(sig[0])
# sig_pdf = [sig[0][i]/sig_norm for i in range(len(sig[0]))]
# back_norm = sum(back[0])
# back_pdf = [back[0][i]/back_norm for i in range(len(back[0]))]
# joint = [p_s*sig[0][i] + p_b*back[0][i] for i in range(len(sig[0]))]
# joint_norm = sum(joint)
# joint = [joint[i]/joint_norm for i in range(len(joint))]


# sig_ent = 0.0
# back_ent = 0.0
# joint_ent = 0.0
# for i in range(len(joint)):
# 	if joint[i] > 0.0:
# 		joint_ent -= joint[i] * np.log2(joint[i])
# 	if sig_pdf[i] > 0.0:
# 		sig_ent -= sig_pdf[i] * np.log2(sig_pdf[i])
# 	if back_pdf[i] > 0.0:
# 		back_ent -= back_pdf[i] * np.log2(back_pdf[i])
# print(joint_ent,sig_ent,back_ent,p_s,p_b)
# mi_val = joint_ent - p_s*sig_ent - p_b*back_ent
# print(mi_val)
# axs.set_ylabel("MI: %s" % mi_val)
# axs.set_title("Histogram of Rank for Melis Predictions")
# plt.savefig("melis.png")
# plt.show()
