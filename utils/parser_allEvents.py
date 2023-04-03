import csv
import numpy as np

allevts_signal = []
allevts_background = []
with open("storage/atlas-higgs-challenge-2014-v2.csv","r") as file:
	reader = csv.reader(file,delimiter=",")
	next(reader)
	for row in reader:
		if row[-3] == 's':
			allevts_signal.append([float(row[i]) for i in range(1,32)])
		else:
			allevts_background.append([float(row[i]) for i in range(1,32)])

with open("storage/allevents_signal.csv","w") as file:
	writer = csv.writer(file,delimiter=",")
	writer.writerows(allevts_signal)

with open("storage/allevents_background.csv","w") as file:
	writer = csv.writer(file,delimiter=",")
	writer.writerows(allevts_background)
