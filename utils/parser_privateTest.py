import csv
import numpy as np

private_signal = []
private_background = []
with open("storage/atlas-higgs-challenge-2014-v2.csv","r") as file:
	reader = csv.reader(file,delimiter=",")
	next(reader)
	for row in reader:
		if row[-2] == 'v':
			if row[-3] == 's':
				private_signal.append([float(row[i]) for i in range(1,32)])
			else:
				private_background.append([float(row[i]) for i in range(1,32)])

with open("storage/private_signal.csv","w") as file:
	writer = csv.writer(file,delimiter=",")
	writer.writerows(private_signal)

with open("storage/private_background.csv","w") as file:
	writer = csv.writer(file,delimiter=",")
	writer.writerows(private_background)
