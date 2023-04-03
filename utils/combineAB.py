import csv
import numpy as np

AB = []
with open("trainingForRelease-justAB.csv","r") as file:
    reader = csv.reader(file,delimiter=",")
    next(reader)
    for row in reader:
        AB.append([int(row[0]),float(row[1]),float(row[2])])
with open("testForRelease-justAB.csv","r") as file:
    reader = csv.reader(file,delimiter=",")
    next(reader)
    for row in reader:
        AB.append([int(row[0]),float(row[1]),float(row[2])])
allvars = []
with open("atlas-higgs-challenge-2014-v2.csv","r") as file:
    reader = csv.reader(file,delimiter=",")
    next(reader)
    for row in reader:
        allvars.append([row[i] for i in range(len(row))])

private_signal = []
private_background = []
for i in range(len(allvars)):
    if allvars[i][-2] == 'v':
        temp = [AB[i][1],AB[i][2]]
        for j in range(1,32):
            temp.append(float(allvars[i][j]))
        if allvars[i][-3] == 's':
            private_signal.append(temp)
        else:
            private_background.append(temp)

with open("private_signal_AB.csv", "w") as file:
    writer = csv.writer(file,delimiter=",")
    writer.writerows(private_signal)

with open("private_background_AB.csv", "w") as file:
    writer = csv.writer(file,delimiter=",")
    writer.writerows(private_background)