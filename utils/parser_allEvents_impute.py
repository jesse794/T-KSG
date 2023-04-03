import csv
import numpy as np

# Figure out average value for each column (excluding the -999's)
with open("storage/atlas-higgs-challenge-2014-v2.csv","r") as file:
	reader = csv.reader(file,delimiter=",")
	for irow, row in enumerate(reader):
		if irow==0:  # use the header row to count cols and initialize
			nCols = len(row)
			colSum = [0 for _ in range(nCols)]
			colCount = [0 for _ in range(nCols)]
		else:
			for icol, col in enumerate(row):
				if icol<=nCols-5 and float(col) != -999: # skip last four cols
					colSum[icol]+=float(col)
					colCount[icol]+=1

		colAvgs = [colSum[_]/colCount[_]
		if colCount[_]>0 else 0 for _ in range(nCols)]

# Now read rows, replace the -999's with col avg and write output
allevts_signal = []
allevts_background = []
with open("storage/atlas-higgs-challenge-2014-v2.csv","r") as file:
	reader = csv.reader(file,delimiter=",")
	next(reader)
	for row in reader:  # Loop over rows
		# the -4 is a kludge to avoid the last 4 cols which are textual
		# or are a weight value
		for iCol in range(nCols-4):  # Replace any -999 in row w/ avg
			if float(row[iCol]) == -999: row[iCol] = colAvgs[iCol]
		if row[-3] == 's':
			allevts_signal.append([float(row[i]) for i in range(1,32)])
		else:
			allevts_background.append([float(row[i]) for i in range(1,32)])

with open("storage/allevents_signal_imputed.csv","w") as file:
	writer = csv.writer(file,delimiter=",")
	writer.writerows(allevts_signal)

with open("storage/allevents_background_imputed.csv","w") as file:
	writer = csv.writer(file,delimiter=",")
	writer.writerows(allevts_background)
