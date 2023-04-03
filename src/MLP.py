# ML and IT scripts										 Nicholas Carrara
'''

'''
#-----------------------------------------------------------------------------
#	Required packages
import os
# set the backend to tensorflow
os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.spatial as ss
from scipy.special import digamma
import numpy.random as nr
from math import log
import seaborn as sns
import random
import csv
import itertools
import pandas as pd
import keras
# tensor functionality uses K
from keras import backend as K
from keras import optimizers, models, layers
from keras import initializers as ki
from scipy.special import digamma, gamma
from math import log2
# uproot for grabbing events from a TTree
import uproot

#-----------------------------------------------------------------------------
#	Converting csv and root files to npz
#-----------------------------------------------------------------------------
#	convert_csv_to_npz(file       - specify the filename "file"
# 					   var_set    - the column numbers of the variables
#                      class_list - possible list of different classes
#                      class_loc  - possible column for the class label)
#-----------------------------------------------------------------------------
def convert_csv_to_npz(
	file: str,
	var_set: list,
	class_list=[],
	class_loc=-1
):
	# check that csv file exists
	if(not os.path.exists(file+".csv")):
		print("File " + file + ".csv does not exist!")
		return
	data = []
	with open(file+".csv",'r') as temp:
		reader = csv.reader(temp,delimiter=",")
		for row in reader:
			data.append([float(row[i]) for i in range(len(row))])
	# if there are no classes
	if(class_loc == -1):
		np.savez(file+'.npz',np.asarray(data,dtype=np.float32))
		print("Set of %s events in %s variables saved in file %s.npz" %
		      (len(data),len(var_set),file))
	else:
		for j in range(len(class_list)):
			temp_data = [data[i] for i in range(len(data)) if
			             data[i][class_loc] == class_list[j]]
			np.savez(file+'_class%s.npz' % j,np.asarray(temp_data,
			                                            dtype=np.float32))
			print("Set of %s events in %s variables saved in %s_class%s.npz"
			      % (len(data),len(var_set),file,j))
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	convert_root_to_npz(file       - specify the filename "file"
#                       tree       - name of the tree
# 					   	var_set    - list of variable names in the tree
#-----------------------------------------------------------------------------
def convert_root_to_npz(
	file: str,
	tree: str,
	var_set: list
):
	# check that csv file exists
	if(not os.path.exists(file+".root")):
		print("File " + file + ".root does not exist!")
		return
	rootfile = uproot.open(file+'.root')
	temp = []
	for j in range(len(var_set)):
		temp.append(rootfile[tree].array(var_set[j]))
	data = [[temp[i][j] for i in range(len(var_set))]
	        for j in range(len(temp[0]))]
	np.savez(file+'.npz',data)
	print("Set of %s events in %s variables saved in file %s.npz" %
		      (len(data),len(var_set),file))
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	generate_binary_training_testing_data(signal_file     - the signal file
#                                         background_file - the background file
#                                         labels          - the labels in npz
# 										  var_set      - which variables to use
#                                         ratio_of_file- amount of file to use
#										  testing_ratio- amount of data testing
#										  symmetric_signals- p(s) = p(b)
#										  percentage_signal- p(s)
#										  percentage_background- p(b))
#-----------------------------------------------------------------------------
def generate_binary_training_testing_data(
	signal_file: str,
	background_file: str,
	labels: list,
 	var_set=[],
	ratio_of_file=1.0,
	testing_ratio=0.3,
	symmetric_signals=False,
	percentage_signal=1.0,
	percentage_background=1.0
):
	# First load up the signal and background files, then determine the
	# amount of the files to use with ratio_of_file
	print("loading...")
	signal = np.load(signal_file)[labels[0]]
	background = np.load(background_file)[labels[1]]
	#	if the user only wants certain variables
	if (var_set != []):
		signal = signal[:,var_set]
		background = background[:,var_set]
	np.random.shuffle(signal)
	np.random.shuffle(background)
	# Now determine which is smallest, the length of signal background
	# or the ratios
	if (symmetric_signals==True):
		num_of_events = np.amin([len(signal),len(background),
		                         int(len(signal)*ratio_of_file),
								 int(len(background)*ratio_of_file)])
		signal = signal[:][:num_of_events]
		background = background[:][:num_of_events]
		#	Now determine the amount of testing data
		temp_data = []
		data, answer = [], []
		for j in range(len(signal)):
			temp_data.append([signal[j],1.0])
			temp_data.append([background[j],-1.0])
		np.random.shuffle(temp_data)
		for j in range(len(temp_data)):
			data.append(temp_data[j][0])
			answer.append([temp_data[j][1]])
	else:
		temp_data = []
		data, answer = [], []
		num_signal = int(percentage_signal * len(signal))
		num_background = int(percentage_background * len(background))
		for j in range(num_signal):
			temp_data.append([signal[j],1.0])
		for j in range(num_background):
			temp_data.append([background[j],-1.0])
		np.random.shuffle(temp_data)
		for j in range(len(temp_data)):
			data.append(temp_data[j][0])
			answer.append([temp_data[j][1]])
	if ( testing_ratio == 0.0 ):
		print("Loaded files %s and %s with %s events for training and 0 events \
		       for testing." % (signal_file, background_file, len(data)))
		return data, answer
	else:
		#	Otherwise we partition the amount of testing data
		num_of_testing = int(len(data)*testing_ratio)
		train_data = list( data[:][:-num_of_testing] )
		train_answer = list( answer[:][:-num_of_testing] )
		test_data = list( data[:][-num_of_testing:] )
		test_answer = list( answer[:][-num_of_testing:] )
		# print("Loaded files %s and %s with " % (signal_file, background_file))
		# print("|----------------------------------------|")
		# print("|     Class     |  Training  |  Testing  |")
		# print("|----------------------------------------|")
		# print("|   %s  |  %s  |  %s  " % (signal_file,num_signal,
		#       int(len(test_data)*num_signal/(num_signal + num_background))))
		# print("|   %s  |  %s  |  %s  " % (background_file,num_background,
		#       int(len(test_data)*num_signal/(num_signal + num_background))))
		# print("|----------------------------------------|")
		# print("|     Total     |  %s  |  %s  " % (len(train_data),len(test_data)))
		# print("|----------------------------------------|")
		return train_data, train_answer, test_data, test_answer
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	Information Theory
#-----------------------------------------------------------------------------
'''
	This set of functions are used for calculating mutual information (MI)
	in various contexts.  Most of this code is copied from NPEET;
	https://github.com/gregversteeg/NPEET
'''
#-----------------------------------------------------------------------------
#	mi (original from Greg)
# Mutual information of x and y
# x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
# if x is a one-dimensional scalar and we have four samples"""

#-----------------------------------------------------------------------------
def mi(x, y, k=3, base=2):
	assert len(x) == len(y), "Lists should have same length"
	assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
	intens = 1e-10  # small noise to break degeneracy, see doc.
	x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
	y = [list(p + intens * nr.rand(len(y[0]))) for p in y]
	points = joint_space(x, y)
	# Find nearest neighbors in joint space, p=inf means max-norm
	tree = ss.cKDTree(points)
	dvec = [tree.query(point, k+1, p=float('inf'))[0][k] for point in points]
	a = avgdigamma(x, dvec, k)
	b = avgdigamma(y, dvec, k)
	c, d = digamma(k), digamma(len(x))
	return (- a - b + c + d) / log(base)

#-----------------------------------------------------------------------------
#	mi_binary
#-----------------------------------------------------------------------------
def mi_binary(x, y, k=3, base=2):
	""" MI between continuous input vector and binary answer"""
	assert len(x) == len(y), "Lists should have same length"
	assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
	intens = 1e-10  # small noise to break degeneracy, see doc.
	x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
	points = joint_space(x, y)
	# Find nearest neighbors in joint space, p=inf means max-norm
	tree = ss.cKDTree(points)
	dvec = [tree.query(point, k+1, p=float('inf'))[0][k] for point in points]
	a, margs = avgdigamma(x, dvec, k)
	avg_b_mis = 0
	n_s = 0
	for i in range(len(margs)):
		if y[i] == [1.0]:
			avg_b_mis += margs[i] - k
			n_s += 1
	avg_b_mis /= n_s
	b, c, d = avgdigamma2(y), digamma(k), digamma(len(x))
	return (-a - b + c + d) / log(base)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	mi_binary with weights
#-----------------------------------------------------------------------------
def mi_binary_weights(x, y, weights, fraction, k=3, base=2):
	""" MI between input vector of continuous variables
	and a binary answer in case where events need to
	be culled based on their weight."""
	assert len(x) == len(y), "Lists should have same length"
	assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
	dvecs = []
	margs = []
	intens = 1e-10  # small noise to break degeneracy, see doc
	if fraction >= 1.0:
		weight_s = 1.0/fraction
		weight_b = 1.0
	else:
		weight_s = 1.0
		weight_b = fraction
	x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
	points = joint_space(x, y)
	n_s = 0
	n_b = 0
	# Find nearest neighbors in joint space, p=inf means max-norm
	tree = ss.cKDTree(points)
	marg_tree = ss.cKDTree(x)
	avg_b_mis = 0
	a = 0.0
	for i in range(len(x)):
		# determine whether the current point is considered
		temp_weight = 0
		if y[i] == [1.0]:
			temp_weight = weight_s * weights[i]
		else:
			temp_weight = weight_b * weights[i]
		if temp_weight >= np.random.uniform(0,1,1)[0]:
			if y[i] == [1.0]:
				n_s += 1
			else:
				n_b += 1
			# collect list of neighbors for each point
			neighbors = tree.query(points[i], k+20, p=float('inf'))
			dvec = 0.0
			num_passed = 0
			num_searched = 0
			# search for the passing k value
			for j in range(1,len(neighbors[1])):
				# if weights[j] > random.uniform
				num_searched += 1
				if y[i] == [1.0]:
					if weight_s * weights[neighbors[1][j]] >= np.random.uniform(0,1,1)[0]:
						num_passed += 1
						dvec = neighbors[0][j]
				else:
					if weight_b * weights[neighbors[1][j]] >= np.random.uniform(0,1,1)[0]:
						num_passed += 1
						dvec = neighbors[0][j]
				if num_passed >= k:
					break
			# search within the marginal space to find points
			neighbors = marg_tree.query_ball_point(x[i], dvec,
												p=float('inf'))
			num_passed = k
			for j in range(1,len(neighbors)):
				# only look at points in the opposite class
				if y[i] != y[neighbors[j]]:
					if y[i] == [1.0]:
						if weight_b * weights[neighbors[j]] >= np.random.uniform(0,1,1)[0]:
							num_passed += 1
					else:
						if weight_s * weights[neighbors[j]] >= np.random.uniform(0,1,1)[0]:
							num_passed += 1
			if y[i] == [1.0]:
				avg_b_mis += (num_passed-k)
			a += digamma(num_passed)
			margs.append(num_passed)
			dvecs.append(dvec)
		#print(i,n_s,n_b,weights[i])
	avg_b_mis /= n_s
	a /= (n_s+n_b)
	b = (n_s)/(n_s+n_b) * digamma(n_s) + (n_b)/(n_s+n_b) * digamma(n_b)
	c, d = digamma(k), digamma(n_s + n_b)
	return (-a - b + c + d) / log(base)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	mi_binary with weights for discrete variables
#-----------------------------------------------------------------------------
def mi_binary_discrete_weights(x, y, disc_list, weights, fraction, k=3, base=2):
	""" MI between input vector of discrete variables
	and a binary answer in case where events need to
	be culled based on their weight."""
	assert len(x) == len(y), "Lists should have same length"
	assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
	dvecs = []
	margs = []
	intens = 1e-10  # small noise to break degeneracy, see doc
	if fraction >= 1.0:
		weight_s = 1.0/fraction
		weight_b = 1.0
	else:
		weight_s = 1.0
		weight_b = fraction


	# find the unique values of the discrete variable
	unique_disc = []
	unique_disc_nums = []
	unique_disc_sig = []
	unique_disc_back = []
	total_sig = 0
	total_back = 0
	for i in range(len(x)):
		temp_disc = [x[i][j] for j in range(len(x[0])) if j in disc_list]
		if temp_disc not in unique_disc:
			unique_disc.append(temp_disc)
			unique_disc_nums.append([i])
			if y[i][0] == 1.0 :
				unique_disc_sig.append(1)
				total_sig += 1
				unique_disc_back.append(0)
			else:
				unique_disc_back.append(1)
				total_back += 1
				unique_disc_sig.append(0)
		else:
			for j in range(len(unique_disc)):
				if temp_disc == unique_disc[j]:
					unique_disc_nums[j].append(i)
					if y[i][0] == 1.0:
						unique_disc_sig[j] += 1
						total_sig += 1
					else:
						unique_disc_back[j] += 1
						total_back += 1
	#   find mi of discrete vars

	joint_prob = []
	signal_prob = []
	back_prob = []
	for i in range(len(unique_disc_nums)):
		temp_joint_prob = 0
		temp_sig_prob = 0
		temp_back_prob = 0
		for j in range(len(unique_disc_nums[i])):
			if y[unique_disc_nums[i][j]] == [1.0]:
				temp_weight = weight_s * weights[unique_disc_nums[i][j]]
			else:
				temp_weight = weight_b * weights[unique_disc_nums[i][j]]
			if temp_weight >= np.random.uniform(0,1,1)[0]:
				temp_joint_prob += 1
				if y[unique_disc_nums[i][j]] == [1.0]:
					temp_sig_prob +=1
				else:
					temp_back_prob +=1
		signal_prob.append(temp_sig_prob)
		back_prob.append(temp_back_prob)
		joint_prob.append(temp_joint_prob)
		#print('joint',i,temp_joint_prob,len(unique_disc_nums[i]),temp_sig_prob,temp_back_prob)
	#print(joint_prob,signal_prob,back_prob)
	total_events = sum(joint_prob)
	total_joint = sum(joint_prob)
	total_sig = sum(signal_prob)
	total_back = sum(back_prob)
	joint_prob = [joint_prob[i]/total_joint for i in range(len(joint_prob))]
	signal_prob = [signal_prob[i]/total_sig for i in range(len(signal_prob))]
	back_prob = [back_prob[i]/total_back for i in range(len(back_prob))]
	#   compute the mi
	disc_mi = 0
	for i in range(len(joint_prob)):
		if(joint_prob[i] > 0):
			disc_mi -= joint_prob[i]*log(joint_prob[i])
	for i in range(len(signal_prob)):
		if(signal_prob[i] > 0):
			disc_mi += (total_sig/total_joint)*signal_prob[i]*log(signal_prob[i])
	for i in range(len(back_prob)):
		if(back_prob[i] > 0):
			disc_mi += (total_back/total_joint)*back_prob[i]*log(back_prob[i])
	#print(unique_disc,disc_mi)
	a = []
	b = []
	c = []
	d = []
	for l in range(len(unique_disc)):
		temp_x = [x[unique_disc_nums[l][i]] for i in range(len(unique_disc_nums[l]))]
		temp_x = [list(p + intens * nr.rand(len(x[0]))) for p in temp_x]
		temp_y = [y[unique_disc_nums[l][i]] for i in range(len(unique_disc_nums[l]))]
		temp_weights = [weights[unique_disc_nums[l][i]] for i in range(len(unique_disc_nums[l]))]
		points = joint_space(temp_x, temp_y)
		n_s = 0
		n_b = 0
		# Find nearest neighbors in joint space, p=inf means max-norm
		tree = ss.cKDTree(points)
		marg_tree = ss.cKDTree(temp_x)
		avg_b_mis = 0
		temp_a = 0.0
		for i in range(len(temp_x)):
			# determine whether the current point is considered
			temp_weight = 0
			if temp_y[i] == [1.0]:
				temp_weight = weight_s * temp_weights[i]
			else:
				temp_weight = weight_b * temp_weights[i]
			if temp_weight >= np.random.uniform(0,1,1)[0]:
				if temp_y[i] == [1.0]:
					n_s += 1
				else:
					n_b += 1
				# collect list of neighbors for each point
				neighbors = tree.query(points[i], k+20, p=float('inf'))
				dvec = 0.0
				num_passed = 0
				num_searched = 0
				# search for the passing k value
				for j in range(1,len(neighbors[1])):
					# if weights[j] > random.uniform
					num_searched += 1
					if temp_y[i] == [1.0]:
						if weight_s * temp_weights[neighbors[1][j]] >= np.random.uniform(0,1,1)[0]:
							num_passed += 1
							dvec = neighbors[0][j]
					else:
						if weight_b * temp_weights[neighbors[1][j]] >= np.random.uniform(0,1,1)[0]:
							num_passed += 1
							dvec = neighbors[0][j]
					if num_passed >= k:
						break
				# search within the marginal space to find points
				neighbors = marg_tree.query_ball_point(temp_x[i], dvec,
													p=float('inf'))
				num_passed = k
				for j in range(1,len(neighbors)):
					# only look at points in the opposite class
					if temp_y[i] != temp_y[neighbors[j]]:
						if temp_y[i] == [1.0]:
							if weight_b * temp_weights[neighbors[j]] >= np.random.uniform(0,1,1)[0]:
								num_passed += 1
						else:
							if weight_s * temp_weights[neighbors[j]] >= np.random.uniform(0,1,1)[0]:
								num_passed += 1
				if temp_y[i] == [1.0]:
					avg_b_mis += (num_passed-k)
				temp_a += digamma(num_passed)
				margs.append(num_passed)
				dvecs.append(dvec)
		#print(temp_a,joint_prob[l])
		d_s, d_b = 0,0
		if n_s > 0:
			d_s = digamma(n_s)
		if n_b > 0:
			d_b = digamma(n_b)
		if n_s == 0 and n_b == 0:
			continue
		temp_b = (n_s)/(n_s+n_b) * d_s + (n_b)/(n_s+n_b) * d_b
		temp_c = digamma(k)
		temp_d = digamma(n_s + n_b)
		a.append(temp_a*joint_prob[l]/(n_s+n_b))
		b.append(temp_b*joint_prob[l])
		c.append(temp_c*joint_prob[l])
		d.append(temp_d*joint_prob[l])
	print(disc_mi,a,b,c,d)
	a_final = sum(a)
	b_final = sum(b)
	c_final = sum(c)
	d_final = sum(d)
	return (disc_mi -a_final - b_final + c_final + d_final) / log(base)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	mi_binary for discrete variables
#-----------------------------------------------------------------------------
def mi_binary_discrete(x, y, disc_list=[], k=3, base=2):
	""" MI between input vector of discrete variables
	and a binary answer """
	assert len(x) == len(y), "Lists should have same length"
	assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
	intens = 1e-10  # small noise to break degeneracy, see doc.
	#x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
	avg_mi = 0
	disc_mi = 0
	#   find unique sets of discrete variables
	unique_disc = []
	unique_disc_nums = []
	unique_disc_sig = []
	unique_disc_back = []
	total_sig = 0
	total_back = 0
	for i in range(len(x)):
	    temp_disc = [x[i][j] for j in range(len(x[0])) if j in disc_list]
	    if temp_disc not in unique_disc:
	        unique_disc.append(temp_disc)
	        unique_disc_nums.append([i])
	        if y[i][0] == 1.0 :
	            unique_disc_sig.append(1)
	            total_sig += 1
	            unique_disc_back.append(0)
	        else:
	            unique_disc_back.append(1)
	            total_back += 1
	            unique_disc_sig.append(0)
	    else:
	        for j in range(len(unique_disc)):
	            if temp_disc == unique_disc[j]:
	                unique_disc_nums[j].append(i)
	                if y[i][0] == 1.0:
	                    unique_disc_sig[j] += 1
	                    total_sig += 1
	                else:
	                    unique_disc_back[j] += 1
	                    total_back += 1
	#   find mi of discrete vars
	total_events = len(x)
	joint_prob = []
	signal_prob = []
	back_prob = []
	for i in range(len(unique_disc_nums)):
	    joint_prob.append(len(unique_disc_nums[i])/total_events)
	for i in range(len(unique_disc_sig)):
	    signal_prob.append(unique_disc_sig[i]/total_sig)
	for i in range(len(unique_disc_back)):
	    back_prob.append(unique_disc_back[i]/total_back)
	#   compute the mi
	disc_mi = 0
	for i in range(len(joint_prob)):
	    if(joint_prob[i] > 0):
	        disc_mi -= joint_prob[i]*log2(joint_prob[i])
	for i in range(len(signal_prob)):
	    if(signal_prob[i] > 0):
	        disc_mi += (total_sig/total_events)*signal_prob[i]*log2(signal_prob[i])
	for i in range(len(back_prob)):
	    if(back_prob[i] > 0):
	        disc_mi += (total_back/total_events)*back_prob[i]*log2(back_prob[i])
	for l in range(len(unique_disc)):
	    temp_x = [x[unique_disc_nums[l][i]] for i in range(len(unique_disc_nums[l]))]
	    temp_y = [y[unique_disc_nums[l][i]] for i in range(len(unique_disc_nums[l]))]
	    #y = [list(p + intens * nr.rand(len(y[0]))) for p in y]
	    points = zip2(temp_x,temp_y)
	    # Find nearest neighbors in joint space, p=inf means max-norm
	    tree = ss.cKDTree(points)
	    dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
	    a, lists =  avgdigamma(temp_x, dvec)
	    b, c, d = avgdigamma2(temp_y), digamma(k), digamma(len(temp_x))
	    avg_mi += ((-a-b+c+d)/log(base))/len(unique_disc)
	return disc_mi + avg_mi
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	joint_space
#-----------------------------------------------------------------------------
def joint_space(*args):
    """ zip2(x, y) takes the lists of vectors and makes it a list of vectors
	 in a joint space
     E.g. zip2([[1], [2], [3]], [[4], [5], [6]]) = [[1, 4], [2, 5], [3, 6]]"""
    return [sum(sublist, []) for sublist in zip(*args)]
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	avgdigamma
#-----------------------------------------------------------------------------
def avgdigamma(points, dvec, k):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
	N = len(points)
	tree = ss.cKDTree(points)
	avg = 0.
	margs = []
	for i in range(N):
		dist = dvec[i]
		# subtlety, we don't include the boundary point,
		# but we are implicitly adding 1 to kraskov def bc center point
		# is included
		num_points = len(tree.query_ball_point(points[i],
						 dist - 1e-15,
						 p=float('inf')))
		if num_points > 0:
			avg += digamma(num_points) / N
			margs.append(num_points)
		else:
			avg += digamma(k) / N
			margs.append(k)
	return avg, margs
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	avgdigamma2(points)
#		This is used to evaluate the <digamma(y)> when y is a binary
#		category variable.
#-----------------------------------------------------------------------------
def avgdigamma2(points):
	N = len(points)
	num_sig = 0.0
	num_back = 0.0
	for i in range(len(points)):
		if (points[i][0] > 0.0):
			num_sig += 1.0
		else:
			num_back += 1.0
	#print(num_sig,num_back)
	avg = (num_sig)/N * digamma(num_sig) + (num_back)/N * digamma(num_back)
	return avg
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	avgdigamma3(points)
#		This is used to evaluate the <digamma(y)> when y is a binary
#		category variable.
#-----------------------------------------------------------------------------
def avgdigamma3(points,fraction):
	num_sig = 0.0
	num_back = 0.0
	for i in range(len(points)):
		if (points[i][0] > 0.0):
			num_sig += 1.0
		else:
			num_back += 1.0
	if fraction >= 1.0:
		num_back = int(fraction*num_sig)
	else:
		num_sig = int(num_sig/fraction)
	N = num_back + num_sig
	avg = (num_sig)/N * digamma(num_sig) + (num_back)/N * digamma(num_back)
	return avg
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	Neural Network class
#-----------------------------------------------------------------------------
'''
	This class generates a sequential MLP model using the Keras library.
	The user must specify the topology of the network; i.e. the number
	of nodes in each layer; e.g. [4,10,3,1]
	Default values are chosen for the other options, however the user has
	the ability to pick an optimizer, an activation function for each layer,
	and an initializer for the weights.


'''
#-----------------------------------------------------------------------------
class MLP:
	#-------------------------------------------------------------------------
	#   __init__(self,topology - layer structure for the network, [4,10,3,1]
	#                 optimizer- select between; 'SGD' - Stochastic Grad. Des.
	#											 'RMSprop' - RMS Prop.
	#											 'Adagrad' - Adagrad
	#                                            'Adadelta'- Adadelta
	#                                            'Adam'    - Adam
	#                                            'Adamax'  - Adamax
	#                                            'Nadam'   - Nadam
	#                 opt_params- 'SGD'     - [lr,momentum,decay,nesterov]
    #                             'RMSprop' - [lr,rho,epsilon,decay]
	#                             'Adagrad' - [lr,epsilon,decay]
	#                             'Adadelta'- [lr,rho,epsilon,decay]
	#                             'Adam'    - [lr,beta_1,beta_2,epsilon,decay,
	#                                          amsgrad]
	#                             'Adamax'  - [lr,beta_1,beta_2,epsilon,decay]
	#                             'Nadam'   - [lr,beta_1,beta_2,epsilon,
	#                                          scheduled_decay]
	#                 activation- select between; 'tanh'     - Tanh function
	#                                             'elu'      - exp. linear
	#											  'selu'- scaled exp. linear
	#                                             'softplus' - Soft plus
	#                                             'softsign' - Soft sign
	#                                             'relu'     - rect. linear
	#                                             'sigmoid'  - Sigmoid
	#                                             'hard_sigmoid' - Hard Sig.
	#                                             'exponential' - exp
	#                                             'linear' - Linear
	#                 act_params- 'elu'       - [alpha]
	#                             'relu'      - [alpha,max_value,threshold]
	#                 initializer- select between; 'zeros' - set all to zero
	#                                              'ones'  - set all to one
	#                                              'constant' - Constant
	#                                              'normal'- random normal
	#                                              'uniform' - random uniform
	#                                              'truncated_normal'
	#                                              'variance_scaling'
	#                                              'orthogonal'
	#                 init_params - 'constant' - [value]
	#                               'normal'   - [mean,stddev,seed]
	#                               'uniform'  - [minval,maxval,seed]
	#                               'truncated_normal'- [mean,stddev,seed]
	#                               'variance_scaling'- [scale,mode,dist,seed]
	#                               'orthogonal'      - [gain,seed]
	# 				  filename - if provided, load this model, ignore params
	#-------------------------------------------------------------------------
	def __init__(self,
		topology: list=[],
		optimizer='SGD',
		opt_params=[],
	    activation=None,
		act_params=[],
		initializer=None,
		init_params=[],
		loss: str='mean_squared_error',
		filename=None
	):
		self.topology = topology
		self.opt_params = []
		self.activations = []
		self.act_params = []
		self.initializers = []
		self.init_params = []
		self.loss = loss
		self.optimizer_name = optimizer
		self.normalization = 'Standard'
		self.normalization_params = []
		self.history = None

		# load model from given file, ignore all else
		if filename is not None:
			self.set_model_from_file(filename)
			return

		# determine the initializers
		if initializer == None:
			self.initializers = [ki.normal()
			                     for i in range(len(self.topology)-1)]
		elif isinstance(initializer,str):
			if(initializer == 'zeros'):
				self.initializers = [ki.zeros()
			                         for i in range(len(self.topology)-1)]
			elif(initializer == 'ones'):
				self.initializers = [ki.ones()
			                         for i in range(len(self.topology)-1)]
			elif(initializer == 'constant'):
				if len(init_params) == 0:
					self.initializers = [ki.constant()
			                             for i in range(len(self.topology)-1)]
				else:
					assert len(init_params) == 1,"Must provide 1 parameter \
						                          for constant initialization!"
					self.initializers = [ki.constant(value=init_params[0])
			                         for i in range(len(self.topology)-2)]
			elif(initializer == 'normal'):
				if len(init_params) == 0:
					self.initializers = [ki.normal()
			                             for i in range(len(self.topology)-1)]
				else:
					assert len(init_params) == 3,"Must provide 3 parameters \
						                          for normal initialization!"
					self.initializers = [ki.normal(mean=init_params[0],
				                                   stddev=init_params[1],
												   seed=init_params[2])
			                         for i in range(len(self.topology)-2)]
			elif(initializer == 'uniform'):
				if len(init_params) == 0:
					self.initializers = [ki.uniform()
			                             for i in range(len(self.topology)-1)]
				else:
					assert len(init_params) == 3,"Must provide 3 parameters \
						                          for uniform initialization!"
					self.initializers = [ki.uniform(minval=init_params[0],
				                                    maxval=init_params[1],
													seed=init_params[2])
			                         for i in range(len(self.topology)-2)]
			elif(initializer == 'truncated_normal'):
				if len(init_params) == 0:
					self.initializers = [ki.truncated_normal()
			                             for i in range(len(self.topology)-1)]
				else:
					assert len(init_params) == 3,"Must provide 3 parameters \
						                          for truncated normal \
												  initialization!"
					self.initializers = [ki.truncated_normal(init_params[0],
				                                             init_params[1],
															 init_params[2])
			                         for i in range(len(self.topology)-1)]
			elif(initializer == 'variance_scaling'):
				if len(init_params) == 0:
					self.initializers = [ki.variance_scaling()
			                             for i in range(len(self.topology)-1)]
				else:
					assert len(init_params) == 4,"Must provide 4 parameters \
												  for variance scaling \
												  initialization!"
					assert isinstance(opt_params[1],str), "Parameter 2 must \
						                                   be of type string!"
					assert isinstance(opt_params[2],str), "Parameter 3 must \
						                                   be of type string!"
					assert (opt_params[1] == 'fan_in' or opt_params[1] ==
					        'fan_out' or opt_params[1] == 'fan_avg'), "Mode \
							must be either 'fan_in', 'fan_out' or 'fan_avg'!"
					assert (opt_params[2] == 'normal' or opt_params[2] ==
					        'uniform'), "Distribution must be either 'normal'\
								         or 'uniform'!"
					self.initializers = [ki.variance_scaling(init_params[0],
				                                             init_params[1],
															 init_params[2],
															 init_params[3])
			                         for i in range(len(self.topology)-1)]
			elif(initializer == 'orthogonal'):
				if len(init_params) == 0:
					self.initializers = [ki.orthogonal()
			                             for i in range(len(self.topology)-1)]
				else:
					assert len(init_params) == 2,"Must provide 2 parameters \
												  for orthogonal \
					                              initialization!"
					self.initializers = [ki.orthogonal(gain=opt_params[0],
				                                       seed=opt_params[1])
									 for i in range(len(self.topology)-1)]
		elif isinstance(initializer,list):
			assert len(initializer) == len(self.topology)-1,"Must provide an \
															initializer for \
															each layer!"
			assert len(init_params) == len(self.topology)-1,"Must provide \
				                                            params for \
				                                            each layer \
															initializer!"
			for j in range(len(initializer)):
				if(initializer[j] == 'zeros'):
					self.initializers.append(ki.zeros())
				elif(initializer[j] == 'ones'):
					self.initializers.append(ki.ones())
				elif(initializer[j] == 'constant'):
					if len(init_params[j]) == 0:
						self.initializers.append(ki.constant())
					else:
						assert len(init_params[j]) == 1,"Must provide 1 \
							                             parameter for \
														 constant \
														 initialization!"
						self.initializers.append(ki.constant(
													      init_params[j][0]))
				elif(initializer[j] == 'normal'):
					if len(init_params[j]) == 0:
						self.initializers.append(ki.normal())
					else:
						assert len(init_params[j]) == 3,"Must provide 3 \
														 parameters for \
														 normal \
														 initialization!"
						self.initializers.append(ki.normal(init_params[j][0],
														   init_params[j][1],
														   init_params[j][2]))
				elif(initializer[j] == 'uniform'):
					if len(init_params[j]) == 0:
						self.initializers.append(ki.uniform())
					else:
						assert len(init_params[j]) == 3,"Must provide 3 \
							                             parameters for \
														 uniform \
														 initialization!"
						self.initializers.append(ki.uniform(init_params[j][0],
															init_params[j][1],
															init_params[j][2]))
				elif(initializer[j] == 'truncated_normal'):
					if len(init_params[j]) == 0:
						self.initializers.append(ki.truncated_normal())
					else:
						assert len(init_params[j]) == 3,"Must provide 3 \
							                             parameters for \
														 truncated normal\
														 initialization!"
						self.initializers.append(ki.truncated_normal(
															init_params[j][0],
															init_params[j][1],
															init_params[j][2]))
				elif(initializer[j] == 'variance_scaling'):
					if len(init_params[j]) == 0:
						self.initializers.append(ki.variance_scaling())
					else:
						assert len(init_params[j]) == 4,"Must provide 4 \
														 parameters for \
														 variance scaling\
														 initialization!"
						assert isinstance(opt_params[1],str),"Parameter 2\
							                                  must be of\
															  type string!"
						assert isinstance(opt_params[2],str),"Parameter 3\
															  must be of\
															  type string!"
						assert (opt_params[1] == 'fan_in' or
						        opt_params[1] == 'fan_out' or
							    opt_params[1] == 'fan_avg'),"Mode must be\
									                         either 'fan_in',\
														     'fan_out' \
															 or 'fan_avg'!"
						assert (opt_params[2] == 'normal' or
						        opt_params[2] == 'uniform'),"Distribution\
									                         must be either\
														     'normal' or \
															 'uniform'!"
						self.initializers.append(ki.variance_scaling(
												scale=init_params[j][0],
												mode=init_params[j][1],
												distribution=init_params[j][2],
												seed=init_params[j][3]))
				elif(initializer[j] == 'orthogonal'):
					if len(init_params[j]) == 0:
						self.initializers.append(ki.orthogonal())
					else:
						assert len(init_params[j]) == 2,"Must provide 2 \
							                             parameters for \
														 orthogonal \
														 initialization!"
						self.initializers.append(ki.orthogonal(
							                        gain=opt_params[0],
													seed=opt_params[1]))

		# set optimizer using optimizer_name
		self.set_optimizer()

		# determine the activations
		if activation == None:
			# set all activations to tanh
			self.activations = ['tanh' for i in range(len(self.topology))]
		elif isinstance(activation,str):
			assert (activation in ['tanh','elu','selu','softplus',
			                       'softsign','relu','sigmoid','hard_sigmoid',
								   'exponential','linear']), "Activation \
								   must be one of the allowed types!"
			self.activations = [activation for i in range(len(self.topology))]
			if (activation == 'elu'):
				if len(act_params) == 0:
					self.act_params = [[1.0] for i in range(len(self.topology))]
				else:
					assert len(act_params) == 1, "Must provide 1 parameter for\
						                          elu activation!"
					self.act_params = [act_params[0] for i in
					                   range(len(self.topology))]
			if (activation == 'relu'):
				if len(act_params) == 0:
					self.act_params = [[0.0,None,0.0] for i in
					                   range(len(self.topology))]
				else:
					assert len(act_params) == 3,"Must provide 3 parameters for\
						                         relu activation!"
					self.act_params = [act_params for i in
					                   range(len(self.topology))]
		elif isinstance(activation,list):
			assert len(activation) == len(self.topology),"Number of activations\
														  must equal the \
				                                          number of layers!"
			for j in range(len(activation)):
				assert (activation[j] in ['tanh','elu','selu',
				                          'softplus','softsign','relu',
										'sigmoid','hard_sigmoid','exponential',
										'linear']),"Activation must be one of \
											        the allowed types!"
				self.activations.append(activation[j])
				if (activation[j] == 'elu'):
					if len(act_params[j]) == 0:
						self.act_params.append([1.0])
					else:
						assert len(act_params[j]) == 1,"Must provide 1 \
														parameter \
														for elu activation!"
						self.act_params.append(act_params[0])
				elif (activation[j] == 'relu'):
					if len(act_params[j]) == 0:
						self.act_params.append([0.0,None,0.0])
					else:
						assert len(act_params[j]) == 3,"Must provide 3 \
								                            parameters for\
														relu activation!"
						self.act_params.append(act_params[j])
				else:
					self.act_params.append([])

		self.build_model()
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	set optimizer using self.optimizer_name
	#-------------------------------------------------------------------------
	def set_optimizer(self):
	# determine the optimizer and its parameters
		if (self.optimizer_name == 'SGD'):
			# check opt_params
			if len(opt_params) == 0:
				self.optimizer = optimizers.SGD()
				self.opt_params = [0.01,0.0,0.0,False]
			else:
				assert len(self.opt_params) == 4, "Must provide 4 parameters for \
					                          SGD!"
				assert isinstance(self.opt_params[3],bool), "Parameter 4 (nesterov\
														acceleration) must be\
													    of type bool!"
				self.optimizer = optimizers.SGD(lr=self.opt_params[0],
												momentum=self.opt_params[1],
												decay=self.opt_params[2],
												nesterov=self.opt_params[3])
		elif (self.optimizer_name == 'RMSprop'):
			# check opt_params
			if len(self.opt_params) == 0:
				self.optimizer = optimizers.RMSprop()
				self.opt_params = [0.001,0.9,'None',0.0]
			else:
				assert len(self.opt_params) == 4, "Must provide 4 parameters for \
					                          RMSprop!"
				self.optimizer = optimizers.RMSprop(lr=self.opt_params[0],
													rho=self.opt_params[1],
													epsilon=self.opt_params[2],
													decay=self.opt_params[3])
		elif (self.optimizer_name == 'Adagrad'):
			# check opt_params
			if len(self.opt_params) == 0:
				self.optimizer = optimizers.Adagrad()
				self.opt_params = [0.01,'None',0.0]
			else:
				assert len(self.opt_params) == 3, "Must provide 3 parameters for \
					                          Adagrad!"
				self.optimizer = optimizers.Adagrad(lr=self.opt_params[0],
													epsilon=self.opt_params[1],
													decay=self.opt_params[2])
		elif (self.optimizer_name == 'Adadelta'):
			# check opt_params
			if len(self.opt_params) == 0:
				self.optimizer = optimizers.Adadelta()
				self.opt_params = [1.0,0.95,'None',0.0]
			else:
				assert len(self.opt_params) == 4, "Must provide 4 parameters for \
					                          Adadelta!"
				self.optimizer = optimizers.Adadelta(lr=self.opt_params[0],
				                                     rho=self.opt_params[1],
													 epsilon=self.opt_params[2],
													 decay=self.opt_params[3])
		elif (self.optimizer_name == 'Adam'):
			# check opt_params
			if len(self.opt_params) == 0:
				self.optimizer = optimizers.Adam()
				self.opt_params = [0.001,0.9,0.999,'None',0.0,False]
			else:
				assert len(self.opt_params) == 6, "Must provide 6 parameters for \
					                          Adam!"
				assert isinstance(self.opt_params[5],bool), "Parameter 6 (amsgrad)\
					                                    must be of type bool!"
				self.optimizer = optimizers.Adam(lr=self.opt_params[0],
				                                 beta_1=self.opt_params[1],
												 beta_2=self.opt_params[2],
												 epsilon=self.opt_params[3],
												 decay=self.opt_params[4],
												 amsgrad=self.opt_params[5])
		elif (self.optimizer_name == 'Adamax'):
			# check opt_params
			if len(self.opt_params) == 0:
				self.optimizer = optimizers.Adamax()
				self.opt_params = [0.002,0.9,0.999,'None',0.0]
			else:
				assert len(self.opt_params) == 5, "Must provide 5 parameters for \
					                          Adamax!"
				self.optimizer = optimizers.Adamax(lr=self.opt_params[0],
				                                   beta_1=self.opt_params[1],
												   beta_2=self.opt_params[2],
												   epsilon=self.opt_params[3],
												   decay=self.opt_params[4])
		elif (self.optimizer_name == 'Nadam'):
			# check opt_params
			if len(self.opt_params) == 0:
				self.optimizer = optimizers.Nadam()
				self.opt_params = [0.002,0.9,0.999,'None',0.004]
			else:
				assert len(self.opt_params) == 5, "Must provide 5 parameters for \
					                          Nadam!"
				self.optimizer = optimizers.Nadam(lr=self.opt_params[0],
				                                  beta_1=self.opt_params[1],
												  beta_2=self.opt_params[2],
												  epsilon=self.opt_params[3],
												  schedule_decay=self.opt_params[4])
	#-------------------------------------------------------------------------


	#-------------------------------------------------------------------------
	#   build and compile sequential model
	#-------------------------------------------------------------------------
	def build_model(self):
		# build the sequential model
		self.model = models.Sequential()
		# now we build and compile the model
		# no biases from the input layer, since the inputs are physical
		self.model.add(layers.Dense(self.topology[1],input_dim=self.topology[0],
				       kernel_initializer=self.initializers[0],
					   use_bias=True))
		self.num_additions = 0
		for i in range( 1, len( self.topology ) - 1 ):
			# This "layer" object applies the activation from the output
			# of the previous
			if(self.activations[i] not in ['elu','relu']):
				self.model.add(layers.Activation(self.activations[i]))
			elif(self.activations[i] == 'elu'):
				self.model.add(layers.ELU(self.act_params[i]))
			elif(self.activations[i] == 'relu'):
				self.model.add(layers.ReLU(self.act_params[i][1],
								           self.act_params[i][0],
										   self.act_params[i][2]))
			#	Adding the next layer
			self.model.add(layers.Dense(self.topology[i+1],
			                     kernel_initializer=self.initializers[i],
								 use_bias=True))
			self.num_additions += 2
		if(self.activations[-1] not in ['elu','relu']):
			self.model.add(layers.Activation(self.activations[-1]))
		elif(self.activations[-1] == 'elu'):
			self.model.add(layers.ELU(self.act_params[-1]))
		elif(self.activations[-1] == 'relu'):
			self.model.add(layers.ReLU(self.act_params[-1][1],
									   self.act_params[-1][0],
									   self.act_params[-1][2]))
		#	We want to retrieve the values from the output
		self.output_function = K.function([self.model.layers[0].input],
		                    [self.model.layers[self.num_additions + 1].output])
		# now compile the model
		self.model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])

	#-------------------------------------------------------------------------
	#   get_layer_weights(self,layer - layer is an index to the desired layer)
	#-------------------------------------------------------------------------
	def get_layer_weights(self,layer):
		if layer >= len(self.topology):
			print("ERROR! index %s exceeds number of layers %s!" % (layer,len(self.topology)))
			return 0
		# check that 'layer' is an actual layer with weights
		try:
			return self.model.layers[layer].get_weights()[0].tolist()
		except:
			print("ERROR! layer %s is not a layer with weights!" % layer)
			return 0
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#   get_layer_biases(self,layer - layer is an index to the desired layer)
	#-------------------------------------------------------------------------
	def get_layer_biases(self,layer):
		if layer >= len(self.topology):
			print("ERROR! index %s exceeds number of layers %s!" % (layer,len(self.topology)))
			return 0
		# check that 'layer' is an actual layer with weights
		try:
			return self.model.layers[layer].get_weights()[1].tolist()
		except:
			print("ERROR! layer %s is not a layer with weights!" % layer)
			return 0
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	get weights
	#-------------------------------------------------------------------------
	def get_weights(self):
		weights = []
		for i in range(len(self.model.layers)):
			try:
				weights.append(self.model.layers[i].get_weights()[0].tolist())
			except:
				continue
		return weights
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	set weights
	#-------------------------------------------------------------------------
	def set_weights(self, weights):
		try:
			self.model.set_weights(weights)
		except:
			return
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	set weights from file
	#-------------------------------------------------------------------------
	def set_weights_from_file(self, weights_file, biases_file):
		weights = []
		biases = []
		with open(weights_file,"r") as file:
			reader = csv.reader(file,delimiter=",")
			topology = next(reader)
			for row in reader:
				weights.append([float(row[i]) for i in range(len(row))])
		with open(biases_file,"r") as file:
			reader = csv.reader(file,delimiter=",")
			next(reader)
			for row in reader:
				biases.append([float(row[i]) for i in range(len(row))])
		topology = [int(topology[i]) for i in range(len(topology))]
		new_weights = []
		index_left = 0
		index_right = topology[0]
		for j in range(len(topology)-1):
			new_weights.append(np.asarray([weights[l] for l in range(index_left,index_right)]))
			if j < len(topology) - 1:
				new_weights.append(np.asarray(biases[j]))
			index_left = index_right
			index_right += topology[j+1]
		try:
			self.model.set_weights(new_weights)
		except:
			return
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	get biases
	#-------------------------------------------------------------------------
	def get_biases(self):
		biases = []
		for i in range(len(self.model.layers)):
			try:
				biases.append(self.model.layers[i].get_weights()[1].tolist())
			except:
				continue
		return biases
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	save_weights_to_file(self,filename - name to store the weights)
	#-------------------------------------------------------------------------
	def save_weights_to_file(self,filename):
		weights = self.get_weights()
		weights_to_save = [self.topology]
		for i in range(len(weights)):
			for j in range(len(weights[i])):
				weights_to_save.append(weights[i][j])
		with open(filename,"w") as file:
			writer = csv.writer(file,delimiter=",")
			writer.writerows(weights_to_save)
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	save_biases_to_file(self,filename - name to store the biases)
	#-------------------------------------------------------------------------
	def save_biases_to_file(self,filename):
		biases = self.get_biases()
		biases_to_save = [self.topology]
		for i in range(len(biases)):
			biases_to_save.append(biases[i])
		with open(filename,"w") as file:
			writer = csv.writer(file,delimiter=",")
			writer.writerows(biases_to_save)
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	save_model
	#-------------------------------------------------------------------------
	def save_model(self,filename):
		params = [['#Topology'],self.topology,
			      ['#Optimizer'],[self.optimizer_name],
				  ['#OptParams'],self.opt_params,
				  ['#Activations'],self.activations,
				  ['#ActParams']]
		for j in range(len(self.act_params)):
			if len(self.act_params[j]) == 0:
				params.append(['None'])
			else:
				params.append(self.act_params[j])
		params.append(['#Loss']),
		params.append([self.loss])
		params.append(['#Weights'])
		weights = self.get_weights()
		for i in range(len(weights)):
			for j in range(len(weights[i])):
				params.append(weights[i][j])
		params.append(['#Biases'])
		biases = self.get_biases()
		for i in range(len(biases)):
			params.append(biases[i])
		params.append(['#Normalization'])
		params.append([self.normalization])
		params.append(['#NormalizationParams'])
		for i in range(len(self.normalization_params)):
			params.append(self.normalization_params[i])
		with open(filename,"w") as file:
			writer = csv.writer(file,delimiter=',')
			writer.writerows(params)
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	set_model_from_file
	#-------------------------------------------------------------------------
	def set_model_from_file(self,filename):
		params = []
		with open(filename,"r") as file:
			reader = csv.reader(file,delimiter=",")
			for row in reader:
				params.append(row)
		# iterate over each row starting with topology
		self.topology = [int(params[1][i]) for i in range(len(params[1]))]
		self.optimizer_name = params[3][0]
		# set optimizer using optimizer_name
		self.set_optimizer()
		self.opt_params = params[5]
		self.activations = params[7]
		self.act_params = [params[9+i] for i in range(0,len(self.activations))]
		self.loss = params[10+len(self.activations)][0]
		# now for the weights
		weights = []
		biases = []
		weights_start = 12 + len(self.activations)
		num_weights = int(sum([self.topology[i] for i in range(len(self.topology)-1)]))
		for j in range(weights_start,weights_start + num_weights):
			weights.append([float(params[j][l]) for l in range(len(params[j]))])
		biases_start = 13 + len(self.activations) + len(weights)
		for j in range(biases_start,biases_start + len(self.topology)-1):
			biases.append([float(params[j][l]) for l in range(len(params[j]))])
		new_weights = []
		index_left = 0
		index_right = self.topology[0]
		for j in range(len(self.topology)-1):
			new_weights.append(np.asarray([weights[l]
							   for l in range(index_left,index_right)]))
			if j < len(self.topology) - 1:
				new_weights.append(np.asarray(biases[j]))
			index_left = index_right
			index_right += self.topology[j+1]

		# set initializers to default
		if not self.initializers:
			self.initializers = [ki.normal()
			                     for i in range(len(self.topology)-1)]

		self.build_model()

		self.model.set_weights(new_weights)

		# finally get normalization parameters
		final = 14 + len(self.activations) + len(weights) + len(biases)
		self.normalization = params[final][0]
		if final < len(params)+2:
			for j in range(final+2,len(params)):
				self.normalization_params.append([float(params[j][l])
				                for l in range(len(params[j]))])
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	find_normalization_parameters(self,data)
	#-------------------------------------------------------------------------
	def find_normalization_parameters(self,data):
		# determine the normalization parameters
		self.normalization_params = []
		if (self.normalization == 'Standard'):
			for j in range(len(data[0])):
				temp_var = [data[i][j] for i in range(len(data))]
				self.normalization_params.append([np.mean(temp_var),
				                                  np.std(temp_var)])
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	normalize_data(self,data)
	#-------------------------------------------------------------------------
	def normalize_data(self,data):
		# determine the normalization parameters
		if (self.normalization == 'Standard'):
			nps = self.normalization_params
			data = [[(data[i][j]-nps[j][0])/nps[j][1] for j in
			         range(len(data[0]))] for i in range(len(data))]
		return data
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	set_classes(self,answer)
	#								'tanh' (default)- [-1.0,1.0]
	#								'elu'  			- [-1.0,inf]
	#								'selu'          - [-1.673,inf]
	#                               'softplus'      - [0.0,inf]
	#                               'softsign'      - [-inf,inf]
	#                               'relu'          - [0.0,inf]
	#                               'sigmoid'       - [0.0,1.0]
	#                               'hard_sigmoid'  - [0.0,1.0]
	#                               'exponential'   - [0.0,inf]
	#                               'linear'		- [-inf,inf]
	#-------------------------------------------------------------------------
	def set_classes(self,answer):
		# change to the correct expected output
		new_answer = []
		if (self.activations[-1] == 'tanh'):
			return np.copy(answer)
		elif (self.activations[-1] == 'elu'):
			for j in range(len(answer)):
				if(answer[j][0] == 1.0):
					new_answer.append([1.0])
				else:
					new_answer.append([-1.0])
		elif (self.activations[-1] == 'selu'):
			for j in range(len(answer)):
				if(answer[j][0] == 1.0):
					new_answer.append([1.0])
				else:
					new_answer.append([-1.673])
		elif (self.activations[-1] == 'softplus' or
			  self.activations[-1] == 'relu' or
			  self.activations[-1] == 'exponential'):
			for j in range(len(answer)):
				if(answer[j][0] == 1.0):
					new_answer.append([1.0])
				else:
					new_answer.append([0.0])
		elif (self.activations[-1] == 'softsign' or
		      self.activations[-1] == 'linear'):
			for j in range(len(answer)):
				if(answer[j][0] == 1.0):
					new_answer.append([1.0])
				else:
					new_answer.append([-1.0])
		elif (self.activations[-1] == 'sigmoid' or
		      self.activations[-1] == 'hard_sigmoid'):
			for j in range(len(answer)):
				if(answer[j][0] == 1.0):
					new_answer.append([1.0])
				else:
					new_answer.append([0.0])
		return new_answer
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	train(self, training_data   - should be a list of lists
	# 				training_answer - a single list of output values
	#               num_epochs      - number of training epochs
	#               batch           - number of events for batch training
	#               type_norm       - normalization type
	#               sample_weights  - possible weights for samples
	#-------------------------------------------------------------------------
	def train(self,
		      training_data,
			  training_answer,
			  validation_split=0.25,
			  num_epochs=1,
			  batch=256,
			  sample_weights=None
	):
		train_data = np.copy(training_data)
		# Anytime we are training a network, we must renormalize
		# according to the data
		self.find_normalization_parameters(training_data)
		train_data = self.normalize_data(train_data)
		# set the training answer to match the expected output
		train_answer = self.set_classes(training_answer)
		# training session
		print("training model...")
		self.history = self.model.fit(np.array(train_data), np.array(train_answer),
		               validation_split=validation_split,
					   epochs=num_epochs, batch_size=batch,
					   sample_weight=sample_weights)
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	evaluate(self, testing_data   - should be a list of lists
	#                  testing_answer - a single list of output values
	#                  score_output   - whether to display the score
	#                  batch          - batch amount for the score
	#-------------------------------------------------------------------------
	def evaluate(self,
				 testing_data,
				 testing_answer,
				 score_output=True,
				 batch=256
	):
		test_data = np.copy(testing_data)
		#	We don't want to normalize the actual testing data, only a copy of it
		test_data = self.normalize_data(test_data)
		# set the testing answer to match the expected output
		test_answer = self.set_classes(testing_answer)
		if (score_output == True):
			score = self.model.evaluate(np.array(test_data),
			                            np.array(test_answer),
										batch_size=batch,
										verbose=0)
			#	Prints a score for the network based on the training data
			print('Score: %s' % (score))
		activations = self.output_function([test_data])
		return [[activations[0][i][0], test_answer[i][0]]
			     for i in range(len(testing_answer))]
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	save_results_to_file(self, filename)
	#-------------------------------------------------------------------------
	def save_results_to_file(self,
							 filename,
							 data,
							 results
	):
		events = np.concatenate((data,results),axis=1)
		with open(filename,"w") as file:
			writer = csv.writer(file,delimiter=",")
			writer.writerows(events)
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	plot_history(self)
	#-------------------------------------------------------------------------
	def plot_history(self, show=True, save=True, filename='History'):
		# Plot training & validation accuracy values
		fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(11,5),
		                        constrained_layout=True)
		ax = axs[0]
		ax.plot(self.history.history['acc'])
		ax.plot(self.history.history['val_acc'])
		ax.set_title('Model accuracy')
		ax.set_ylabel('Accuracy')
		ax.set_xlabel('Epoch')
		ax.legend(['Train', 'Test'], loc='upper left')

		# Plot training & validation loss values
		ax = axs[1]
		ax.plot(self.history.history['loss'])
		ax.plot(self.history.history['val_loss'])
		ax.set_title('Model loss')
		ax.set_ylabel('Loss')
		ax.set_xlabel('Epoch')
		ax.legend(['Train', 'Test'], loc='upper left')
		fig.suptitle('Training/Validation Error and Accuracy vs. Epoch')
		if(save):
			plt.savefig(filename + '.png')
		if(show):
			plt.show()
	#-------------------------------------------------------------------------

	#-------------------------------------------------------------------------
	#	plot_auc(self, results)
	#-------------------------------------------------------------------------
	def plot_auc(self, results, show=True, save=True, filename='AccRej'):
		# determine the range of results values
		outputs = [results[i][0] for i in range(len(results))]
		answers = [results[i][1] for i in range(len(results))]
		max_val = np.max(outputs)
		min_val = np.min(outputs)
		# generate bins
		bin_width = (max_val - min_val)/100.0
		bin_edges = [(min_val + i*bin_width) for i in range(101)]
		# create signal and background lists
		sig_events = [outputs[i] for i in range(len(outputs))
					  if answers[i] == 1.0]
		back_events = [outputs[i] for i in range(len(outputs))
		               if answers[i] == -1.0]
		# generate histograms
		sig_hist, back_hist = [0 for i in range(100)], [0 for i in range(100)]
		for i in range(len(sig_events)):
			for j in range(len(sig_hist)):
				if (sig_events[i] >= bin_edges[j] and
				    sig_events[i] < bin_edges[j+1]):
				   sig_hist[j] += 1
				   continue
		for i in range(len(back_events)):
			for j in range(len(back_hist)):
				if (back_events[i] >= bin_edges[j] and
				    back_events[i] < bin_edges[j+1]):
				   back_hist[j] += 1
				   continue
		# determine sig/back acc/rej
		sig_acc, back_acc, = [1.0], [1.0]
		sig_rej, back_rej = [10e-15], [10e-15]
		for i in range(len(sig_hist)):
			sig_acc.append(sig_acc[-1] - sig_hist[i]/len(sig_events))
			back_acc.append(back_acc[-1] - back_hist[i]/len(back_events))
			sig_rej.append(sig_rej[-1] + sig_hist[i]/len(sig_events))
			back_rej.append(back_rej[-1] + back_hist[i]/len(back_events))
		# calculate AUC's
		acc_acc_auc = 0.0
		rej_acc_auc = 0.0
		for i in range(1,len(sig_hist)):
			acc_acc_auc += (1/100) * (back_acc[i]+back_acc[i-1])/2.0
			rej_acc_auc += (1/100) * (back_rej[i]+back_rej[i-1])/2.0
		# Plot acceptance/acceptance curve
		fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(11,5),
								constrained_layout=True)
		ax = axs[0]
		ax.plot(sig_acc,back_acc,color='k',linestyle='--',
				label=r'AUC $\sim$ %.4f' % acc_acc_auc)
		ax.set_title('Background Acc. vs. Signal Acc.')
		ax.set_ylabel('Background Acceptance')
		ax.set_yscale('log')
		ax.set_xlabel('Signal Acceptance')
		#ax.set_xscale('log')
		ax.legend()

		# Plot rejection/acceptance curve
		ax = axs[1]
		ax.plot(sig_acc,back_rej,color='k',linestyle='--',
				label=r'AUC $\sim$ %.4f' % rej_acc_auc)
		ax.set_title('Background Rej. vs. Signal Acc.')
		ax.set_ylabel('Background Rejection')
		ax.set_yscale('log')
		ax.set_xlabel('Signal Acceptance')
		#ax.set_xscale('log')
		ax.legend()
		fig.suptitle('Signal/Background Accept/Reject Curves')
		if(save):
			plt.savefig(filename + '.png')
		if(show):
			plt.show()
	#-------------------------------------------------------------------------
#-----------------------------------------------------------------------------

if __name__ == "__main__":
	# create npz file of root data
	convert_root_to_npz('Data/golden_backgrounds',
	                    'golden',
						['r_cm','z_cm','S1c','log10S2c','weight'])
	convert_root_to_npz('Data/golden_signal_WIMP_10GeV_Run3',
	                    'golden',
						['r_cm','z_cm','S1c','log10S2c','weight'])
	# generate testing and training data
	data = generate_binary_training_testing_data(
		'Data/golden_signal_WIMP_10GeV_Run3.npz',
		'Data/golden_backgrounds.npz',
		labels=['arr_0','arr_0'],
		symmetric_signals=True,
		testing_ratio=0.0,
		var_set=[0,1,2,3])
	# setup mlp
	mlp = MLP([4,10,3,1],
			  optimizer='Nadam',
			  activation=['hard_sigmoid','tanh','sigmoid','softsign'],
			  initializer=['normal','normal','normal'],
			  init_params=[[0.0,1.0,0],[0.0,1.0,1],[0.0,1.0,5.0]])
	mlp.save_weights_to_file('weights_before.csv')
	mlp.save_biases_to_file('biases_before.csv')
	mlp.save_model('model_before.csv')
	mlp.set_model_from_file('model_after.csv')
	mlp.save_model('new_model.csv')
	# train mlp
	mlp.train(data[0],data[1],num_epochs=50,batch=25)
	mlp.save_weights_to_file('weights_after.csv')
	mlp.save_biases_to_file('biases_after.csv')
	mlp.save_model('model_after.csv')
	# test mlp
	mlp.plot_history()
	results = mlp.evaluate(data[0],data[1])
	mlp.plot_auc(results)
	# save test results
	test_results = mlp.evaluate(data[0][:10],data[1][:10])
	mlp.save_results_to_file('results.csv',data[0][:10],test_results)

