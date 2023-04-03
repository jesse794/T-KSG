# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# #### Example Notebook
#
# Create a dateset with five Gaussian variables and one spike variable, then document how the code works by demonstrating various features.
#
#     - Shows how to create a run object and perform playouts on it.
#     - Shows how to add additional playouts onto existing run object.
#     - Shows the full-tree mode, the compressed-tree mode, and the fixed path mode.
#     - Shows how to read/write run objects to disk using shelf module so that one can add more playouts later.
#     - Shows how to automatically take a run's top n playouts and rerun the paths with more events.

# Running in Python
import time
import sys
sys.path.append("../../src")
import CTreeMI as ct
import MI
import numpy as np


# +
# Make a toy dataset for testing

nSig = 5000
nBkd = 5000

sigV0 = [[np.random.normal(0, 1)] for i in range(nSig)]
sigV1 = [[np.random.normal(0.25, 1)] for i in range(nSig)]
sigV2 = [[np.random.normal(0.5, 1)] for i in range(nSig)]
sigV3 = [[np.random.normal(0.75, 1)] for i in range(nSig)]
sigV4 = [[np.random.normal(1.0, 1)] for i in range(nSig)]
sigV5 = [[np.random.normal(1.25, 1)] for i in range(nSig)]
sigV6 = [[-999] for i in range(nSig)]

bkdV0 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV1 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV2 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV3 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV4 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV5 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV6 = [[-999] for i in range(nBkd)]

sigVars = MI.joint_space(sigV0, sigV1, sigV2, sigV3, sigV4, sigV5, sigV6)
bkdVars = MI.joint_space(bkdV0, bkdV1, bkdV2, bkdV3, bkdV4, bkdV5, bkdV6)

sigWts = [[1.0] for i in range(nSig)]
bkdWts = [[1.0] for i in range(nBkd)]
varNums = [0, 1, 2, 3, 4, 5, 6]
varNames = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5', 'var6']
setName = 'test Gaussian'
myDataset = ct.dataset(setName, sigVars, bkdVars, sigWts, bkdWts, varNums, varNames)
# -
MI.mi(sigV6, bkdV6,1,2) # just a sanity-check on the continuous MI function.

# Create a run object and pass it a dataset.
# Alternatively, you can read back a run object that you stored from a previous running.
# Uncomment one of the following two lines.
myRun = ct.run(myDataset, gateEvalParm=50, nParallelParm=8)
# myRun = ct.shelfRun(filename='ctree_shelf.db', mode='read', key='example')

# +
# To launch playouts, you need to pass it a run that you've created, the type of playouts you want, and a
# number of playout to execute.

# type [-1] is compressed tree.

# type [-2] is full tree, though whether it behaves as a standard full-tree depends on the gatepolicy value, which
# is passed as the second argument when creating the run object.  See full documentation for details.

# Any other list for type is interpreted as fixed-path running.  So, for example, passing [1,2,5] will run
# all playouts through variables 1,2,5 only (counting from zero).

ct.playouts(myRun, eventsPerPlayout=1000, pathType=[-2], cpParm=0.7071, numPlays=8)

# +
# Note that once a run object is created, you can continue to run playouts on it, and it will continue to accrue
# statistics from the playouts.  Here, we run some fixed-path playouts

ct.playouts(myRun,0,[0], numPlays=1)
ct.playouts(myRun,0,[1], numPlays=1)
ct.playouts(myRun,0,[2], numPlays=1)
ct.playouts(myRun,0,[3], numPlays=1)
ct.playouts(myRun,0,[4], numPlays=1)
ct.playouts(myRun,0,[5], numPlays=1)
ct.playouts(myRun,0,[6], numPlays=1)
# -

# Report text information for the current run.
# The report results will be cumulative if you later do more playouts using the same run.
ct.textReports(myRun,playoutsToReport=0,includeGateInfo=True)

# Report plot-based information for the current run.
# The report results will be cumulative if you later do more playouts using the same run.
ct.plotReports(myRun,'myPlots_fullTree.pdf',includeGateInfo=True)

# Write the run object (which has all completed playouts) to a file so you can load it back in later
# and continue playouts from where you left off.
ct.shelfRun(filename='ctree_shelf', mode='write', key='example', myRunIn=myRun)

# +
# uncomment if you want to delete the persisted run object in the shelf file.
# ct.shelfRun(filename='ctree_shelf', mode='delete', key='example')
# -

# #### Create a new run object so you can run other playouts without affecting the previous set of playouts

# Create a run object and pass it a dataset
myRun2 = ct.run(myDataset, gateEvalParm=50, nParallelParm=8)

# +
# To launch playouts, you need to pass it a run that you've created, the type of playouts you want, and a
# number of playout to execute.

# type [-1] is compressed tree.

# type [-2] is full tree, though whether it behaves as a standard full-tree depends on the gatepolicy value, which
# is passed as the second argument when creating the run object.  See full documentation for details.

# Any other list for type is interpreted as fixed-path running.  So, for example, passing [1,2,5] will run
# all playouts through variables 1,2,5 only (counting from zero).

ct.playouts(myRun2,eventsPerPlayout=1000,pathType=[-1],numPlays=8)

# +
# Note that once a run object is created, you can continue to run playouts on it, and it will continue to accrue
# statistics from the playouts.

for i in range(4):
    ct.playouts(myRun2, eventsPerPlayout=1000, pathType=[-1], numPlays=8)
    if (i+1) % 4 == 0: ct.treeNodesReport(myRun2)
# -

# Report text information for the current run.
# The report results will be cumulative if you later do more playouts using the same run.
ct.textReports(myRun2,playoutsToReport=0,includeGateInfo=True)

# Report plot-based information for the current run.
# The report results will be cumulative if you later do more playouts using the same run.
ct.plotReports(myRun2,'myPlots_compressedTree.pdf',includeGateInfo=True)

# #### Read back the old run object from file, add more playouts to it, then write it back to file.

# Read the first run object back off the shelf.
myRunReadBack = ct.shelfRun(filename='ctree_shelf', mode='read', key='example')

# Run a few more playouts on the read-back run object, print updated information, then rewrite the object to file.
# Writing and then reading-back run objects allows true continuation of runs.
ct.playouts(myRunReadBack, eventsPerPlayout=1000, pathType=[-2], cpParm=0.7071, numPlays=8)
ct.textReports(myRunReadBack,playoutsToReport=0, includeGateInfo=True)
ct.shelfRun(filename='ctree_shelf', mode='write', key='example', myRunIn=myRunReadBack)

# +
# Note that the shelf file will continue to grow each time you write back to it.  
# This can be a problem in cases where the output is large.  And because the run object 
# has a copy of the dataset, the output will be large in any all cases where the dataset
# is large.
# Depending on the underlying DB implementation, there are two ways of handling this.
#  1) If the implementation allows for db reorganization, just execute the following on the file:
#     (Be sure the file has been properly closed or you'll get an error when opening it.)
#     import dbm
#     db=dbm.open('myfile.db',flag='w')
#     db.reorganize() 
#     db.close()
#  2) If the implementation doesn't support db reorganization, then you need to:
#       a) read the desired run object from the shelf file
#       b) complete more playouts on the run object
#       c) write the run object to a new db file
#       d) delete the old db file
# -

# #### Demonstrate the approach of gradually decreasing cpParm, which I currently think is a better approach than using gates
#
# Use the threshold bandit mode with a threshold cut of 10%

# +
myRun3 = ct.run(myDataset, nParallelParm=8)

startTime = time.time()
playoutsSoFar = 0

for i in range(32):
    currCP = (1/2**0.5) * (0.94)**i  # 1/sqrt(2) is the standard baseline value to use.
    ct.playouts(myRun3, 2000, pathType = [-2], cpParm = currCP, 
                banditMode='thresh', pctThreshCut=10, numPlays = 8)
    newPlayoutsSoFar = (i+1)*8
    if (i+1) % 4 == 0:
        elapsedTime = time.time() - startTime
        print('new playouts completed so far =', newPlayoutsSoFar, '   elapsed time =', elapsedTime)

elapsedTime = time.time() - startTime
print('Total elapsed time =', elapsedTime)

#  Now replay the 10 best performing paths using all the events
ct.replayBest(myRun3,10,0)

ct.textReports(myRun3,0)  # Don't include gate information in reports
ct.plotReports(myRun3)  # Don't include gate information in plots
