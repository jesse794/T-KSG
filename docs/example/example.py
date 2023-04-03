# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] tags=[]
# #### Example Notebook
#
# Create a dateset with five Gaussian variables and one spike variable, then document how the code works by demonstrating various features.
#
#     - Shows how to create a run object and perform playouts on it.
#     - Shows how to add additional playouts onto existing run object.
#     - Shows the full-tree mode and the fixed path mode.
#     - Shows how to read/write run objects to disk using shelf module so that one can add more playouts later.
#     - Shows how to automatically take a run's top n playouts and rerun the paths with more events.
#

# + tags=[]
# Make jupyter notebook display full width to prevent linefeeds in wide tables
# from IPython.core.display import display, HTML
from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

# + tags=[]
# Running in Python
import time
import random
import sys
sys.path.append("../../src")
import CTreeMI as ct
import CTreeUtils as ctu
import mi
import numpy as np
import pandas as pd
import csv
import numpy
import matplotlib.pyplot as plt


# + tags=[]
# Make a toy dataset for testing

# Number of sig and bkd events
nSig = 5000
nBkd = 5000

# Generate the individual variable distributions
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

# merge variable distributions into sig and bkd datasets
sigVars = mi.joint_space(sigV0, sigV1, sigV2, sigV3, sigV4, sigV5, sigV6)
bkdVars = mi.joint_space(bkdV0, bkdV1, bkdV2, bkdV3, bkdV4, bkdV5, bkdV6)

# add weights to events
# sigWts = [[1.0] for i in range(nSig)]
# bkdWts = [[1.0] for i in range(nBkd)]

sigWts = [[1.0]] * nSig
bkdWts = [[1.0]] * nBkd

# add names and id numbers to variables
varNums = [0, 1, 2, 3, 4, 5, 6]
varNames = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5', 'var6']

# add name to dataset
setName = 'test Gaussian'

# Call routine that creates the dataset object from the above information
myDataset = ct.dataset(setName, sigVars, bkdVars, sigWts, bkdWts, varNums, varNames)
# + tags=[]
mi.mi(sigV1, bkdV2,k=1,base=2) # just a sanity-check on the continuous MI function.

# + tags=[]
# Create a run object and pass it a dataset.
# Alternatively, you can read back a run object that you stored from previous running.
# Uncomment one of the following two lines.
myRun = ct.run(myDataset, gateEvalParm=30, nParallelParm=12)
# myRun = ctu.shelfRun(filename='example', mode='read', key='example')

# + tags=[]
# To launch playouts, you need to pass it a run that you've created, the type of playouts you want, and a
# number of playouts to execute.

# pathType [-1] is compressed tree.

# pathType [-2] is full tree, though how it behaves precisely depends on other arguments used when creating the run and 
# in launching playouts.  See full documentation for details.

# If you pass a list for the type argument, it is interpreted as fixed-path running.  So, for example, passing [1,2,5] will run
# all playouts through variables 1,2,5 only (counting from zero).
startTime = time.time()
print("{:>5s} {:>7s} {:>10s} {:>10s} {:>10s}  {:>10s}".
      format('cycle', 'time', 'pathsopen', 'pathsTstd', 'pathsUntstd', 'highScore'))
for i in range(8):
    cycleStartTime = time.time()
    ct.playouts(myRun, eventsPerPlayout=1000, pathType=[-2], projectedGateDecision=True, pctDefn='absolute',
                cpParm=0.2, banditMode='thresh', banditThreshCut=5.0,
                gateMode='threshPctSignif', gateThreshCut=10.0, gateIncCut=-5.0, gateExcCut=-5.0,
                sampleRandomly=True, numPlays=12)
    print("{:>5d} {:>7.2f} {:>.4e} {:>.4e}  {:>.4e}    {:>8.3f}".
          format(i, time.time() - cycleStartTime, myRun.pathsCount()['nOpen'], 
                 myRun.pathsCount()['nTestedOpen'], myRun.pathsCount()['nUntestedOpen'], myRun.highestScore))
print('----------------------------------')
print('Total elapsed time = {:>7.2f}'.format(time.time() - startTime))

# + tags=[]
# Note that once a run object is created, you can continue to run playouts on it, and it will continue to accrue
# statistics from the playouts.
# Here, we run more playouts on an existing run object.
startTime = time.time()
print("{:>5s} {:>7s} {:>10s} {:>10s} {:>10s}  {:>10s}".
      format('cycle', 'time', 'pathsopen', 'pathsTstd', 'pathsUntstd', 'highScore'))
for i in range(8):
    cycleStartTime = time.time()
    ct.playouts(myRun, eventsPerPlayout=1000, pathType=[-2], projectedGateDecision=True, pctDefn='absolute',
                cpParm=0.2, banditMode='thresh', banditThreshCut=5.0,
                gateMode='threshPctSignif', gateThreshCut=10.0, gateIncCut=-5.0, gateExcCut=-5.0,
                sampleRandomly=True, numPlays=12)
    print("{:>5d} {:>7.2f} {:>.4e} {:>.4e}  {:>.4e}    {:>8.3f}".
          format(i, time.time() - cycleStartTime, myRun.pathsCount()['nOpen'], 
                 myRun.pathsCount()['nTestedOpen'], myRun.pathsCount()['nUntestedOpen'], myRun.highestScore))
print('----------------------------------')
print('Total elapsed time = {:>7.2f}'.format(time.time() - startTime))

# + tags=[]
# Here, we force completion of any paths that are still open but that haven't been tested (note the change in "banditMode" argument)
startTime = time.time()
print("{:>5s} {:>7s} {:>10s} {:>10s} {:>10s}  {:>10s}".
      format('cycle', 'time', 'pathsopen', 'pathsTstd', 'pathsUntstd', 'highScore'))
for i in range(8):
    cycleStartTime = time.time()
    ct.playouts(myRun, eventsPerPlayout=1000, pathType=[-2], projectedGateDecision=True, pctDefn='absolute',
                cpParm=0.2, banditMode='untested', banditThreshCut=5.0,
                gateMode='threshPctSignif', gateThreshCut=10.0, gateIncCut=-5.0, gateExcCut=-5.0,
                sampleRandomly=True, numPlays=12)
    print("{:>5d} {:>7.2f} {:>.4e} {:>.4e}  {:>.4e}    {:>8.3f}".
          format(i, time.time() - cycleStartTime, myRun.pathsCount()['nOpen'], 
                 myRun.pathsCount()['nTestedOpen'], myRun.pathsCount()['nUntestedOpen'], myRun.highestScore))
print('----------------------------------')
print('Total elapsed time = {:>7.2f}'.format(time.time() - startTime))

# + tags=[]
# Note that once a run object is created, you can continue to run playouts on it, and it will continue to accrue
# statistics from the playouts.  
#  Here, we run some fixed-path playouts on an existing run object
ct.playouts(myRun,0,[0], numPlays=1)
ct.playouts(myRun,0,[1], numPlays=1)
ct.playouts(myRun,0,[2], numPlays=1)
ct.playouts(myRun,0,[3], numPlays=1)
ct.playouts(myRun,0,[4], numPlays=1)
ct.playouts(myRun,0,[5], numPlays=1)
ct.playouts(myRun,0,[6], numPlays=1)

# + tags=[]
aaa,bbb=ct.errProbBounds(.435)

# + tags=[]
aaa

# + tags=[] jupyter={"outputs_hidden": true}
# Report text information for the current run.
# The report results are cumulative.  So if you do more playouts later using the same run and then call 
# for a report, the new report will include results from all playouts that have ever been done on that run object.
ct.textReports(myRun,playoutsToReport=0,includeGateInfo=True)

# + tags=[]
# Write the run object (which has all completed playouts) to a file so you can load it back in later
# and continue playouts from where you left off.
ctu.shelfRun(filename='example', mode='write', key='example', myRunIn=myRun)

# + tags=[]
# uncomment if you want to delete the persisted run object in the shelf file.
# ctu.shelfRun(filename='example', mode='delete', key='example')
# -

# #### Create a new run object so you can run other playouts without affecting the previous set of playouts

# + tags=[]
# Create a run object and pass it a dataset
myRun2 = ct.run(myDataset, gateEvalParm=30, nParallelParm=12)

# + tags=[]
# Run a few playouts on this new run object
startTime = time.time()
print("{:>5s} {:>7s} {:>10s} {:>10s} {:>10s}  {:>10s}".
      format('cycle', 'time', 'pathsopen', 'pathsTstd', 'pathsUntstd', 'highScore'))
for i in range(16):
    cycleStartTime = time.time()
    ct.playouts(myRun2, eventsPerPlayout=1000, pathType=[-2], projectedGateDecision=True, pctDefn='absolute',
                cpParm=0.2, banditMode='thresh', banditThreshCut=5.0,
                gateMode='threshPctSignif', gateThreshCut=10.0, gateIncCut=-5.0, gateExcCut=-5.0,
                sampleRandomly=True, numPlays=12)
    print("{:>5d} {:>7.2f} {:>.4e} {:>.4e}  {:>.4e}    {:>8.3f}".
          format(i, time.time() - cycleStartTime, myRun.pathsCount()['nOpen'], 
                 myRun2.pathsCount()['nTestedOpen'], myRun.pathsCount()['nUntestedOpen'], myRun.highestScore))
print('----------------------------------')
print('Total elapsed time = {:>7.2f}'.format(time.time() - startTime))
# -

# #### Read back the old run object from file, add more playouts to it, then write it back to file.

# + tags=[]
# Read the first run object back off the shelf.
myRunReadBack = ctu.shelfRun(filename='example', mode='read', key='example')

# + tags=[]
# Run a few more playouts on the read-back run object, print updated information, then rewrite the object to file.
# Writing and then reading-back run objects allows true continuation of runs.
ct.playouts(myRunReadBack, eventsPerPlayout=1000, pathType=[-2], projectedGateDecision=True, pctDefn='absolute',
                cpParm=0.2, banditMode='thresh', banditThreshCut=5.0,
                gateMode='threshPctSignif', gateThreshCut=10.0, gateIncCut=-5.0, gateExcCut=-5.0,
                sampleRandomly=True, numPlays=12)
ct.textReports(myRunReadBack,playoutsToReport=0, includeGateInfo=True)
ctu.shelfRun(filename='example', mode='write', key='example', myRunIn=myRunReadBack)

# + tags=[]
# Note that the shelf file will continue to grow each time you write back to it.  
# This can be a problem in cases where the output is large.  And because the run object 
# has a copy of the dataset, the output will be large in all cases where the dataset
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

# + tags=[]
#  Now replay the 10 best performing paths using all the events
ct.replayBest(myRunReadBack,10,0)
ct.textReports(myRunReadBack,playoutsToReport=0, includeGateInfo=True)

# + tags=[]
# Print playout reports but restrict them only to playouts with specific numbers of variables used
# Note that the best playout w/, n variables shown here is not necessarily the best one could achieve with
# n perfectly-chosen variables.  The reason is that these playouts were initially run w/o any limit on the 
# number of variables.  So, the paths with only a small number of variables (which is what you're looking at
# here) were competing against the top overall paths rather than against the top paths with only n variables.  
# This isn't the same as letting the MC tree run freely but telling it to only ever use n variables.  
# The result below is correct for what it is, but it's important to understand the limitation on the result.
for nVar in range(1,8):
    print("Report restricted to playouts with number of variables =", nVar)
    ct.playoutsReport(myRunReadBack, numToReport=0, sortByScore=True, nVarRestrict=nVar)

# + tags=[]
# Report plot-based information for the current run.
# The report results will be cumulative if you later do more playouts using the same run.
ct.plotReports(myRun,'myPlots_1.pdf',includeGateInfo=True)

# + tags=[]
# Report plot-based information for the current run.
# The report results will be cumulative if you later do more playouts using the same run.
ct.plotReports(myRun2,'myPlots_2.pdf',includeGateInfo=True)
