# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Running in Python
import time
import sys
sys.path.append("../../src")
import CTreeMI as ct
import MLP
import numpy as np


# +
# Make a toy dataset for testing

nSig = 10000
nBkd = 10000

sigV0 = [[np.random.normal(0, 1)] for i in range(nSig)]
sigV1 = [[np.random.normal(0.25, 1)] for i in range(nSig)]
sigV2 = [[np.random.normal(0.5, 1)] for i in range(nSig)]
sigV3 = [[np.random.normal(0.75, 1)] for i in range(nSig)]
sigV4 = [[np.random.normal(1.0, 1)] for i in range(nSig)]
sigV5 = [[np.random.normal(1.25, 1)] for i in range(nSig)]

bkdV0 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV1 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV2 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV3 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV4 = [[np.random.normal(0, 1)] for i in range(nBkd)]
bkdV5 = [[np.random.normal(0, 1)] for i in range(nBkd)]

sigVars = MLP.joint_space(sigV0, sigV1, sigV2, sigV3, sigV4, sigV5)
bkdVars = MLP.joint_space(bkdV0, bkdV1, bkdV2, bkdV3, bkdV4, bkdV5)

sigWts = [[1.0] for i in range(nSig)]
bkdWts = [[1.0] for i in range(nBkd)]
varNums = [0, 1, 2, 3, 4, 5]
varNames = ['var0', 'var1', 'var2', 'var3', 'var4', 'var5']
setName = 'test Gaussian'
myDataset = ct.dataset(setName, sigVars, bkdVars, sigWts, bkdWts, varNums, varNames)
# -
# Create a run object and pass it a dataset
myRun = ct.run(myDataset,20, 5, 8)

# +
# To launch playouts, you need to pass it a run that you've created, the type of playouts you want, and a 
# number of playout to execute.

# type [-1] is compressed tree.
# type [-2] is full tree
# Any other list for type is interpreted as fixed-path running.  So, for example, passing [1,2,5] will run
# all playouts through variables 1,2,5 only (counting from zero).
ct.playouts(myRun,1000,[-2],100)

# +
# Note that once a run object is created, you can continue to run playouts on it, and it will continue to accrue
# statistics from the playout.

# This is a second set of playouts executed on the same run object. 
ct.playouts(myRun,1000,[-2],100)

# -

# Report text information for the current run.
# The report results will be cumulative if you later do more playouts using the same run.
ct.textReports(myRun,0)

# Report plot-based information for the current run.
# The report results will be cumulative if you later do more playouts using the same run.
ct.plotReports(myRun,0)
