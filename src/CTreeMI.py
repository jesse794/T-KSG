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

# +

"""
Usage:

Code written by Jesse Ernst and Nick Carrara
Based on https://arxiv.org/abs/1708.09449 by Nick Carrara and Jesse Ernst and
on a modified Monte Carlo Tree Search method by Jesse Ernst and Nick Carrara.

1. Create a dataset myDataset = dataset(setName, sigVars, bkdVars, sigWts, bkdWts, varNums, varNames)
sigVars, bkdVars, sigWts, bkdWts are lists of variable values <br/>
varNums is just a list of integers for the variables <br/>
varNames is a list of names for the variables <br/>
Optional arguments allow for specifying some variables as discrete and also for normalizing inputs <br/>

2. Create a run and pass it a dataset: myRun = run(myDataset).
A run holds all information about all nodes and all playouts for all playouts that you have run on it (see below). <br/>
myDataset is the dataset to use for the run. Additional arguments define how the path will be optimized (see below)<br/>

3. Execute one or more playouts for your run: playouts(myRun,100,[-2],50)<br/>
myRun: is the run to execute playouts on<br/>
100: is the number of events to use per playout<br/>
[-2]: describes the type of playouts to execute (see below)<br/>
50: is the number of playouts to execute<br/>

    playout types:

    - [-1] is compressed tree.  I.e. nodes are included/excluded
    based on their performance independent of what other
    variables were inc/exc.

    - [-2] is full tree, but see parameter settings for details on this.

    - [i,j,k] Any other list for this argument is interpreted as fixed-path running.
    All playouts will use only variable numbers i,j,k (counting from zero)

    - Numerous optional arguments define how the paths will be optimized (see below)

4. Execute more playouts on the run as needed.
Once a run object is created and playouts have been run (steps above), you can continue to run playouts on it,
and it will continue to accrue statistics from these playouts and add the results to the run's stored information.
So, you could keep calling playouts(myRun,100,[-1],50) if after looking at the reports (see below), you decide that
you want to continue with more playouts starting from the current state of the run.

5. Print text-based report textReports(myRun,n,i)
myRun specifies the run you want to report on, n indicates how many playouts should be included (0=all) in the
all-playouts section of the report, and  i indicates whether or not gate information should be included (see below)

6. Make plot-based reports plotReports(myRun,0,makePDF=False)
Makes a bunch of plots to the screen if you're in something like jupyter or have XQuartz running.  If you set
makePDF=True, then it will also send the plots to a file.

Additional parameters you can include when creating a run:

- getEvalParm: Both sides of a node must be visited this many times before either gate can be closed.  This is a
subtle and important parameter for full-tree mode.  In full-tree mode, closing a gate on one of the variables means
that if any node anywhere in the tree has gateEvalParm visits to both its include and exclude sides,
then the variable will be evaluated for possible closing one of the gates.  However, when that gate is closed,
that variable is now always included or excluded no matter where it appears in the tree in future path searches. The
nice feature of this approach is that initially, only variables near the top of the tree  (i.e., small var numbers),
which receive more visits can be shut off, since they are the nodes that will reach this threshold first.  However,
as low-numbered variables have one of their gates closed, gates further down the tree will get more visits.  So the
gate actions will propagate down the tree as playouts increase.  If one wants a proper full-tree mode with no gates
ever being closed, just set this parameter to be very large. There is currently no way to turn off gates in
individual nodes in a full tree.

- nodeEvalParm: Only used in compress-tree mode Both sides of a node must be visited this many times before policy
can be implemented to choose a side. Below this value, we flip a coin to choose a side.

- nParallel: Number of playouts that will be batched in parallel processing.  Speeds the code significantly,
but do not set this to be greater than 8 without reading the longer comment in the code itself, even if you are on a
machine with more than eight cores.

"""

# Jesse Ernst, Nick Carrara
import platform
import sys
import numpy as np
import random
import pandas as pd
import scipy
import scipy.stats
from scipy import special
from scipy.stats import norm
from scipy import interpolate
from copy import deepcopy
import operator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing as mp
import math
import mi

# print("python : " + platform.python_version())
# print("numpy : " + np.__version__)
# print("pandas : " + pd.__version__)
# print("scipy : " + scipy.__version__)


# -

# List of global variables.  These will be changed only at the start of the playouts init method
# They will then be restored at the end of the set of playouts
globCPParm = 0.7071  # The cpParm value used in the bandit formula to balance exploration against exploitation
globBanditMode = "thresh"  # Defines what metric the bandit formula uses to score include against exclude
globBanditThreshCut = 5.0  # In bandit threshold mode, score must be high enough to be counted as win
globGateMode = 'threshPctAbs'  # Sets the mode used for closing gates
globGateThreshCut = 5.0  # For gate-closings in thresh mode, score must be high enough to be included in count
globGateIncCut = 2.0  # In thresh mode for gates, defines cut for closing inc gate (i.e. always exc var)
globGateExcCut = 2.0  # In thresh mode for gates, defines cut for closing exc gate (i.e. always inc var)
globksgK = 1  # This sets the number of neighbors that the ksg method will include
globNVarsMax = 0  # This sets the max number of variables allowed in any path (zero means unlimited)
globProjectedGateDecision = False  # Should summary node information determine gate closings for all nodes in layer
globPctDefn = 'percentile'  # Should percent calls to threshold be absolute relative to top score or percentiles.


def initializeGlobalVars():
    """Restore all global parameters back to their default values"""
    # This is a little kludgey, as this block needs to remain the same as the block just above it where the global
    # variables are created.
    global globCPParm, globBanditMode, globBanditThreshCut, \
        globGateMode, globGateThreshCut, globGateIncCut, globGateExcCut,\
        globksgK, globNVarsMax, globProjectedGateDecision, globPctDefn
    globCPParm = 0.7071  # The cpParm value used in the bandit formula to balance exploration against exploitation
    globBanditMode = "thresh"  # Defines what metric the bandit formula uses to score include against exclude
    globBanditThreshCut = 5.0  # In bandit threshold mode, score must be high enough to be counted as win
    globGateMode = 'threshPctAbs'  # Sets the mode used for closing gates
    globGateThreshCut = 5.0  # For gate-closings in thresh mode, score must be high enough to be included in count
    globGateIncCut = 2.0  # In thresh mode for gates, defines cut for closing inc gate (i.e. always exc var)
    globGateExcCut = 2.0  # In thresh mode for gates, defines cut for closing exc gate (i.e. always inc var)
    globksgK = 1  # This sets the number of neighbors that the ksg method will include
    globNVarsMax = 0  # This sets the max number of variables allowed in any path (zero means unlimited)
    globProjectedGateDecision = False  # Should summary node information determine gate closings for all nodes in layer
    globPctDefn = 'percentile'  # Should percent calls to threshold be absolute relative to top score or percentiles.


class dataset:
    """Set of events"""

    def __init__(self, setName, sigVars, bkdVars, sigWts, bkdWts, varNums, varNames, discVars=None,
                 normed=False, normExclusions=None):
        self.setName = setName
        self.sigVars = sigVars
        self.bkdVars = bkdVars
        self.sigWts = [_[0] for _ in sigWts]
        self.bkdWts = [_[0] for _ in bkdWts]
        self.sigWtMax = max(sigWts)[0]
        self.sigWtMin = min(sigWts)[0]
        self.bkdWtMax = max(bkdWts)[0]
        self.bkdWtMin = min(bkdWts)[0]
        self.varNums = varNums
        self.varNames = varNames
        if discVars is None:
            discVars = []
        self.discVars = discVars
        # This is a list of specific values that shouldn't be included in normalization (-999 is obvious example)
        if normExclusions is None:
            normExclusions = []
        if normed is True:
            self.normalizeData(normExclusions, stretchToGaussian=True)

        # Sample efficiency is the fraction of events you'd expect to pass weighted sampling.
        # One extreme is if all weights in the file are equal, then it should be 1.
        # The other extreme is if a few events have large weights and all the others have small weights.  Then,
        # the sampleEff value would tend toward zero.
        self.sigSampleEff = np.mean([x / self.sigWtMax for x in self.sigWts])
        self.bkdSampleEff = np.mean([x / self.bkdWtMax for x in self.bkdWts])

        self.nTimesDrawnSig = [0] * len(sigVars)  # initialize counters that tell num times each event has been drawn
        self.nTimesDrawnBkd = [0] * len(bkdVars)  # initialize counters that tell num times each event has been drawn

        assert (len(varNames) == len(varNums))
        assert (len(sigVars) == len(sigWts))
        assert (len(bkdVars) == len(bkdWts))

        # Shuffle the order of the events
        self.shuffleData()

    def reorderVariables(self, key):
        """Change the order of the variables in signal and background arrays according to
        the values in a key array."""

        # First rearrange the sig and bkd variables.  Then rearrange the variable names so they remain consistent
        # with the reordered list.  Finally, make sure discrete-variable labels are relative to the numbers in the
        # new variable order.

        nKeys = len(key)
        nVariables = len(self.sigVars[0])
        assert(nKeys == nVariables), "{0:30s} {1:10d} {2:10d}".\
            format("key length not equal to nvars:", nKeys, nVariables)
        # Make list that will be used for reordering
        keyRankTemp = scipy.stats.rankdata(key, method='ordinal')
        keyRank = [nKeys - _ for _ in keyRankTemp]

        # Rearrange the variables names
        newVarNames = [-9999]*nVariables
        for oldSpot, newSpot in enumerate(keyRank):
            newVarNames[newSpot] = self.varNames[oldSpot]
        self.varNames = deepcopy(newVarNames)

        # Rearrange the sig variables
        nRows = len(self.sigVars)
        tempArray = [[0 for i in range(nVariables)] for j in range(nRows)]
        for row in range(nRows):  # Loop over rows
            for oldSpot, newSpot in enumerate(keyRank):
                tempArray[row][newSpot] = self.sigVars[row][oldSpot]
        self.sigVars = deepcopy(tempArray)  # Copy the adjusted array back on top of the original

        # Rearrange the bkd variables
        nRows = len(self.bkdVars)
        tempArray = [[0 for i in range(nVariables)] for j in range(nRows)]
        for row in range(nRows):  # Loop over rows
            for oldSpot, newSpot in enumerate(keyRank):
                tempArray[row][newSpot] = self.bkdVars[row][oldSpot]
        self.bkdVars = deepcopy(tempArray)  # Copy the adjusted array back on top of the original

        # Now mark the new spots where the discrete variables are
        tempArray = []
        for iDisc in self.discVars:  # Loop over discrete variable numbers
            newSpot = keyRank[iDisc]  # Location for this discrete variables in the rearranged list
            tempArray.append(newSpot)
        self.discVars = deepcopy(tempArray)
        return

    def shuffleData(self):
        """For both sig and bkd, shuffles the variables and weights while keeping them in sync with one another. """
        # Now shuffle sigvars, sigwts, bkdvars, and bkdwts and be sure weights and var values stay in sync
        savedState = random.getstate()  # get state of random generator
        random.shuffle(self.sigVars)  # Shuffle vars
        random.setstate(savedState)  # restore random generator to previous state so shuffle will be identical
        random.shuffle(self.sigWts)  # shuffle weights

        savedState = random.getstate()  # get state of random generator
        random.shuffle(self.bkdVars)  # Shuffle vars
        random.setstate(savedState)  # restore random generator to previous state so shuffle will be identical
        random.shuffle(self.bkdWts)  # shuffle weights

    def resetDrawnCounters(self):
        """Reset the counters that keep track of number of times each signal and background event has been drawn"""
        self.nTimesDrawnSig = [0] * len(self.sigVars)
        self.nTimesDrawnBkd = [0] * len(self.bkdVars)

    def normalizeData(self, normExclusions, stretchToGaussian=False):
        """Normalize the dataset variables excluding values in exclusions list.  Resulting lists will have a mean
        of zero and an rms of one."""
        nSigEvts = len(self.sigVars)
        nBkdEvts = len(self.bkdVars)
        for iVar in self.varNums:  # Make adjustments to each variable separately.

            if iVar in self.discVars: continue  # Don't do anything to discrete variables.

            # Make True/False lists showing which events will have their values adjusted.
            sigAdjust = [self.sigVars[_][iVar] not in normExclusions for _ in range(nSigEvts)]
            bkdAdjust = [self.bkdVars[_][iVar] not in normExclusions for _ in range(nBkdEvts)]

            # Make lists that point to where each element of the to-be-modified lists sits in the original lists
            sigAdjustPointer = [_ for _ in range(nSigEvts) if sigAdjust[_]]
            bkdAdjustPointer = [_ for _ in range(nBkdEvts) if bkdAdjust[_]]
            nSigEvtsAdj = len(sigAdjustPointer)  # This is smaller than nSigEvts because not all evts will be adj.
            nBkdEvtsAdj = len(bkdAdjustPointer)  # This is smaller than nBkdEvts because not all evts will be adj.

            # Find mean and sigma of merged list of signal and background.  Only include values that are not
            # in the normExclusions list.  This prevents things like -999's from being included in the calculation
            # of a variable's avg value.
            tempSig = [self.sigVars[_][iVar] for _ in range(nSigEvts) if sigAdjust[_]]
            tempBkd = [self.bkdVars[_][iVar] for _ in range(nBkdEvts) if bkdAdjust[_]]
            tempMerge = tempSig + tempBkd
            mergedMean = np.mean(tempMerge)
            mergedStd = np.std(tempMerge)

            # Adjust the values of events that are not in exclude list.
            for i in range(nSigEvts):
                origValue = self.sigVars[i][iVar]
                scaledValue = float((origValue - mergedMean) / mergedStd) if sigAdjust[i] else origValue
                self.sigVars[i][iVar] = scaledValue
            for i in range(nBkdEvts):
                origValue = self.bkdVars[i][iVar]
                scaledValue = float((origValue - mergedMean) / mergedStd) if bkdAdjust[i] else origValue
                self.bkdVars[i][iVar] = scaledValue

            if stretchToGaussian:  # Try to convert the shape of the distribution to a Gaussian
                # Remake the merged list since the values were updated in the initial normalization step.
                tempSig = [self.sigVars[_][iVar] for _ in range(nSigEvts) if sigAdjust[_]]
                tempBkd = [self.bkdVars[_][iVar] for _ in range(nBkdEvts) if bkdAdjust[_]]
                tempMerge = tempSig + tempBkd

                # Rank the values and then shift them into range of -1 to +1.
                mergedRanks = scipy.stats.rankdata(tempMerge)  # Make a list ranking the values
                mergedRanks = [float(mergedRanks[_]/len(mergedRanks)) for _ in range(len(mergedRanks))]
                mergedRanks = [2.0 * mergedRanks[_] - 1.0 for _ in range(len(mergedRanks))]

                # split the merged ranks back into separate signal and background ranking lists
                sigRanks = mergedRanks[:nSigEvtsAdj]  # Ranks of the sig events that are ranked
                bkdRanks = mergedRanks[-nBkdEvtsAdj:]  # Ranks of the bkd events that are ranked

                # protect against passing +1 or -1 into erfinv.
                sigRanks = np.asarray([min(_, 1-1e-6) for _ in sigRanks])
                bkdRanks = np.asarray([min(_, 1-1e-6) for _ in bkdRanks])

                # Get the stretched values
                sigAdjustedValues = [special.erfinv(_) for _ in sigRanks]
                bkdAdjustedValues = [special.erfinv(_) for _ in bkdRanks]

                # Put the stretched values back into the sig and bkd lists at the correct spots.
                for i in range(nSigEvtsAdj):
                    self.sigVars[sigAdjustPointer[i]][iVar] = sigAdjustedValues[i]
                for i in range(nBkdEvtsAdj):
                    self.bkdVars[bkdAdjustPointer[i]][iVar] = bkdAdjustedValues[i]


class run:
    """The run object holds the data and also the results from multiple playouts and paths through the data"""

    def __init__(self, locDataset, gateEvalParm=999999, nodeEvalParm=5, nParallelParm=8):
        self.dataset = locDataset
        self.varNums = locDataset.varNums
        self.varNames = locDataset.varNames

        # Both sides of a node must be visited this many times before either gate can be closed.
        # If it's set very high, it effectively means that gates will never be closed.
        minAllowedGateEvalParm = 10
        assert gateEvalParm >= minAllowedGateEvalParm, "{0:55s} {1:10d} {2:10d}".\
            format("Given value for gateEvalParm below minimum allowed", gateEvalParm, minAllowedGateEvalParm)
        # To allow minVisitsForGateEval to be changed between sets of playouts, make it a list.  Then when you want to
        # change it, call a routine that will append the new value onto the end of the list.  Then when using the list
        # to see if you have enough playouts through a node to consider closing a gate, just pull off the last entry in
        # the list.  This way, you can modify the value, but still keep track of all values used.
        self.minVisitsForGateEval = [gateEvalParm]

        # Both sides of a node must be visited this many times before policy can be implemented to choose a side.  Below
        # this value, we flip a coin to choose a side.  This is currently only used in compressed-tree mode.  In full
        # tree mode, this parameter is meaningless, as comparisons are made with the bandit formula as soon as each
        # side has at least one visit.
        self.minVisitsForNodeEval = nodeEvalParm

        # Number of playouts to run in parallel using python's multiprocessing calls.  One can't just launch all the
        # playouts into one big multiprocessing pool because each playout needs to make its own dataset that holds just
        # the variables it is using and is a random selection of all the events (since each playout runs only on a
        # subsample of events).  Typically, I'd expect that this should be set to something around the number of cores,
        # though it might have to be lower if you are running a large number of events in a playout, and don't have
        # enough memory to hold as many copies of the data as you have cores in the machine.
        # CAUTION: You also should not set this too high, because information on nodes paths are not updated until all
        # parallel processes are completed.  I.e., playouts are run in batches of nParallal.  Thus if you set it too
        # high, you'll be running many times w/o new information about which path to try next.  In particular, depending
        # on the tree type and the variable policy, you could wind up running the same exact path nParallel times.
        # That's not necessarily bad, as it'll improve your statistics w/o running-time penalty, but it's not going
        # to give you any more paths (though I guess you could set nEvents per playout smaller if you knew each was
        # going to be run nParallel times).
        self.nParallel = nParallelParm
        self.nodesColl = nodesCollection(self)
        self.pathStatsColl = pathStatsColl()
        self.playoutList = []
        self.highestScore = 0
        self.nodescollhighestScore = 0
        self.nFallbackPath = 0
        self.reachableStubs = []  # These mark paths that only go partway down the tree.  Caused by untested paths.

    def modifyMinVisitsForGateEval(self, minVisits):
        """Append a new value onto the list of values used for minimum number of visits.  The last value will always
        be the one that is used in gate closings.  This allows us to modify the value between sets of playouts, but
        also keep track of all the values that have been used during running."""
        self.minVisitsForGateEval.append(minVisits)
        print(self.minVisitsForGateEval)
        return

    def updateRunStats(self):
        """Update information for the run that changes as playouts are added"""
        # The highest score is the highest average value for any path.  This way, if a path is visited n times,
        # the highest overall score is not the highest of the n visits, but the average of them.  The idea here
        # is that one doesn't want to bias toward scoring a path higher just because it has been visited many times
        # and so is more likely to have one of its playouts be a high fluctuation.
        self.highestScore = max([_.mean for _ in self.pathStatsColl.pathStatsDict.values()])

    def getThresholdScore(self, cut, mode):
        """Returns a cutoff score based on the given cut (given as a percent) and the mode.  If the mode=absolute,
        then it returns the score that is cut percent below the largest mean score.  If mode=percentile, then it
        returns the score that is in the top cut percent of scores."""
        if mode == 'absolute':  # if, e.g., cut=2, then the result will be 98% of the highest score.
            topScoreOverall = self.highestScore  # Highest scoring path (averaged over all playouts with that path).
            thresholdScore = topScoreOverall * (1.0 - cut/100.0)
        elif mode == 'percentile':  # if, e.g., cut=2, then the result should be a score that is above 98% of scores.
            tempScoreList = [_.mean for _ in self.pathStatsColl.pathStatsDict.values()]
            spotToPick = math.floor(len(tempScoreList)*(cut/100.0))
            thresholdScore = sorted(tempScoreList, reverse=True)[spotToPick:][0]
        else:
            print("ERROR: getThresholdScore called with unknown mode")
            sys.exit()
        return thresholdScore

    def pathsCount(self):
        """Returns a variety of counts to understand the number of paths that are available, tested, closed, etc.
        There are nine different things to count:  Tested, Untested, and All, for each of Open, Closed, and Total."""
        locDataset = self.dataset
        locNodesColl = self.nodesColl.all
        smryNodes = self.nodesColl.smry
        numberOfVariables = len(locDataset.varNums)
        lastVarNum = numberOfVariables - 1  # subtract one because variable numbers start from zero
        lastLayerNodes = locNodesColl[lastVarNum].values()

        # total number of possible paths.
        allTotal = 2**numberOfVariables  # note that the path of all excludes (no variables) won't ever be allowed.

        # total (open and closed) num paths tested.  Each unique path gets dic entry, so length is num paths tested
        testedTotal = len(self.pathStatsColl.pathStatsDict)

        # Find untestedOpen by using the reachableStubsList
        # First make list of which variables have both gate sides open (T/F for each variable).
        bothSidesOpen = [smryNodes[_].incPathOpen and smryNodes[_].excPathOpen for _ in range(0, numberOfVariables)]
        # Initialize number of untested open paths to zero because you'll add counts as you go through the routine.
        # However, if there aren't any summary nodes yet, then we must just be starting the run, and you should
        # initialize it to be all paths, since everything is open and untested when starting playouts on new run object.
        untestedOpen = 0 if len(self.playoutList) > 0 else allTotal
        for stubNode in self.reachableStubs:  # Loop over all stub nodes
            # A node in the last layer with an ID of zero is all the way down the tree and along the right
            # edge.  In that case, it's exclude path is never allowed (since it would correspond to a path w/o any
            # variables), and its include path has definitely been followed, otherwise the node wouldn't yet have been
            # created.
            if stubNode.nodeIsInLastLayer() and stubNode.nodeID != 0:
                untestedOpen += 1
            else:
                stubVarnum = stubNode.varNum
                nVarsBelow = numberOfVariables - stubVarnum - 1  # number of variables below the stub
                nVarsBlockedBelow = bothSidesOpen[-nVarsBelow:].count(False)  # count num of lower levels w/ closed gate
                reachablePathsBelowHere = 2**(nVarsBelow - nVarsBlockedBelow)
                untestedOpen += reachablePathsBelowHere

        # Find testedOpen by using last-layer nodes
        # Count reachable last-layer nodes, and then increment testedOpen path count if one or both sides of its
        # output sides are open and have been visited.
        testedOpen = 0  # initialize number of tested open paths
        for currNode in lastLayerNodes:
            if currNode.nodeReachableFromRoot():
                if currNode.incPathOpen and currNode.nVisitsIncPath > 0: testedOpen += 1
                if currNode.excPathOpen and currNode.nVisitsExcPath > 0: testedOpen += 1

        # The rest of the counts are just simple arithmetic from the above
        testedClosed = testedTotal - testedOpen
        untestedTotal = allTotal - testedTotal
        untestedClosed = untestedTotal - untestedOpen
        allOpen = untestedOpen + testedOpen
        allClosed = untestedClosed + testedClosed

        # Gather all results into a dictionary
        resultD = {'nTotal': allTotal, 'nTested': testedTotal, 'nUntested': untestedTotal,
                   'nOpen': allOpen, 'nClosed': allClosed,
                   'nTestedOpen': testedOpen, 'nTestedClosed': testedClosed,
                   'nUntestedOpen': untestedOpen, 'nUntestedClosed': untestedClosed}
        return resultD


class nodesCollection:
    """A collection of nodes"""

    def __init__(self, locRun):
        self.run = locRun

        # This will be a nested dictionary for all nodes.  The outer key is var number.
        # The inner key will be locationID for the node.
        self.all = {}
        for i in range(len(locRun.dataset.varNums)):
            varNum = locRun.dataset.varNums[i]
            self.all[varNum] = {}  # Initially an empty dictionary for each var number.  Will be filled over playouts

        # This will be a dictionary of summary nodes (one per variable) with var number as key
        self.smry = {}
        for i in range(len(locRun.dataset.varNums)):
            varNum = locRun.dataset.varNums[i]
            varName = locRun.dataset.varNames[i]
            self.smry[varNum] = node(varNum, varName, locRun)  # Summary nodes for each var to hold stats.

    def getNode(self, varNum, nodeID, createNode=False):
        """Extracts the node with the given variable number and nodeID from the nodesCollection.
        If createNode==True, then it will create the node if it doesn't already exist in the dictionary"""
        # From the nested dictionary, get the dictionary for this particular variable
        varDict = self.all[varNum]
        if nodeID in varDict:
            myNode = varDict[nodeID]  # Get the node from the dictionary.
        else:  # requested node not yet in dict.  So either create it or return None.
            if createNode:
                newNode = node(varNum, self.run.dataset.varNames[varNum], self.run, nodeID)
                varDict[nodeID] = newNode   # Put new node into dictionary
                myNode = varDict[nodeID]  # Get the node from the dictionary.
            else:
                myNode = None
        return myNode

    def resetNodesKeepGates(self):
        """This routine will loop over all tree and summary nodes and reset all statistics except gates statuses"""
        # Make a long unstructured list of all nodes:  tree and summary all thrown into one long list.
        allNodesList = []
        treeNodes = self.all
        for iVar in self.run.varNums:  # For each variable layer, add all tree nodes to the long list of nodes.
            layerNodes = list(treeNodes[iVar].values())  # List of nodes for the current variable
            allNodesList.extend(layerNodes)
        smryNodes = list(self.smry.values())  # Now get the summary nodes.
        allNodesList.extend(smryNodes)  # Add the summary nodes to the long list.

        # Now loop over the list of all nodes and reset statistics, leaving gate statuses intact.
        for currNode in allNodesList:
            currNode.nVisitsIncPath = 0
            currNode.nVisitsExcPath = 0
            currNode.nVisitsIncPathUnique = 0
            currNode.nVisitsExcPathUnique = 0
            currNode.maxScoreIncPath = 0
            currNode.maxScoreExcPath = 0
            currNode.sumScoreIncPath = 0
            currNode.sumScoreExcPath = 0
            currNode.sumScoreSqIncPath = 0
            currNode.sumScoreSqExcPath = 0
            currNode.allScoresIncPath = {}
            currNode.allScoresExcPath = {}
            currNode.meanScoreInc = 0
            currNode.meanScoreIncErr = 0
            currNode.rmsScoreInc = 0
            currNode.meanScoreExc = 0
            currNode.meanScoreExcErr = 0
            currNode.rmsScoreExc = 0
        return


class node:
    """One node corresponds to one variable.  Each node has a variable
    that can either be included or excluded in a path.  Each
    node also has a pair of gates that control whether or not the node's
    variable should always be included or always be excluded from
    paths."""

    def __init__(self, varNum, varName, locRun, nodeID=-999):
        self.varNum = varNum
        self.varName = varName
        self.run = locRun
        self.nodeID = nodeID
        self.nVisitsIncPath = 0
        self.nVisitsExcPath = 0
        self.nVisitsIncPathUnique = 0  # Unique playout paths that pass through node and include its variable
        self.nVisitsExcPathUnique = 0  # Unique playout paths that pass through node and exclude its variable
        self.maxScoreIncPath = 0
        self.maxScoreExcPath = 0
        self.sumScoreIncPath = 0
        self.sumScoreExcPath = 0
        self.sumScoreSqIncPath = 0
        self.sumScoreSqExcPath = 0
        self.allScoresIncPath = {}  # scores dic with pathid as key for playouts through node that include its variable
        self.allScoresExcPath = {}  # scores dic with pathid as key for playouts through node that exclude its variable
        # Mean and rms of scores through node that include or exclude the node's variable.  In computing the
        # statistics, multiple playouts with the same path are averaged together first, and then the different
        # paths are averaged to make the statistics in the next six lines.  This ensures that paths that have multiple
        # playouts are still only counted once when taking the averages etc.
        self.meanScoreInc = 0  # Averages over multiple playouts with same path before averaging over paths
        self.meanScoreIncErr = 0    # Averages over multiple playouts with same path before averaging over paths
        self.rmsScoreInc = 0    # Averages over multiple playouts with same path before averaging over paths
        self.meanScoreExc = 0  # Averages over multiple playouts with same path before averaging over paths
        self.meanScoreExcErr = 0    # Averages over multiple playouts with same path before averaging over paths
        self.rmsScoreExc = 0    # Averages over multiple playouts with same path before averaging over paths
        self.incPathOpen = True
        self.excPathOpen = True
        self.newlyCreated = True  # Used by gate-setting code to see if node was just created.
        # The call to setGates below must be made now as the node is being created because gate settings might be based
        # on smry node info, so you don't want to leave the new node's gates with just some default values.
        self.setGates()

    def __copy__(self):
        newNode = node(self.varNum, self.varName, self.run, self.nodeID)
        newNode.nVisitsIncPath = self.nVisitsIncPath
        newNode.nVisitsExcPath = self.nVisitsExcPath
        newNode.nVisitsIncPathUnique = self.nVisitsIncPathUnique
        newNode.nVisitsExcPathUnique = self.nVisitsExcPathUnique
        newNode.maxScoreIncPath = self.maxScoreIncPath
        newNode.maxScoreExcPath = self.maxScoreExcPath
        newNode.sumScoreIncPath = self.sumScoreIncPath
        newNode.sumScoreExcPath = self.sumScoreExcPath
        newNode.sumScoreSqIncPath = self.sumScoreSqIncPath
        newNode.sumScoreSqExcPath = self.sumScoreSqExcPath
        newNode.allScoresIncPath = self.allScoresIncPath
        newNode.allScoresExcPath = self.allScoresExcPath
        newNode.meanScoreInc = self.meanScoreInc
        newNode.meanScoreIncErr = self.meanScoreIncErr
        newNode.rmsScoreInc = self.rmsScoreInc
        newNode.meanScoreExc = self.meanScoreExc
        newNode.meanScoreExcErr = self.meanScoreExcErr
        newNode.rmsScoreExc = self.rmsScoreExc
        newNode.incPathOpen = self.incPathOpen
        newNode.excPathOpen = self.excPathOpen
        newNode.newlyCreated = self.newlyCreated
        return newNode

    def getNodeParent(self):
        """Get the parent of a node that is specified by giving its nodeID and variable"""
        childNode = self
        locNodeCollection = self.run.nodesColl
        if childNode.nodeIsTheRoot():
            result = None  # Root node has no parent
        else:
            childID = childNode.nodeID
            childVarNum = childNode.varNum
            parentVarNum = childVarNum - 1
            # If the child node is on the include side of its parent, then the parent's id is the child's id after
            # removing the bit corresponding to the parent's variable number.  Otherwise the parent will have the
            # same id as the child
            parentID = childID - 2**parentVarNum if childNode.whichSideIsNode() == 1 else childID
            result = locNodeCollection.getNode(parentVarNum, parentID)
        return result

    def countChildNodes(self):
        """Tells how many child nodes a given node has.  Recall that the children of a node are only created when
        they are needed at the time a path is being created"""
        locNode = self
        numChildNodes = 0
        if locNode.getNodeChild('inc') is not None: numChildNodes += 1
        if locNode.getNodeChild('exc') is not None: numChildNodes += 1
        return numChildNodes

    def getNodeChild(self, side):
        """Get the child node of a node that is specified by giving its nodeID and variable.  One also needs to specify
        which side one wants, since each node has up to two children, except those in the last layer which have zero."""
        locNode = self
        locNodeCollection = locNode.run.nodesColl
        locNodeID = locNode.nodeID
        locNodeVarNum = locNode.varNum
        if locNode.nodeIsInLastLayer():
            result = None  # Nodes in the last layer have no children
        else:
            if side == 'inc':
                # There is no off-by-one-bug in the next line.  A node in layer n (starting from zero) can contain
                # variables up to n-1 (layer zero has no variables).
                # This is because the layer-n node decides to fork left or right depending on whether one is including
                # or excluding var n.  So, the root node (layer 0) can not contain any variables.  Of the two layer 1
                # nodes, one has var 0 and the other does not.
                childID = locNodeID + 2**locNodeVarNum
                childVarNum = locNode.varNum + 1
            elif side == 'exc':
                childID = locNodeID
                childVarNum = locNode.varNum + 1
            else:
                print("ERROR: Something is wrong.  Child node request must be for either include-side of exclude side")
                sys.exit()
            result = locNodeCollection.getNode(childVarNum, childID)
        return result

    def nodeReachableFromParent(self):
        """Given a node, checks the parent's gate status to see if the input node is reachable"""
        childNode = self
        parentNode = self.getNodeParent()
        # The result is the status of the gate on whichever side of the parent the child node sits.
        result = parentNode.incPathOpen if childNode.whichSideIsNode() == 1 else parentNode.excPathOpen
        return result

    def nodeReachableFromRoot(self):
        """Given a node, checks whether or not gates in the tree are such that one could reach the node."""
        result = True  # Initialize to true, then set false if at any point node isn't reachable from its parent
        currNode = self
        while not currNode.nodeIsTheRoot():  # Step upwards until you reach the root node
            if currNode.nodeReachableFromParent():  # Can you reach the parent of the current node?
                currNode = currNode.getNodeParent()  # update currNode to point to its parent
            else:
                result = False
                break
        return result

    def whichSideIsNode(self):
        """Tells whether a given node is on the include or the exclude side (1 or 0) of its parent."""
        # Be careful here.  To decide whether a node is on the include vs exclude side of its parent, you
        # need to look at whether the parent node includes or excludes the variable associated with the child.
        childVarNum = self.varNum
        parentVarNum = childVarNum - 1
        result = 1 if self.nodeID & 2**parentVarNum > 0 else 0
        return result

    def nodeIsTheRoot(self):
        """Tells whether or not the given node is the root node"""
        result = True if self.varNum == 0 else False
        return result

    def nodeIsInLastLayer(self):
        """Tells whether or not the given node is in the last layer of the tree"""
        # Need -1 in next line because variable numbers start from 0
        result = True if self.varNum == len(self.run.dataset.varNums)-1 else False
        return result

    def includeVarPolicy(self, pathType):
        """Given a node and a pathType, implements policy for choosing whether to
        include or exclude the node's variable based on the nodes current performance and visits.

        In full tree mode, you look at the inc/exc results for the specific node
        in the full tree.

        In compressed tree mode, you look at the inc/exc summary node
        for that variable. (The information in the summary node for that variable is
        a projection of all the full-tree nodes with the same variable number).

        In fixed path mode, you just inc/exc a variable based on the user-requested variables.

        In coin-flip mode, you randomly decide whether or not to include the variable.
        """
        locEpsilon = 1e-6
        cpParm = globCPParm  # Set cpParm to global value that was set by playout's init method
        banditMode = globBanditMode  # Set bandit comparison mode to global value that was set by playout's init method
        banditThreshCut = globBanditThreshCut
        includeVar = None  # initialize to prevent uninitialized warning
        if self.incPathOpen and not self.excPathOpen:  # exclude gate is closed
            includeVar = True
        elif self.excPathOpen and not self.incPathOpen:  # include gate is closed
            includeVar = False
        elif self.incPathOpen and self.excPathOpen:  # both gates are open, so you need to make comparison
            if pathType[0] == -1:  # compressed-tree mode
                if self.nVisitsIncPathUnique < self.run.minVisitsForNodeEval or \
                        self.nVisitsExcPathUnique < self.run.minVisitsForNodeEval:
                    includeVar = np.random.uniform(0.0, 1.0, 1)[0] >= 0.5  # flip coin
                else:  # both have been visited enough, so compare inc to exc
                    banditScoreInc, banditScoreExc = banditScore(self, cpParm, banditMode, banditThreshCut)
                    # If bandit scores are extremely close, then flip a coin.  Prevents system from getting stuck
                    includeVar = (banditScoreInc >= banditScoreExc) if \
                        abs(banditScoreInc - banditScoreExc) > locEpsilon else \
                        np.random.uniform(0.0, 1.0, 1)[0] >= 0.5
            elif pathType[0] == -2:  # full-tree mode
                if self.nVisitsIncPath == 0 and self.nVisitsExcPath == 0:
                    includeVar = (np.random.uniform(0.0, 1.0, 1)[0] >= 0.5)  # flip a coin
                elif self.nVisitsIncPath == 0 and self.nVisitsExcPath != 0:  # only exclude has been visited
                    includeVar = True
                elif self.nVisitsIncPath != 0 and self.nVisitsExcPath == 0:  # only include has been visited
                    includeVar = False
                else:  # both have been visited, so make comparison to choose inc vs. exc
                    banditScoreInc, banditScoreExc = banditScore(self, cpParm, banditMode, banditThreshCut)
                    # If bandit scores are extremely close, then flip a coin.  Prevents system from getting stuck
                    includeVar = (banditScoreInc >= banditScoreExc) if \
                        abs(banditScoreInc - banditScoreExc) > locEpsilon else \
                        np.random.uniform(0.0, 1.0, 1)[0] >= 0.5
            elif pathType[0] == -9:  # coin-flip mode
                includeVar = (np.random.uniform(0.0, 1.0, 1)[0] >= 0.5)  # flip a coin
            elif pathType[0] >= 0:  # fixed-path mode
                pass
            else:  # unknown mode
                print("ERROR: unknown variable-comparison mode")
                sys.exit()
        else:
            print("ERROR: Something is wrong, the gates on both sides are closed")
            sys.exit()

        if pathType[0] >= 0:  # for fixed-path running, ignore all above calculations and just inc/exc based on list.
            includeVar = True if self.varNum in pathType else False
        return includeVar

    def nodeStatsUpdate(self, locPath, playoutScore):
        """Given a score for a playout, update the statistics on the node's performance."""
        included = self.varNum in locPath.incVars  # See if node's variable is listed in the path's included variables
        locPathIdStr, locPathIdDec = locPath.pathId()  # get the ID num and string for the path.

        if included:
            self.sumScoreIncPath += playoutScore
            self.sumScoreSqIncPath += playoutScore ** 2
            self.maxScoreIncPath = max(self.maxScoreIncPath, playoutScore)
            self.nVisitsIncPath += 1
            if locPathIdStr in self.allScoresIncPath:  # Put score in dictionary at spot corresponding to the path
                self.allScoresIncPath[locPathIdStr].append(playoutScore)
            else:  # Make new dictionary entry is one doesn't exist already
                self.allScoresIncPath[locPathIdStr] = [playoutScore]
            self.nVisitsIncPathUnique = self.countUniquePaths(True)
            self.meanScoreInc, self.meanScoreIncErr, self.rmsScoreInc = self.getMeanScore(True)
        else:
            self.sumScoreExcPath += playoutScore
            self.sumScoreSqExcPath += playoutScore ** 2
            self.maxScoreExcPath = max(self.maxScoreExcPath, playoutScore)
            self.nVisitsExcPath += 1
            if locPathIdStr in self.allScoresExcPath:  # Put score in dictionary at spot corresponding to the path
                self.allScoresExcPath[locPathIdStr].append(playoutScore)
            else:  # Make new dictionary entry is one doesn't exist already
                self.allScoresExcPath[locPathIdStr] = [playoutScore]
            self.nVisitsExcPathUnique = self.countUniquePaths(False)
            self.meanScoreExc, self.meanScoreExcErr, self.rmsScoreExc = self.getMeanScore(False)

        self.setGates()  # update which gates should be open for the node

        return

    def nodeStatsAdd(self, secondNode):
        """Add the stats from a second node to this node."""
        self.nVisitsIncPath += secondNode.nVisitsIncPath
        self.nVisitsExcPath += secondNode.nVisitsExcPath
        self.maxScoreIncPath = max(self.maxScoreIncPath, secondNode.maxScoreIncPath)
        self.maxScoreExcPath = max(self.maxScoreExcPath, secondNode.maxScoreExcPath)
        self.sumScoreIncPath += secondNode.sumScoreIncPath
        self.sumScoreExcPath += secondNode.sumScoreExcPath
        self.sumScoreSqIncPath += secondNode.sumScoreSqIncPath
        self.sumScoreSqExcPath += secondNode.sumScoreSqExcPath
        self.allScoresIncPath = {**self.allScoresIncPath, **secondNode.allScoresIncPath}
        self.allScoresExcPath = {**self.allScoresExcPath, **secondNode.allScoresExcPath}
        self.nVisitsIncPathUnique += secondNode.nVisitsIncPathUnique
        self.nVisitsExcPathUnique += secondNode.nVisitsExcPathUnique
        # The calls to get mean/rms/stdev in the two lines below aren't really adding the new node to the old node per
        # se, they are really just getting the stats for the node now that the information from the additional node has
        # been added.
        self.meanScoreInc, self.meanScoreIncErr, self.rmsScoreInc = self.getMeanScore(True)
        self.meanScoreExc, self.meanScoreExcErr, self.rmsScoreExc = self.getMeanScore(False)
        return

    def countUniquePaths(self, included):
        """Given a node, this returns the number of unique paths that go through the node and either include
        or exclude (dep on argument) its variable."""
        if included:
            nUnique = len(self.allScoresIncPath.values())
        else:
            nUnique = len(self.allScoresExcPath.values())
        return nUnique

    def fracPathsOverThresh(self, included, thresh):
        """Given a node, this returns the fraction of the paths that either include or exclude (dep on argument)
        the variable and have an average score above a given threshold.  Note that we average over multiple playouts
        that follow the same path."""
        # Each entry in the inc dictionary represents one specific path that passes through the node and incl (or excl)
        # the node's variable.  Because multiple playouts may follow the same path, the value for the specific key
        # path is a list that contains all the scores that that path achieved.  Here, we'll be interested in getting
        # the mean score from the (possibly) multiple playouts.
        meanPathScores = []
        if included:
            for pathId, scoresList in self.allScoresIncPath.items():
                # Get the mean score from all times that this path was followed and its variable was included
                meanPathScores.append(sum(scoresList)/len(scoresList))
        else:
            for pathId, scoresList in self.allScoresExcPath.items():
                # Get the mean score from all times that this path was followed and its variable was excluded
                meanPathScores.append(sum(scoresList)/len(scoresList))

        # Get fraction of scores over threshold
        frac, fracErr = fracOverThresh(meanPathScores, thresh)
        return frac, fracErr

    def getMeanScore(self, included):
        """This is a slightly subtle function.  Given a node, it finds the average scores for when the node's variables
        are included (or excluded dep on second argument).  However, since any given path through a node might be
        played out multiple times, one first averages over all the identical playouts and then just gives that a weight
        of one when computing the final average."""

        # The mean score should only count each unique path once.  So, if two unique paths pass through a node and
        # include its variable but one was evaluated once and the other 100 times, they should still get equal weight
        # when evaluating the avg score that results from passing through that node and including its variable.

        # Get the mean score from the (possibly) multiple playouts that all have the same path.
        # Do this either for the case where the node's variable was included or excluded (dep on argument).
        # Each entry in the dictionary is a list of scores for all the playouts with the same path.
        if included:
            meansForPaths = [sum(_)/len(_) for _ in self.allScoresIncPath.values()]
        else:
            meansForPaths = [sum(_)/len(_) for _ in self.allScoresExcPath.values()]

        # Now get the mean score for the paths that include (exclude) the variable.  In these mean values, each
        # unique path is getting equal weight regardless of how many times it has been visited.
        mean, errOnMean, rms = meanSigmaFromList(meansForPaths, doFit=True)
        return mean, errOnMean, rms

    def setGates(self):
        """Consider closing either the include or the exclude side of a node depending on various stats that the node
        has shown during playouts so far.  Note that a closed path could be reopened in the future if the performance
        of the side that has remained open falls sufficiently.  The gate closing decision will be based on results
        either from the individual node, or on the summary node (which includes all nodes for that variable)
        depending on the value of the globProjectedGateDecision.
        The performance of the two sides is compared in one of four ways:
         (mean, threshPctDiff, threshPctAbs, threshPctSignif)

         mean: criteria is diff between mean scores.
         threshPctDiff: criteria is diff between pct of scores over thresh.
         threshPctAbs:  criteria is pct of scores over thresh.
         threshPctSignif:  criteria is significance of pct of scores over thresh
        """

        # Get the summary node for the nodes layer (summary nodes are labelled by the variable number of the layer)
        # If self is a smry node, then locSmryNode and self will wind up being the same thing, but this shouldn't cause
        # any problems in the code below.
        if not hasattr(self.run, 'nodesColl'): return
        locSmryNode = self.run.nodesColl.smry[self.varNum]

        # This block ensures that if we are in projectedGateDecision mode, that newly created nodes will take on the
        # gate values of their corresponding smry node.
        if self.newlyCreated:
            self.newlyCreated = False
            if globProjectedGateDecision:
                self.incPathOpen = locSmryNode.incPathOpen
                self.excPathOpen = locSmryNode.excPathOpen

        # Decide whether gate-closing decision for the node should be based on information from the node itself or
        # on information from all the nodes for the corresponding variable (i.e. the summary node for the layer)
        # Note in the if/else block below, that if self is a smrynode then testNode will be self either way.
        if globProjectedGateDecision:  # set testnode = smrynode so decision will be based on the layer's summary node
            testNode = locSmryNode
        else:  # Decide based on the node's own statistics
            testNode = self

        # See if we have passed the threshold number of playouts on each side and so can make gate-closing decision.
        # If not, then just make sure both gates are open.
        if testNode.nVisitsIncPathUnique > testNode.run.minVisitsForGateEval[-1] and \
           testNode.nVisitsExcPathUnique > testNode.run.minVisitsForGateEval[-1]:
            diff = 0  # initialize variable to prevent syntax warning about using before defining.
            nSigmaDiff = 0  # initialize variable to prevent syntax warning about using before defining.
            gateDecisionAlreadyMade = False
            if globGateMode == 'mean':  # criteria is diff between mean scores.
                diff = testNode.meanScoreInc - testNode.meanScoreExc
                errOnDiff = (testNode.meanScoreIncErr ** 2 + testNode.meanScoreExcErr ** 2) ** 0.5
                nSigmaDiff = abs(diff / errOnDiff)
            elif globGateMode == 'threshPctDiff':  # criteria is diff between pct of scores over thresh.
                threshold = self.run.getThresholdScore(globGateThreshCut, globPctDefn)  # scores over thresh are counted
                fracOverThreshInc, errOnFracInc = testNode.fracPathsOverThresh(True, threshold)
                fracOverThreshExc, errOnFracExc = testNode.fracPathsOverThresh(False, threshold)
                diff = fracOverThreshInc - fracOverThreshExc
                errOnDiff = (errOnFracInc**2 + errOnFracExc**2) ** 0.5
                nSigmaDiff = 0 if errOnDiff == 0 else abs(diff/errOnDiff)
            elif globGateMode == 'threshPctAbs':  # criteria is pct of scores over thresh
                threshold = self.run.getThresholdScore(globGateThreshCut, globPctDefn)  # scores over thresh are counted
                fracOverThreshInc, errOnFracInc = testNode.fracPathsOverThresh(True, threshold)
                fracOverThreshExc, errOnFracExc = testNode.fracPathsOverThresh(False, threshold)
                self.incPathOpen = (fracOverThreshInc > globGateIncCut/100.0)  # Test condition and set inc-gate status
                self.excPathOpen = (fracOverThreshExc > globGateExcCut/100.0)  # Test condition and set exc-gate status
                gateDecisionAlreadyMade = True
            elif globGateMode == 'threshPctSignif':  # criteria is significance of pct of scores over thresh
                # So, for example, if the threshold cut is 50th percentile, then we close a gate if significantly less
                # than 50% of paths through that side of the node are below the 50th percentile.
                # If a variable is neutral (doesn't help and doesn't hurt), then n% of the paths through both sides
                # should be in the top n%.

                # Note that we do not use the global percent definition in this routine, since it only makes sense
                # to use percentile in this routine.  Rather than doing an "assert" to force that definition throughout,
                # we just override the definition if the rest of the code is using "absolute" rather than "percentile."
                # This allows the rest of the code to continue using absolute if that's what the user wants.
                locPctDefn = 'percentile'
                threshold = self.run.getThresholdScore(globGateThreshCut, locPctDefn)  # scores over thresh are counted
                fracOverThreshInc, errOnFracInc = testNode.fracPathsOverThresh(True, threshold)
                fracOverThreshExc, errOnFracExc = testNode.fracPathsOverThresh(False, threshold)
                # If e.g., cut (globGateThreshCut) is 50%, then that means we ask for the fraction of paths above the
                # 50th percentile.  If we then find 40+/-10% of events above the 50th percentile, then we have a
                # significance of -1 sigma.  That -1 then gets compared to the cut to decide about closing the gate.
                signifInc = (fracOverThreshInc - globGateThreshCut/100.0)/errOnFracInc
                signifExc = (fracOverThreshExc - globGateThreshCut/100.0)/errOnFracExc
                # Be careful with the signs.  If we have 40+/-10% and we expect 50%, then signif is -1.0 sigma.  So
                # a gate-closing cut might be something like -2.0.  Negative values for this cut are likely to be a
                # standard use case, since you'll generally be closing a gate if leaving it open causes you to have
                # fewer than x% of the paths in the top x percentile.  However, for the case where you want to see how
                # well you can do with a very limited number of variables, you may want to have significantly positive
                # cut values for gateIncCut.  A positive cut for gateIncCut says that unless a variable makes a positive
                # impact, it will be excluded; no-harm isn't good enough.
                self.incPathOpen = (signifInc > globGateIncCut)  # Test condition and set inc-gate status
                self.excPathOpen = (signifExc > globGateExcCut)  # Test condition and set exc-gate status
                gateDecisionAlreadyMade = True
            else:
                print("ERROR: unknown gate-closing comparison mode")
                sys.exit()

            if not gateDecisionAlreadyMade:  # Depending on gate-closing mode, the decision may have already been made
                # The next couple lines implement an important (and somewhat difficult) decision:
                # What do you do with a variable that doesn't seem to help or hurt the score?
                # Keep it, i.e., close the exclude gate, or reject it, i.e., close the include gate.
                # Note that the catch-all else block at the end ensures that gates that were previously closed
                # will reopen if the conditions for closing one side are no longer satisfied.
                if (diff < 0) and nSigmaDiff > globGateExcCut:  # Close include gate (i.e., exclude the variable)
                    self.incPathOpen = False
                    self.excPathOpen = True
                elif (diff > 0) and nSigmaDiff > globGateIncCut:  # Close exclude gate (i.e., include the variable)
                    self.incPathOpen = True
                    self.excPathOpen = False
                else:  # we don't currently satisfy conditions for closing either side
                    self.incPathOpen = True
                    self.excPathOpen = True

            # If we're in an edge case where neither gate satisfies the condition to be open, we obviously can't have
            # both gates closed, so rather than trying to do something fancy, just open both gates and continue.  My
            # suspicion is that this will only happen when playout counts are still low (i.e. early in running).
            if not (self.incPathOpen or self.excPathOpen):
                self.incPathOpen = True
                self.excPathOpen = True
        else:
            self.incPathOpen = True
            self.excPathOpen = True
        return


class pathStatsColl:
    """Objects that collect information on multiple paths over multiple playouts"""

    def __init__(self):
        self.pathStatsDict = dict()

    def pathStatsUpdate(self, locPath, score):
        """Given a path and a score, add that information either to a
        new entry in the dictionary, or update the existing entry"""
        locPathIdStr, locPathIdDec = locPath.pathId()  # get the ID num and string for the path.
        if locPathIdStr not in self.pathStatsDict:  # If path doesn't yet have pathstats obj in dict then create/add it.
            newStats = pathStats(locPath)
            self.pathStatsDict[locPathIdStr] = newStats
        # retrieve path and update info
        currStats = self.pathStatsDict[locPathIdStr]
        currStats.nVisits += 1
        if score < currStats.minScore: currStats.minScore = score
        if score > currStats.maxScore: currStats.maxScore = score
        currStats.sumScore += score
        currStats.sumScoreSq += score ** 2
        currStats.mean, currStats.errOnMean, currStats.stdev = \
            meanSigma(currStats.nVisits, currStats.sumScore, currStats.sumScoreSq)
        self.pathStatsDict[locPathIdStr] = currStats
        return


class pathStats:
    """An object that holds information about a path"""

    def __init__(self, locPath):
        self.path = locPath
        self.pathIdStr, self.pathIdDec = locPath.pathId()
        self.nVisits = 0
        self.maxScore = 0
        self.minScore = 9e9
        self.sumScore = 0
        self.sumScoreSq = 0
        self.mean = 0
        self.errOnMean = 0
        self.stdev = 0


class path:
    """A path through the list of all variables.  One path corresponds
    to one specific set of nodes whose variables should be included in
    the path."""

    def __init__(self, locRun, pathType, allowedVars, requiredVars):
        self.locRun = locRun
        self.pathType = pathType
        self.nodesColl = locRun.nodesColl
        self.allVars = locRun.dataset.varNums
        self.nodesList, self.incVars, self.excVars = self.chooseNodes(self.pathType, allowedVars, requiredVars)
        if len(self.incVars) == 0:  # If you got zero included variables, try again using coin-flip mode.
            self.locRun.nFallbackPath += 1
            while len(self.incVars) == 0:
                self.nodesList, self.incVars, self.excVars = self.chooseNodes([-9], allowedVars, requiredVars)

    def chooseNodes(self, locPathType, allowedVars, requiredVars):
        """Choose a path from the full set of variables. The pathtype
        variables changes how the code chooses nodes when the includeVarPolicy is called."""

        assert len(locPathType) > 0, 'No path-type directive given'
        # This code is somewhat subtle.  You loop over all the variables.  For each variable (starting at the top
        # of the tree) you get the corresponding node.  You then evaluate the variable controlled by the node to see if
        # you include it.  THEN, that decision gets appended onto the incVars and excVars array.  Now those new values
        # for incVars and excVars determine the node that you look at during the next iteration of the loop.
        nodesList = []
        incVars = []
        excVars = []
        varDecisionMade = [False] * len(self.allVars)  # Set true once decision is made on var (prevents reversing it)
        for varNum in self.allVars:
            # Get the nodeID and then the node that that needs to be evaluated to see if its variable will be included.
            currNodeID = nodeIDFromVarList(incVars, varNum)
            currNode = self.nodesColl.getNode(varNum, currNodeID, createNode=True)
            nodesList.append(currNode)

            if varNum in requiredVars:  # if var is in required list, include it in list and set decision as complete
                incVars.append(varNum)
                varDecisionMade[varNum] = True

            if varNum not in allowedVars:  # if var not in allowed list, exclude it and set decision as complete
                excVars.append(varNum)
                varDecisionMade[varNum] = True

            # It's important not to finish the loop if the incl/excl decision has been made based on allowed/required
            # lists.  Otherwise the code in the rest of the loop might reverse the list-based decision.
            if varDecisionMade[varNum]: continue  # Don't complete rest of this iteration if decision is already made

            # In compressed-tree mode, the node to evaluate for inc/exc is the summary node, not the tree node
            if locPathType[0] == -1:
                evalNode = self.nodesColl.smry[varNum]
            else:  # Not pathType mode of -1, and so the node itself, not the summary node, should be used in eval
                evalNode = currNode

            # inc/exc variable based on: node's stats, the policy, and on whether you've reached max num of vars allowed
            # < rather than <= in next line because you're testing whether or not you still have space for one more var
            if (len(incVars) < globNVarsMax or globNVarsMax == 0) and evalNode.includeVarPolicy(locPathType):
                incVars.append(varNum)
            else:  # do not include the variable
                excVars.append(varNum)
        return nodesList, incVars, excVars

    def getIncVarNames(self):
        """report names of the variables included in path"""
        varNameList = []
        for i in self.incVars:
            varNameList.append(i.varName)
        return varNameList

    def pathId(self):
        """Given a path, returns the decimal and the binary string
        representation of the pathID, where the binary representation
        shows which variables are included and which are excluded"""
        pathIdDecimal = 0
        for i in self.incVars:
            pathIdDecimal += 2 ** i
        pathIdStr = np.binary_repr(pathIdDecimal, width=len(self.allVars))
        return pathIdStr, pathIdDecimal


class playouts:
    """A collection of playouts.  Playouts are generally run in multiple threads, then we update statistics when
    all are complete"""
    def __init__(self, locRun, eventsPerPlayout, pathType, cpParm=0.7071, banditMode='thresh', banditThreshCut=0,
                 gateMode='threshPctAbs', gateThreshCut=5.0, gateIncCut=2.0, gateExcCut=1.5,
                 ksgK=1, nVarsMax=0, numPlays=1, allowedVars=None, requiredVars=None,
                 exclusiveEventsOnly=False, sampleRandomly=True, projectedGateDecision=False, pctDefn='percentile'):
        # This method is the only place where global variables are set.  They will therefore remain the same
        # for the entire set of playouts.  They might then be altered in a future set of playouts.
        global globCPParm, globBanditMode, globBanditThreshCut, \
            globGateMode, globGateThreshCut, globGateIncCut, globGateExcCut, \
            globksgK, globNVarsMax, globProjectedGateDecision, globPctDefn
        globCPParm = cpParm
        globBanditMode = banditMode
        globBanditThreshCut = banditThreshCut
        globGateMode = gateMode
        globGateThreshCut = gateThreshCut
        globGateIncCut = gateIncCut
        globGateExcCut = gateExcCut
        globksgK = ksgK
        globNVarsMax = nVarsMax
        globProjectedGateDecision = projectedGateDecision
        globPctDefn = pctDefn

        # If events drawn are to be exclusive, reset counters that tell which events have been drawn and shuffle data
        # Exclusivity of events can not be maintained between sets of playouts.  If you want n playouts that have no
        # overlapping events among the different playouts, then just set numPlays=n when you call playouts.
        if exclusiveEventsOnly:
            locRun.dataset.resetDrawnCounters()
            locRun.dataset.shuffleData()

        # If no allowed variables list is specified in call, then all variables are allowed
        if allowedVars is None: allowedVars = locRun.dataset.varNums
        # Confirm that user hasn't given a specific path and then disallowed one of its variables
        # If user specified a path, all of its variables must be among allowed variables.  Confirm this.
        if pathType[0] >= 0:
            for iVar in pathType:
                assert iVar in allowedVars, "{0:70s} {1:10d}".\
                    format("Variable in specified path not among user-allowed variables", iVar)

        if requiredVars is None: requiredVars = []
        # If the user has some required variables, they obviously need to be among the allowed variables.  Confirm this.
        for iVar in requiredVars:
            assert iVar in allowedVars, "{0:70s} {1:10d}". \
                format("A user-required variable is not among the user-allowed variables", iVar)
        # If user specified a path, all required variables must be included in that path.  Confirm this.
        if pathType[0] >= 0:
            for iVar in requiredVars:
                assert iVar in pathType, "{0:70s} {1:10d}". \
                    format("A variable in specified path is missing a required variable", iVar)

        # Number of required variables must not be more than total number of allowed variables (if set)
        assert (globNVarsMax == 0 or len(requiredVars) <= globNVarsMax), "{0:70s} {1:10d} {2:10d}".\
            format("Number of required vars greater than number of allowed vars", len(requiredVars), globNVarsMax)

        # Number of variables in a fixed path must not be more than total number of allowed variables (if set)
        assert (globNVarsMax == 0 or len(pathType) <= globNVarsMax), "{0:70s} {1:10d} {2:10d}". \
            format("Number of vars in fixed path greater than number of allowed vars", len(requiredVars), globNVarsMax)

        # In the bandit formula, cpParm parameter controls tendency to explore new paths vs exploiting existing ones.
        # 1/sqrt(2) is the default (from the Browne paper).  cpParm=0 would cause tree to only follow best path, and
        # cpParm >> typical max values of MI (i.e. cpParm >> 1) would cause it to always balance number of visits
        # independent of score.  Although, 1/sqrt(2) is the default in the paper, they do say that adjustments are
        # likely needed.  One can use the banditPenaltyDiff utility function in the code to understand the impact of
        # different values of cpParm.

        nParallel = locRun.nParallel
        numPlaysRemaining = numPlays

        while numPlaysRemaining > 0:
            # make list of playouts along with a prepared dataset for each.
            playList = [playout(locRun, eventsPerPlayout, pathType, allowedVars, requiredVars,
                                exclusiveEventsOnly, sampleRandomly)
                        for _ in range(min(numPlaysRemaining, nParallel))]

            # Make the list of datasets.  This can be done single-threaded or in a multiprocess pool.
            # I thought the pool would be much faster, but it seems to be much slower (~20x).
            # I suspect that this is because it's not a cpu intensive process, but rather that it's
            # mostly moving data back and forth.  I'll leave the pool version in place in case one ever
            # wants to use it, but for now I'll just set a switch to always single-process.
            doDataPrepInParallel = False
            if doDataPrepInParallel:
                myPool = mp.Pool(processes=nParallel)  # make multiprocess pool with nParallel processes
                datasetList = myPool.map(playout.prepData, playList)  # fill list with dataset results from processes
                myPool.close()  # close/cleanup the pool
                myPool.join()  # close/cleanup the pool
            else:
                datasetList = [i.prepData() for i in playList]  # create dataset for each playout

            # Process sets in parallel to get nParallel scores.
            doProcessingInParallel = True  # If false, will ignore user-input defining number to run in parallel
            if doProcessingInParallel:
                myPool = mp.Pool(processes=nParallel)  # make multiprocess pool with nParallel processes
                scores = myPool.map(go, datasetList)  # fill a list with the score results from the processes
                myPool.close()  # close/cleanup the pool
                myPool.join()  # close/cleanup the pool
            else:
                scores = [go(i) for i in datasetList]

            for i, val in enumerate(playList):  # loop over playouts in the playout list and update score and paths

                val.score = scores[i]  # attach the score to the playout

                updateNodes(val.path, val.score)  # update nodes by passing current path and its score

                locRun.pathStatsColl.pathStatsUpdate(val.path, val.score)  # add path information

                locRun.playoutList.append(val)  # add current playout to the run's playout list

                locRun.updateRunStats()  # Update the run-level stats to include results of the new playout

            numPlaysRemaining -= nParallel

            updateAllTreeGates(locRun)  # Tree-node gates must be re-updated (see called function for reason)

            makeReachableStubsList(locRun)  # Make list of nodes that are reachable from root and have missing children

            initializeGlobalVars()  # Restore all global vars to prevent carryover from one set of playouts to the next


class playout:
    """one pass through the data along a specific path"""

    def __init__(self, locRun, eventsPerPlayout, pathType, allowedVars, requiredVars,
                 exclusiveEventsOnly, sampleRandomly):
        self.pathType = pathType
        self.allowedVars = allowedVars
        self.requiredVars = requiredVars
        self.exclusiveEventsOnly = exclusiveEventsOnly
        self.sampleRandomly = sampleRandomly
        self.cpParm = globCPParm
        self.banditMode = globBanditMode
        self.banditThreshCut = globBanditThreshCut
        self.projectedGateDecision = globProjectedGateDecision
        self.pctDefn = globPctDefn
        self.gateMode = globGateMode
        self.gateThreshCut = globGateThreshCut
        self.gateIncCut = globGateIncCut
        self.gateExcCut = globGateExcCut
        self.ksgK = globksgK
        self.nVarsMax = globNVarsMax
        self.dataset = locRun.dataset
        self.path = path(locRun, self.pathType, allowedVars, requiredVars)
        # Each playout gets its own copy of the nodes in the playout.
        # This is so that future playouts won't change the information in this playout's copy.
        # This allows us to see the all the nodes' info for each playout at the time the playout was run.
        if pathType[0] != -1:  # Not compressed-tree mode, so playlist gets copy of nodes along the path
            self.nodesList = [i.__copy__() for i in self.path.nodesList]
        else:  # Compressed-tree mode.  Playlist's node copy should be of summary nodes.
            self.nodesList = list(locRun.nodesColl.smry.values())

        # If you're running fixed-path mode, then you'll most likely be calling the same run with multiple playouts and
        # comparing results across playouts.  So the code will assume that you want each playout to begin in the
        # standard starting state (i.e., with all gates open).  If you don't want that, then just comment out this line.
        # Note that opening all the gates ahead of a fixed-path playout won't cause problems if you later run
        # normal playouts where the code chooses a path.  That's because the gates will get reevaluated at the start
        # of that ordinary run based on all the playouts that have come before.  So, they won't still be all
        # open because of this line.
        if pathType[0] >= 0: openAllGates(self.nodesList)  # Recall that a pathType>=0 means fixed path running.

        self.pathsCountD = locRun.pathsCount()  # This dictionary will have all pathcount info at moment playout started
        self.score = -999  # just set to an initial value that makes clear that it hasn't been calculated yet.
        if eventsPerPlayout != 0:  # User requested specific number of sig/bkd events
            self.nEvents = eventsPerPlayout  # nSig and nBkd each equals nEvents (so tot events = 2*nEvents)
        else:  # 0 means use all events.  Since nsig must equal nbkd when running, set  number eqaul to smaller value.
            self.nEvents = min(len(self.dataset.sigWts), len(self.dataset.bkdWts))

    def prepData(self):
        """Removes from the data the columns corresponding to unused variables and then selects the
        requested number of events at random"""
        # Trailing letters following "sig" and "bkd" in vars below just helps keep track of steps in processing the data
        # Note that these will be pointers to the vars, weights, and nDrawn arrays.  So when they get
        # shuffled by the call to draw a weighted sample, the originals will be shuffled too.  Which is fine.
        sigA = self.dataset.sigVars
        bkdA = self.dataset.bkdVars
        sigWtsA = self.dataset.sigWts
        bkdWtsA = self.dataset.bkdWts
        nTimesDrawnSigA = self.dataset.nTimesDrawnSig
        nTimesDrawnBkdA = self.dataset.nTimesDrawnBkd
        sigEff = self.dataset.sigSampleEff
        bkdEff = self.dataset.bkdSampleEff

        # Do a weighted sampling.  This will be either exclusive or non-exclusive events depending on parameter
        sigB = weightedSample(sigA, sigWtsA, sigEff, self.nEvents,
                              self.exclusiveEventsOnly, nTimesDrawnSigA, self.sampleRandomly)
        bkdB = weightedSample(bkdA, bkdWtsA, bkdEff, self.nEvents,
                              self.exclusiveEventsOnly, nTimesDrawnBkdA, self.sampleRandomly)

        # For the selected events, do a row-by-row copy of sigVars keeping only the variables (i.e. columns)
        # that should be included in this playout.
        sigC = []
        for iRow, currRow in enumerate(sigB):
            newRow = [currRow[i] for i in self.path.incVars]  # Make a list of the included vars for the current row
            sigC.append(newRow)  # Add this new row (which now has only incl vars along with the weight) to array

        # Do row-by-row copy of bkdVars keeping only the variables (i.e. columns) that have been selected
        bkdC = []
        for iRow, currRow in enumerate(bkdB):
            newRow = [currRow[i] for i in self.path.incVars]  # Make a list of the included vars for the current row
            bkdC.append(newRow)  # Add this new row (which now has only incl vars along with the weight) to array

        # Make a list of which variable numbers are discrete, where the numbering must now be relative
        # to the variables in the thinned columns.
        discList = []
        for i, incVarNum in enumerate(self.path.incVars):  # Loop over all the included variable numbers.
            if incVarNum in self.dataset.discVars:  # See if the current included variable number is discrete.
                discList.append(i)  # Add it to the list of disc vars (numbering relative to thinned columns).

        return sigC, bkdC, discList


def weightedSample(locVars, locWts, locEff, nReq, exclusiveEvents, locNTimesDrawn, sampleRandomly):
    """From a sample of events return a random sample drawn according to weights.  Depending on argument, the events
    returned may not have been previously drawn since the last time the counters were reset.  Note that requesting
    exclusive events means that in a set of n playouts, the event samples will not overlap between the playouts.
    This request does not hold between sets of playouts.  Independent of this setting, a single playout will never
    have repeated events."""
    # make sure lengths of vars, wts, and nTimesDrawn are all the same
    assert len(locVars) == len(locWts) == len(locNTimesDrawn), "{0:55s} {1:10d} {2:10d} {3:10d}". \
        format("length of vars and weights arrays not equal", len(locVars), len(locWts), len(locNTimesDrawn))

    # Decide whether the requested sample is a small or a large fraction of the available events
    # We'll use different methods for the two cases to make the code more efficient.
    # For small samples, randomly grab and test events.  For large samples, shuffle all data then select from top.
    totEvents = len(locVars)  # Number of events in the full sample that you're drawing from.
    fracEst = (nReq / locEff) / totEvents  # Estimated fraction of full sample you'll need to test to get nReq accepted
    smallSample = True if fracEst < 1.0/2.0 else False

    weightMax = max(locWts)
    resultSample = []
    nSelected = 0
    diceRolls = np.random.uniform(0, 1, totEvents)  # Better performance from throwing all rolls at once

    if smallSample and not exclusiveEvents:  # The small-sample approach won't work for exclusive-event samples
        evtUsed = [False] * totEvents  # initialize list that tells which events have already been checked for inclusion
        nTries = 0
        while nSelected < nReq:
            ranIndex = random.randrange(totEvents)  # Choose a random spot in the vars and weights lists
            weightRatio = locWts[ranIndex] / weightMax
            assert nTries < totEvents, "{:55s} {:10d} {:10d} {:10d}". \
                format("problem in sampling.  Ran out of events", nSelected, nTries, totEvents)
            if not evtUsed[ranIndex]:
                evtUsed[ranIndex] = True  # Event has been tested for inclusion in this sample
                nTries += 1  # increment number of tries.
                if weightRatio > diceRolls[ranIndex]:  # See if we keep this event
                    resultSample.append(locVars[ranIndex])
                    nSelected += 1  # increment the number we've taken.

    else:  # request is either for a large sample or an exclusive sample
        # shuffle vars, weights, and nTimesDrawn counts, but keep them all synced (i.e. random, but all same shuffle)
        if sampleRandomly:
            savedState = random.getstate()  # get state of random generator
            random.shuffle(locVars)  # Shuffle vars
            random.setstate(savedState)  # restore random generator to previous state so shuffle will be identical
            random.shuffle(locWts)  # shuffle weights
            random.setstate(savedState)  # restore random generator to previous state so shuffle will be identical
            random.shuffle(locNTimesDrawn)  # shuffle nTimesDrawn list
        i = 0  # points at row
        while nSelected < nReq:
            assert i < totEvents, 'No events left to test for inclusion in exclusive weighted sample'
            weightRatio = locWts[i] / weightMax
            if weightRatio > diceRolls[i] and (locNTimesDrawn[i] == 0 or not exclusiveEvents):
                resultSample.append(locVars[i])
                locNTimesDrawn[i] += 1  # increment the counter for how many times event has been drawn
                i += 1  # increment the event we're pointing to
                nSelected += 1  # increment the number we've taken
            else:
                i += 1  # increment the event we're pointing to

    return resultSample


def go(locData):
    """make one pass through the data and calculate MI"""
    sigSet = locData[0]
    bkdSet = locData[1]
    discList = locData[2]

    # If list of discrete variables is empty, call the continuous-only version of MI-calc routine.  Otherwise
    # pass the data and the list of discrete variables (numbered relative to the columns in the data you're passing,
    # not to the original variable numbers) to the version that can handle both continuous and discrete variables.
    if len(discList) == 0:
        score = mi.mi_binary(sigSet, bkdSet, globksgK)
    else:
        score = mi.mi_binary_discrete(sigSet, bkdSet, discList, globksgK)
    return score


def banditScore(locNode, cpParm, banditMode, banditThreshCut):
    """Scores for making an include/exclude choice at a node.  The higher the score, the better the choice.
    Comparison method is set by banditMode parameter to be either node's mean value or node's max value"""
    nVisitsInc = locNode.nVisitsIncPath
    nVisitsExc = locNode.nVisitsExcPath
    nVisitsIncU = locNode.nVisitsIncPathUnique
    nVisitsExcU = locNode.nVisitsExcPathUnique
    # To keep system from getting stuck, use a mix of unique-path visits and total visits.  Ideally, one would just
    # use unique visits, but if the over-visit penalty doesn't include anything about total visits, then if it starts
    # visiting a path over and over, there's nothing to kick it into a different state.
    # I've gone back and forth about how much to weight unique visits vs all visits, but at the moment (Wed 04 Mar 2020)
    # I think it should be based completely on all visits.
    allVisitsWeight = 1.0  # 0 means only unique visits matter, 1 means only total visits matter.
    uniqueVisitsWeight = 1.0 - allVisitsWeight
    nVisitsIncFinal = allVisitsWeight * nVisitsInc + uniqueVisitsWeight * nVisitsIncU
    nVisitsExcFinal = allVisitsWeight * nVisitsExc + uniqueVisitsWeight * nVisitsExcU
    nVisitsBothFinal = nVisitsIncFinal + nVisitsExcFinal

    assert nVisitsIncFinal > 0, 'Too few visits on include side to use bandit formula'
    assert nVisitsExcFinal > 0, 'Too few visits on exclude side to use bandit formula'

    if banditMode == 'mean':
        # The routine called here to compute the means, will average over multiple playouts through the node that all
        # have the same path.  Thus a playout that was tried once and another that was tried 100 times each get equal
        # weight in the avg result for finding the avg when the node is included vs excluded.
        meanScoreInc, meanScoreIncErr, rmsScoreInc = locNode.getMeanScore(True)
        meanScoreExc, meanScoreExcErr, rmsScoreExc = locNode.getMeanScore(False)
        scoreForComparisonInc = meanScoreInc
        scoreForComparisonExc = meanScoreExc
    elif banditMode == 'max':
        scoreForComparisonInc = locNode.maxScoreIncPath
        scoreForComparisonExc = locNode.maxScoreExcPath
    elif banditMode == 'thresh':
        threshold = locNode.run.getThresholdScore(banditThreshCut, globPctDefn)  # threshold score that counts as win
        fracOverThreshInc, errOnFracInc = locNode.fracPathsOverThresh(True, threshold)
        fracOverThreshExc, errOnFracExc = locNode.fracPathsOverThresh(False, threshold)
        scoreForComparisonInc = fracOverThreshInc
        scoreForComparisonExc = fracOverThreshExc
    elif banditMode == 'untested':  # Favor whichever side of the node that has more open untested paths below it.
        # Use the list of "stub" nodes (i.e., aren't in last layer, are reachable from root, and have missing children)
        # These nodes mark the ends of partially searched paths (i.e., stubs).
        # So, when deciding whether to inc/exc a variable controlled by a node, ask which side has more stubs below
        # it in the tree.  So you'll loop through the list of stub nodes and see how many of them are descendants of
        # the current node on the inc and exc side.  Then go to whichever side has more of these stub nodes below it.
        nStubInc = nStubExc = 0  # Initialize count of number of stub nodes on each side
        for stubNode in locNode.run.reachableStubs:  # Loop over all stub nodes
            desc = isDescendant(locNode, stubNode)  # See if the current stub node is a descendant of the calling node
            if desc == 1: nStubExc += 1
            if desc == 2: nStubInc += 1
        # The section below is a little subtle. if you just go to the side with more stubs, in the section below,
        # then when running parallel playouts, you'll wind up repeating the same path for all those playouts until stub
        # counts are updated at the end of the set of playouts.  That's not wrong, but it's inefficient.  So instead,
        # as long as both sides of a node have stubs below it, then just set the bandit results to be equal so that the
        # calling code will just flip a coin to decide which way to go.
        if nStubInc > 0 and nStubExc == 0:
            scoreForComparisonInc = 9e9
            scoreForComparisonExc = 0
        elif nStubExc > 0 and nStubInc == 0:
            scoreForComparisonInc = 0
            scoreForComparisonExc = 9e9
        else:
            scoreForComparisonInc = 999
            scoreForComparisonExc = 999
    else:
        print("ERROR: unknown bandit-score comparison mode")
        sys.exit()

    if scoreForComparisonInc == 999 and scoreForComparisonExc == 999:  # Just set banditInc = banditExc
        banditInc = 1.0
        banditExc = 1.0
    else:  # Scores have not been set equal by hand, so go ahead and compute them normally.
        banditInc = scoreForComparisonInc + 2 * cpParm * np.sqrt(2 * np.log(nVisitsBothFinal) / nVisitsIncFinal)
        banditExc = scoreForComparisonExc + 2 * cpParm * np.sqrt(2 * np.log(nVisitsBothFinal) / nVisitsExcFinal)

    # The next section may override the bandit score computed above
    # If cpParm is greater than 10 or if scores are very close, then just boost the side w/ fewer visits.
    # The cpParm>10 check is so that the user can just force tree to go to unsearched paths by passing a cpparm value
    locEpsilon = 1e-6
    if (cpParm > 10 or abs(banditInc - banditExc) < locEpsilon) and banditMode != 'untested':
        if nVisitsIncFinal < nVisitsExcFinal:
            banditInc = 2.0
            banditExc = 1.0
        elif nVisitsExcFinal < nVisitsIncFinal:
            banditInc = 1.0
            banditExc = 2.0
        else:  # Both sides have equal number of visits.  Returning equal vals should cause calling routine to flip coin
            banditInc = 1.0
            banditExc = 1.0

    return banditInc, banditExc


def nodeIDFromVarList(varList, varNum):
    """Given a list of included variable numbers and the variable number associated with the node, return the nodeID.
    The nodeID is such that when represented in binary, it should just show a 1 for each variable number that was
    included on the path taken from the top of the tree to reach the node.  Note that two different nodes in a tree
    could have the same nodeID.  However, they will be associated with different variables.  So, to uniquely identify a
    node, you need its nodeID and the variable it's associated with.  Note that the nodeID for a node in layer n
    can only have 1's in positions 0 through n-1."""
    nodeID = 0
    for i in varList:
        # When computing the ID, only use variables that are higher in the tree than the current variable.
        # So, you only include vars with a number that's less than (not <=) the var num of the node whose ID you want.
        if i < varNum: nodeID += 2 ** i
    return nodeID


def varListFromBinary(binary_str):
    """Given a binary list of inc/exc variables, this routine returns a list of all the variables that are
    included in the path.  It's just a simple bit-comparison method."""
    result = []
    nVars = len(binary_str)
    vars_int = int(binary_str, 2)
    for i in range(nVars):
        if vars_int & 2**i > 0: result.append(i)
    return result


def updateNodes(locPath, score):
    """Given a path and a score, update the information stored
    for each node in the node collection.  This means both in the nested
    dictionary for all nodes and also the list of summary nodes that are
    one per variable"""
    # Update the nodes that were in the path.  These are refs to the nodes that
    # are in the full nodes dictionary, so updating the nodes in the path will
    # update the relevant nodes in the full nested dictionary.
    for iNode in locPath.nodesList:
        iNode.nodeStatsUpdate(locPath, score)
    # Now update the summary nodes
    smryDict = locPath.locRun.nodesColl.smry
    for iVar in locPath.allVars:  # Loop over all the variables in the dataset, and update the smry node for each var
        smryNode = smryDict[iVar]
        smryNode.nodeStatsUpdate(locPath, score)
    return


def gatesStatus(locNodeList):
    """Given a nodelist, this returns a coded list that shows the gate
    status for all the nodes. Note that this is coded in base-4 so that
    in each place, a 0 corresponds to completely closed (which should
    never happen), a 1 corresponds to only the exclude gate open, a 2
    corresponds to only the include gate is open, and a 3 corresponds to
    both gates open.  So, if you pass it a list of 20 nodes, you'll get
    a large integer that when represented in base-4 will show 0,1,2,3 in
    each place indicating the gate status for each of the 20 nodes."""
    statusCode = 0
    for i in range(len(locNodeList)):
        inode = locNodeList[i]
        nodeStatus = 0
        if inode.excPathOpen: nodeStatus += 1
        if inode.incPathOpen: nodeStatus += 2
        statusCode += nodeStatus * 4 ** i
    return statusCode


def openAllGates(locNodeList):
    """Given a nodelist, open both gates for each node."""
    for currNode in locNodeList:
        currNode.incPathOpen = True
        currNode.excPathOpen = True
    return


def updateAllTreeGates(locRun):
    """Update the gate status for all tree nodes that currently exist.  This routine exists because there's a
    bit of a chicken-and-egg problem with gate updating.  The summary nodes can't be updated until the stats
    on the tree nodes are all updated after a playout.  But if projectGateDecision=True, then the gate
    decisions for the tree nodes depend on the summary node information.  So the order after a playout has to be:
    1) update tree nodes, 2) update summary nodes, 3) update tree gates."""
    nodesD = locRun.nodesColl.all
    for iVar in locRun.varNums:
        layerNodes = nodesD[iVar]  # Dictionary of the nodes for the current variable
        for iNode in list(layerNodes.values()):  # Loop over nodes in the layer
            iNode.setGates()
    return


def makeReachableStubsList(locRun):
    """Make a list of nodes that: a) are not in the last layer, b) are reachable from root, and c) don't have visits
    to both side of their output.  These are the nodes that are sitting at the tops of untested paths."""
    locRun.reachableStubs = []
    nodesD = locRun.nodesColl.all  # Dictionary of all tree nodes
    for iVar in locRun.varNums:  # Loop over layers
        layerNodes = nodesD[iVar]  # Dictionary of the nodes for the current variable (i.e. layer)
        # Note that if nodeID is zero, then it has excluded all vars to this point, so don't count it as a stub
        # just because the path with no variables at all hasn't been tested.  The path with zero variables
        # will obviously never be tested, and so you don't want to pull the code toward this degenerate case.
        for iNode in list(layerNodes.values()):  # Loop over nodes in the layer
            if iNode.nodeIsInLastLayer() and iNode.nodeID == 0:  # Node is at tree end and along the exclude-all edge
                if iNode.nVisitsIncPath == 0 and iNode.nodeReachableFromRoot():
                    locRun.reachableStubs.append(iNode)
            else:  # Node is either not in last lyr, or if it is, it's not along the right-edge of the tree.
                if ((iNode.nVisitsIncPath == 0 and iNode.incPathOpen) or
                    (iNode.nVisitsExcPath == 0 and iNode.excPathOpen)) \
                        and iNode.nodeReachableFromRoot():
                    locRun.reachableStubs.append(iNode)
    return


def isDescendant(nodeA, nodeB):
    """Test to see if nodeB is a descendant of nodeA, I.e., is nodeB below nodeA in the tree and reachable from nodeA.
    result=0: nodeB is not a descendant of nodeA
    result=1: nodeB is a descendant of nodeA on the exclude side of nodeA
    result=2: nodeB is a descendant of nodeA on the include side of nodeA
    Note that if nodeA and nodeB are the same, the result will be zero"""
    nodeIDA = nodeA.nodeID
    varnumA = nodeA.varNum
    nodeIDB = nodeB.nodeID
    nodeIDAIncSide = nodeIDA + 2**varnumA  # Node id of the child on the include side of nodeA
    nodeIDAExcSide = nodeIDA + 0  # Node id of the child on the include side of nodeA

    # If nodeB is at same or higher level in tree, it can't be a descendant of nodeA
    if nodeB.varNum <= nodeA.varNum:
        result = 0
        return result

    # A node in layer n has varnum=n, but has varnum+1 variables (vars count from zero).
    # So when testing whether or not the lower node has all the same variables as the upper node, you only want to
    # test those bits representing variables that the upper node could actually contain, which is the first varNum bits.
    # However, the two tests you're doing are to see if nodeB matches either the exc side or the inc side child of
    # nodeA, and those two child nodes have varnum+1 variables.
    numBitsToTest = varnumA + 1
    bitMask = int('0b'+'1'*numBitsToTest, 2)  # Make a mask that lets you grab lowest bits

    # Test by masking to keep only low-end bits nodeB's id, and then xor it with the ID of nodeA.
    # The result of the xor will be zero if the masked ID of nodeB and the ID of the nodeA child match.
    excSideDesc = True if ((nodeIDB & bitMask) ^ nodeIDAExcSide) == 0 else False
    incSideDesc = True if ((nodeIDB & bitMask) ^ nodeIDAIncSide) == 0 else False

    result = 0  # initialize result variable to say nodeB is not a descendant of nodeA
    if excSideDesc: result = 1  # nodeB is desc of nodeA on exclude side
    if incSideDesc: result = 2  # nodeB is desc of nodeA on include side
    # checks for case where code finds nonsense result that nodeB is a descendant on both sides of nodeA
    assert not (incSideDesc and excSideDesc), 'error: nodeB can not be descendent on both sides of nodeA'
    return result


def meanSigmaFromList(locList, doFit=False):
    """Given a list, returns mean, error on mean, and rms"""
    nEntries = len(locList)
    if doFit:
        mean, rms = norm.fit(locList)
        errOnMean = rms/(nEntries ** 0.5)
    else:
        sumVals = sum(locList)
        sumValsSq = sum([_*_ for _ in locList])
        mean, errOnMean, rms = meanSigma(nEntries, sumVals, sumValsSq)
    return mean, errOnMean, rms


def meanSigma(nEntries, sumVals, sumValsSq):
    """Given n, sum and sum of squares, returns mean, error on mean, and rms"""
    if nEntries == 0:
        mean = 0
        errOnMean = 0
        rms = 0
    else:
        mean = sumVals / nEntries
        variance = (sumValsSq / nEntries) - mean ** 2
        rms = variance ** 0.5
        errOnMean = rms / (nEntries ** 0.5)
    return mean, errOnMean, rms


def fracOverThresh(locList, locThresh):
    """Given a list of values and a threshold, returns the fraction that surpass the threshold and its error"""
    totEntries = len(locList)
    nOverThresh = 0
    epsilon = 1e-9
    for entry in locList:
        if entry >= locThresh: nOverThresh += 1
    nOverThreshErr = nOverThresh ** 0.5 if nOverThresh > 2 else nOverThresh  # set err equal to value if number to low
    frac = 0 if totEntries == 0 else float(nOverThresh)/float(totEntries)
    fracErr = 0 if totEntries == 0 else float(nOverThreshErr) / float(totEntries)
    if fracErr == 0: fracErr = epsilon  # prevent downstream divide-by-zero entries when this
    return frac, fracErr


def replayBest(locRun, nBest, nEvents, selType='mean'):
    """Replay the paths with the highest scores.  Highest is defined by the selType argument and can be any key that
    is available for sorting the results in the pathStatsColl.   Each new playout will use nEvents, and typically, this
    will be larger than the event samples used to find the top paths.  Note that although standard playouts can be
    run in parallel, the current code structure does not readily allow for these replays to be run in parallel.  So,
    they are much slower to do."""
    resortedList = sortPathStatsColl(locRun.pathStatsColl, selType)  # This is a sorted list of pathstats.

    for i in range(nBest):  # For the first nBest pathstats elements, get the path then run it
        currPathStats = resortedList[i]
        currPath = currPathStats.path
        varNumList = currPath.incVars
        playouts(locRun, nEvents, varNumList)
    return


def sortPathStatsColl(pStatsColl, sortBy=None):
    """Given a pathstats collection, sort it by some given key, and then return a list of sorted pathstats"""
    # first convert the dictionary into a sorted list.  This will be sorted by pathIdStr
    if sortBy is None:
        sortBy = []
    sortedListTuple = sorted(pStatsColl.pathStatsDict.items(), key=operator.itemgetter(0))
    sortedList = [row[1] for row in sortedListTuple]  # this is a list of pathStats

    if sortBy == 'nVisits':
        resortedList = sorted(sortedList, key=lambda myPathStats: myPathStats.nVisits, reverse=True)
    elif sortBy == 'mean':
        resortedList = sorted(sortedList, key=lambda myPathStats: myPathStats.mean, reverse=True)
    elif sortBy == 'maxScore':
        resortedList = sorted(sortedList, key=lambda myPathStats: myPathStats.maxScore, reverse=True)
    else:
        resortedList = sorted(sortedList, key=lambda myPathStats: myPathStats.pathIdStr, reverse=False)
    return resortedList


def textReports(locRun, playoutsToReport=100, includeGateInfo=True):
    """Function to call all the other main text-based report functions"""

    print('Run Info')
    runReport(locRun)
    print('\n', '=' * 80, '\n')

    print('Playout Parameters')
    playoutParmsReport(locRun)
    print('\n', '=' * 80, '\n')

    print('Summary-Nodes Info')
    smryNodesReport(list(locRun.nodesColl.smry.values()), includeGateInfo)
    print('\n', '=' * 80, '\n')

    if includeGateInfo:
        print('Tree-Nodes Info')
        treeNodesReport(locRun)
        print('\n', '=' * 80, '\n')

    print('Comparision of bandit judgements of inc/exc sides of all nodes in a layer using global parm values.')
    banditCompareSides(locRun, globCPParm, globBanditMode, globBanditThreshCut)
    print('\n', '=' * 80, '\n')

    print('Number of paths open/closed/tested/untested just before each playout')
    print('Note that with parallel playouts these numbers can be wrong until the end of the batch of playouts')
    pathsCountReport(locRun)
    print('\n', '=' * 80, '\n')

    print('Paths Info (sorted by nVisits)')
    pathsReport(locRun.pathStatsColl, 'nVisits')
    print('\n', '=' * 80, '\n')

    print('Paths Info (sorted by mean score)')
    pathsReport(locRun.pathStatsColl, 'mean')
    print('\n', '=' * 80, '\n')

    print('First', playoutsToReport, 'Playouts (0 means all, negative means from bottom of list)')
    playoutsReport(locRun, playoutsToReport, False, includeGateInfo)
    print('\n', '=' * 80, '\n')

    print('Highest', playoutsToReport, 'Playouts (0 means all, negative means from bottom of list)')
    playoutsReport(locRun, playoutsToReport, True, includeGateInfo)
    print('=' * 80)


def binaryEntropy(p):
    """Given probability for one of the two outcomes in a two-state system, compute the Shannon Entropy in bits"""
    entropy = -p*math.log(p, 2) - (1-p)*math.log(1-p, 2)
    return entropy


def errProbBounds(mutualInfo):
    """Given a mutual information value, this returns the Fano-limit lower bound and the Hellman-Raviv upper-bound
    on the probability of a classification error.  Note that this uses the inverse of Shannon Entropy."""
    plist = np.arange(0.0001, 0.5, 0.0001)
    hlist = [binaryEntropy(p) for p in plist]
    h2inv = interpolate.interp1d(hlist, plist)  # Find the probability corresponding to a given entropy
    errLoBound = h2inv(1 - mutualInfo)
    errLoBound = errLoBound.item()
    errUpBound = (1 - mutualInfo)/2.0
    return errLoBound, errUpBound


def runReport(locRun):
    """Print a summary of a run"""
    locDataset = locRun.dataset

    print('gateEvalParm =', locRun.minVisitsForGateEval, 'nodeEvalParm =', locRun.minVisitsForNodeEval,
          'nParallel =', locRun.nParallel)

    print("{0:<15s} {1:>20s} {2:>7s} {3:>7s} {4:>7s} {5:>15s} {6:>15s}".
          format("Dataset Info:", "dataset name", "nVar", "nSig", "nBkd", "Sig Sample Eff", "Bkd Sample Eff"))
    print("{0:<15s} {1:>20s} {2:>7d} {3:>7d} {4:>7d} {5:>15.2f} {6:>15.2f}".
          format(" ", locDataset.setName, len(locDataset.varNums), len(locDataset.sigVars), len(locDataset.bkdVars),
                 locDataset.sigSampleEff, locDataset.bkdSampleEff))
    print('-' * 60)
    for i in range(len(locDataset.varNums)):
        print("{0:>12s} {1:3d} {2:<48s}".format(" ", locDataset.varNums[i], locDataset.varNames[i]))
    print('-' * 40)
    print('Number of playouts = ', len(locRun.playoutList))
    print("Highest scoring path ={0:8.3f}".format(locRun.highestScore))
    errorLoBound, errorUpBound = errProbBounds(locRun.highestScore)  # Get classification error limits for high score
    print("Range of Classification Error Bounds for top MI = {0:8.3f} - {1:8.3f}".format(errorLoBound, errorUpBound))

    # Print the score and list of included variables for several of the highest-scoring paths
    resortedList = sortPathStatsColl(locRun.pathStatsColl, 'mean')
    numberToPrint = min(5, len(resortedList))
    print("{0:17s} {1:d} {2:45s}".
          format('Results lists for', numberToPrint, 'highest scoring paths (score, nVar, varList):'))
    for i in range(numberToPrint):
        currPath = resortedList[i]
        currVarList = varListFromBinary(currPath.pathIdStr)
        print("{0:>9.3f} {1:>3d} {2:>50s}".
              format(currPath.mean, len(currVarList), str(currVarList)))
    print('Number of fallback paths (Should be zero.  For debugging only)= ', locRun.nFallbackPath)
    return


def pathsReport(pStatsColl, sortBy=None):
    """Sort the pathStats collection by given key, and them print report on paths"""
    if sortBy is None:
        sortBy = []
    resortedList = sortPathStatsColl(pStatsColl, sortBy)

    print("{0:>50s}{1:>8s}{2:>9s}{3:>9s}{4:>6s}{5:>11s}".
          format("10987654321098765432109876543210", "nVisits", "maxScore", "minScore", "stdev", "mean"))
    for i in resortedList:
        # Print one entry line in report for the current element
        print("{0:>50s}{1:>8d}{2:>9.3f}{3:>9.3f}{4:>6.3f}{5:>6.3f}{6:>3s}{7:>6.3f}".
              format(i.pathIdStr, i.nVisits, i.maxScore, i.minScore, i.stdev, i.mean, "+/-", i.errOnMean))
    return


def smryNodesReport(nodeList, includeGateInfo=True):
    """Print a report for a given set of summary nodes."""
    if includeGateInfo:
        print("{0:>30s} {1:>4s} {2:>5s} {3:>5s} {4:>11s} {5:>11s} {6:8s} {7:>12s} {8:>7s} {9:>12s} {10:>7s} {11:>5s}".
              format("var", "var", "times", "times",
                     "mean", "mean", "             ", "pct>thresh", "    ", "pct>thresh", "       ", "     "))
        print("{0:>30s} {1:>4s} {2:>5s} {3:>5s} {4:>11s} {5:>11s} {6:8s} {7:>12s} {8:>7s} {9:>12s} {10:>7s} {11:>5s}".
              format("name", "num", "VInc", "VExc",
                     "Vinc", "Vexc", "nsigDiff", "w/ V inc", "sigDevI", "w/ V exc", "sigDevE", "Gates"))
        # Kludgey way to get at the gate threshold cut that was used the playouts.  Since it may have changed between
        # playouts, just get the thresh cut of the last playout.
        threshPct = nodeList[0].run.playoutList[-1].gateThreshCut
        # Note that the line below is a little tricky to interpret.  The 95 percentile score, for example, at the end
        # of a run, when this routine is likely called, will not generally be what the 95 percentile score was
        # at the time the playout was originally played and decisions were made based on its value.
        thresholdValue = nodeList[0].run.getThresholdScore(threshPct, globPctDefn)
        for i in nodeList:
            meanInc = i.getMeanScore(True)[0]
            errOnMeanInc = i.getMeanScore(True)[1]
            meanExc = i.getMeanScore(False)[0]
            errOnMeanExc = i.getMeanScore(False)[1]
            meanDiff = meanInc - meanExc
            errOnMeanDiff = (errOnMeanInc**2 + errOnMeanExc**2) ** 0.5
            nSigmaMeanDiff = 0 if errOnMeanDiff == 0 else meanDiff/errOnMeanDiff

            fracOverThreshInc, errOnFracInc = i.fracPathsOverThresh(True, thresholdValue)
            fracOverThreshExc, errOnFracExc = i.fracPathsOverThresh(False, thresholdValue)
            sigDevInc = (fracOverThreshInc - threshPct/100)/errOnFracInc
            sigDevExc = (fracOverThreshExc - threshPct/100)/errOnFracExc

            sigDevInc = min(sigDevInc, 9.99)
            sigDevInc = max(sigDevInc, -9.99)
            sigDevExc = min(sigDevExc, 9.99)
            sigDevExc = max(sigDevExc, -9.99)

            print("{0:>30s} {1:>4d} {2:>5d} {3:>5d} {4:>4.2f}{5:3s}{6:>4.2f} {7:>4.2f}{8:3s}{9:>4.2f} "
                  "{10:>8.3f} {11:>5.1f}{12:3s}{13:>4.1f} {14:>7.2f} {15:>5.1f}{16:3s}{17:>4.1f} {18:>7.2f} "
                  "{19:>1d} {20:>1d}".
                  format(i.varName, i.varNum,
                         i.nVisitsIncPathUnique, i.nVisitsExcPathUnique,
                         meanInc, '+/-', errOnMeanInc, meanExc, '+/-', errOnMeanExc, nSigmaMeanDiff,
                         100.0*fracOverThreshInc, '+/-', 100.0*errOnFracInc, sigDevInc,
                         100.0*fracOverThreshExc, '+/-', 100.0*errOnFracExc, sigDevExc,
                         i.incPathOpen, i.excPathOpen))

    else:
        print("{0:>30s} {1:>4s} {2:>5s} {3:>5s} {4:>9s} {5:>9s} {6:>13s} {7:>13s}".
              format("var", "var", "times", "times",
                     "max score", "max score", "avg score", "avg score"))
        print("{0:>30s} {1:>4s} {2:>5s} {3:>5s} {4:>9s} {5:>9s} {6:>13s} {7:>13s}".
              format("name", "num", "V Inc", "V Exc",
                     "w/ V inc", "w/ V exc", "w/ V inc", "w/ V exc"))
        for i in nodeList:
            print("{0:>30s} {1:>4d} {2:>5d} {3:>5d} "
                  "{4:>9.3f} {5:>9.3f} {6:>5.3f}{7:3s}{8:>5.3f} {9:>5.3f}{10:3s}{11:>5.3f}".
                  format(i.varName, i.varNum,
                         i.nVisitsIncPath, i.nVisitsExcPath, i.maxScoreIncPath, i.maxScoreExcPath,
                         i.getMeanScore(True)[0], '+/-', i.getMeanScore(True)[1],
                         i.getMeanScore(False)[0], '+/-', i.getMeanScore(False)[1]))

    # Report allowed/required/rejected status for the variable associated with each summary node.
    currRun = nodeList[0].run  # A bit of a kludge to get run associated with node list that was passed to this function
    acceptRejectStatus = varAcceptRejectStatus(currRun)
    print('  ')
    print("allowed variables:", acceptRejectStatus[0])
    print("required variables:", acceptRejectStatus[1])
    print("optional variables:", acceptRejectStatus[2])
    print("rejected variables:", acceptRejectStatus[3])
    return


def treeNodesReport(locRun):
    """Print a report on the gates for the in-tree nodes.  This is a snapshot of their status,
    and so one likely wants to print it at multiple times during a set of playouts using
    something like 'if playoutNum % x ==0: print...' """
    nodesD = locRun.nodesColl.all
    varNumList = locRun.varNums
    varNameList = locRun.varNames

    gateSum = {}  # Make a dictionary to hold the count of how many nodes for each variable have which gate code values
    for iVar in varNumList:  # Loop over all vars
        gateSum[iVar] = {}  # Initialize dictionary to hold counts for each gate-status value
        layerNodes = nodesD[iVar]  # Dictionary of the nodes for the current variable
        for _ in range(4): gateSum[iVar][_] = 0  # Initialize the count for how many nodes are set to each gate value
        for iNode in list(layerNodes.values()):  # Loop over nodes in the layer
            gateCode = gatesStatus([iNode])
            gateSum[iVar][gateCode] += 1

    print('Number of nodes with each gate status for each variable after', len(locRun.playoutList), 'playouts')
    print("{0:>7s} {1:>30s} {2:>9s} {3:>9s} {4:>9s} {5:>9s} {6:>9s} {7:>9s}".
          format(' ', ' ', 'max', 'nodes', 'nodes w/', 'nodes w/', 'nodes w/', 'nodes w/'))
    print("{0:>7s} {1:>30s} {2:>9s} {3:>9s} {4:>9s} {5:>9s} {6:>9s} {7:>9s}".
          format('varNum', 'varName', 'nodes', 'in lyr', 'status=3', 'status=2', 'status=1', 'status=0'))
    for i in varNumList:
        print("{0:>7d} {1:>30s} {2:>9d} {3:>9d} {4:>9d} {5:>9d} {6:>9d} {7:>9d}"
              .format(i, varNameList[i], 2 ** i, len(list(nodesD[i].values())),
                      gateSum[i][3], gateSum[i][2], gateSum[i][1], gateSum[i][0]))
    return


def banditCompareSides(locRun, locCPParm, locBanditMode, locBanditThreshCut):
    """Make a table, for each variable, that shows the distribution of bandit score differences between including
    and excluding the variable.  CPParm and Bandit mode can change from one playout to the next, so you need
    to specify the values for which you want the plots.  The results will give you a sense of how
    likely different variables are to be included/excluded if you launched playouts with the given values.  Note
    that you only get results for nodes that have both gates open, since it just obscures the results if you include
    nodes that, because of gate their status, can't visit both sides anyway."""
    nodesD = locRun.nodesColl.all  # nodesD is the nested dictionary of all nodes.
    varNumList = locRun.varNums
    banditDiffs = {}  # Make a dictionary to hold a list for each variable of its bandit score differences.
    for iVar in varNumList:  # Loop over all vars
        layerNodes = nodesD[iVar]  # layerNodes will be a dictionary of the nodes for the current variable
        banditDiffs[iVar] = []  # Initialize a list for bandit diffs for this variable
        for iNode in list(layerNodes.values()):  # Loop over nodes in the layer
            if not iNode.nodeReachableFromRoot(): continue  # Only look at reachable nodes (i.e., not blocked by gates)
            if iNode.incPathOpen and iNode.excPathOpen and iNode.nVisitsIncPath > 0 and iNode.nVisitsExcPath > 0:
                banditScoreInc, banditScoreExc = banditScore(iNode, locCPParm, locBanditMode, locBanditThreshCut)
                deltaBanditScore = banditScoreInc - banditScoreExc
                banditDiffs[iVar].append(deltaBanditScore)
            else:
                banditDiffs[iVar].append(-999)

    print("Pct of nodes in each layer that will include, exclude, flip-a-coin, or don't have both sides visited.")
    print("Table assumes: cpParm={0:5.3f}, bandit mode={1:8s}, banditThreshCut={2:8.3f}".
          format(locCPParm, locBanditMode, locBanditThreshCut))
    print("{0:>11s} {1:>11s} {2:>11s} {3:>11s} {4:>11s} {5:>11s}".
          format("varNum", "# of nodes", "pct inc", "pct exc", "pct flip", "pct notEval"))
    for iVar in varNumList:  # Loop over all vars and print fraction over zero
        nPos = len([_ for _ in banditDiffs[iVar] if _ > 0])
        nNeg = len([_ for _ in banditDiffs[iVar] if 0 > _ > -999])
        nZer = len([_ for _ in banditDiffs[iVar] if _ == 0])
        nExcept = len([_ for _ in banditDiffs[iVar] if _ == -999])
        nTot = nPos + nNeg + nZer + nExcept
        print("{0:>11d} {1:>11d} {2:>11.2f} {3:>11.2f} {4:>11.2f} {5:>11.2f}".
              format(iVar, nTot, 100.0*nPos/nTot, 100.0*nNeg/nTot, 100.0*nZer/nTot,  100.0*nExcept/nTot))

    # Histograms of this info seems less useful than just a table, so next section is commented out for now.
    # for iVar in varNumList:  # Loop over all vars and plot histograms
    #     plt.figure()
    #     plt.title('Delta Bandit Score for var ' + str(varNumList[iVar]) + ': ' + varNameList[iVar])
    #     plt.hist(banditDiffs[iVar], bins='auto', label=str(len(banditDiffs[iVar])))
    #     plt.legend(loc='upper right')
    #     plt.show()
    #     plt.close()
    return


def playoutParmsReport(locRun):
    """Reports a summary of the parameters used in the playouts.  Rather that listing them all
    for every playout, for parms that change infrequently we just list all different values
    used for all playouts that exist in the run.  For parms that change more frequently, we print a row
    every time any one of the parms changes, and a range of playout numbers where those were the values used.
    One can also execute a call to query the playout list for the run for more detailed information, as all parms
    are saved for every playout."""
    # First deal with parms that are unlikely to change even over the entire range of playouts.  In case there is
    # more than one value that was used, just print them all.
    playList = locRun.playoutList
    print("List of all parameters used for parms that change infrequently.  Individual playouts can also be queried")
    print("nEvents:", list(set([_.nEvents for _ in playList])))
    print("projectedGateDecision:", list(set([_.projectedGateDecision for _ in playList])))
    print("percentDefinition:", list(set([_.pctDefn for _ in playList])))
    print("exlusiveEventsOnly:", list(set([_.exclusiveEventsOnly for _ in playList])))
    print("sampleRandomly:", list(set([_.sampleRandomly for _ in playList])))
    print("banditMode:", list(set([_.banditMode for _ in playList])))
    print("gateMode:", list(set([_.gateMode for _ in playList])))
    print("ksgK:", list(set([_.ksgK for _ in playList])))
    print("nVarsMax:", list(set([_.nVarsMax for _ in playList])))

    # Now print out parms that change more frequently.  Report one line every time one of them changes.
    parmsList = []
    for i, currPlayout in enumerate(playList):
        pNumStr = str(i)
        ptype = currPlayout.pathType[0]
        if ptype == -2:
            ptypeStr = str(-2)
        elif ptype == -1:
            ptypeStr = str(-1)
        else:
            ptypeStr = 'fixed'
        cpParmStr = str(currPlayout.cpParm)
        banditThreshCutStr = str(currPlayout.banditThreshCut)
        gateThreshCutStr = str(currPlayout.gateThreshCut)
        gateIncCutStr = str(currPlayout.gateIncCut)
        gateExcCutStr = str(currPlayout.gateExcCut)

        currParms = [pNumStr, ptypeStr, cpParmStr, banditThreshCutStr, gateThreshCutStr, gateIncCutStr, gateExcCutStr]
        if i == 0: parmsList.append(currParms)  # Get first entry in the list
        lastEntry = parmsList[-1]
        # Add parms only if not repeat of prev entry (of course, you don't count having different playout number as new)
        if currParms[1:7] != lastEntry[1:7]: parmsList.append(currParms)

    # Now print out the parameters.  First column header, then one line whenever any parm has changed.
    print("{:^15s} {:^15s} {:^10s} {:>5s} {:^15s} {:^12s} {:^10s} {:^10s}".
          format("playouts", "# of playouts", "path type", "cpParm",
                 "bandit thresh", "gate thresh", "gateIncCut", "gateExcCut"))
    for i, entry in enumerate(parmsList):
        # Get the number of playouts that used the current set of parameters
        if i < len(parmsList)-1:
            nPlays = str(int(parmsList[i+1][0]) - int(entry[0]))
        else:
            nPlays = str(len(locRun.playoutList) - int(entry[0]))
        print("{:^7s}-{:^7s} {:^15s} {:^10s} {:>5s} {:^15s} {:^12s} {:^10s} {:^10s}".
              format(entry[0], str(int(entry[0])+int(nPlays)-1), nPlays,
                     entry[1], entry[2], entry[3], entry[4], entry[5], entry[6]))


def playoutsReport(locRun, numToReport, sortByScore=False, includeGateInfo=True, nVarRestrict=0):
    """Print a summary of playouts, possibly sorted by score.  The number of playouts to report can be set with
     numToReport.  If numToReport=0, then all playouts are reported."""
    if sortByScore:
        myList = sorted(locRun.playoutList, key=lambda element: element.score, reverse=True)
    else:
        myList = locRun.playoutList

    if abs(numToReport) > len(myList) or numToReport == 0:  # report all playouts
        start = 0
        stop = len(myList)
        step = 1
    elif numToReport < 0:  # only report the last n (w/ n=numToReport)
        start = -1
        stop = numToReport-1  # the -1 prevents off by one bug, since end-of-list counting starts from -1, not 0.
        step = -1
    else:  # report the first n (w/ n=numToReport)
        start = 0
        stop = numToReport
        step = 1

    if includeGateInfo:
        print("{0:>4s} {1:>32s} {2:>32s} {3:>6s} {4:>6s} {5:>10s} {6:>8s} {7:>7s}".
              format(" ", "gates status for each var", "inc/exc status for each var", "play", " ", "number", " ",
                     " "))
        print("{0:>4s} {1:>32s} {2:>32s} {3:>6s} {4:>6s} {5:>10s} {6:>8s} {7:>7s}".
              format("#", "10987654321098765432109876543210", "10987654321098765432109876543210", "type", "cpParm",
                     "Paths Open", "nEvents", "score"))
    else:
        print("{0:>4s} {1:>32s} {2:>6s} {3:>7s} {4:>9s} {5:>8s}".
              format(" ", "inc/exc status for each var", "play", " ", " ",
                     " "))
        print("{0:>4s} {1:>32s} {2:>6s} {3:>7s} {4:>9s} {5:>8s}".
              format("#", "10987654321098765432109876543210", "type", "cpParm", "nEvents", "score"))

    for i in range(start, stop, step):
        currPlayout = myList[i]

        if nVarRestrict != 0 and len(currPlayout.path.incVars) != nVarRestrict: continue
        ptype = currPlayout.pathType[0]
        if ptype == -2:
            ptypeStr = str(-2)
        elif ptype == -1:
            ptypeStr = str(-1)
        else:
            ptypeStr = 'fixed'
        cpParm = currPlayout.cpParm
        codedGateStatus = np.base_repr(gatesStatus(currPlayout.nodesList), 4)  # represent as string in base 4.
        # the npaths open value only makes sense for mode = -1
        nPathsOpenToReport = currPlayout.pathsCountD['nOpen'] if ptypeStr != 'fixed' else 1
        if includeGateInfo:
            print("{0:>4d} {1:>32s} {2:>32s} {3:>6s} {4:>6.3f} {5:>10.3e} {6:>8d} {7:>7.3f}".
                  format(i, codedGateStatus, currPlayout.path.pathId()[0], ptypeStr, cpParm,
                         nPathsOpenToReport, currPlayout.nEvents, currPlayout.score))
        else:
            print("{0:>4d} {1:>32s} {2:>6s} {3:>7.3f} {4:>9d} {5:>8.3f}".
                  format(i, currPlayout.path.pathId()[0], ptypeStr, cpParm,
                         currPlayout.nEvents, currPlayout.score))
    return


def pathsCountReport(locRun):
    """Print a summary of path counts (tested, open, closed, etc. etc.) as playouts proceeded, with one line
    per playout."""
    print("N.B. The counts for tested paths are invalid if you have reset the statistics for the run's nodes")
    print("{:>4s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s} {:>11s}".
          format("#", "Tested", "Untested", "Open", "TstdOpn", "UntstdOpn", "Closed", "TstdClsd", "UntstClsd"))
    for i in range(len(locRun.playoutList)):
        curD = locRun.playoutList[i].pathsCountD
        print("{:>4d}   {:.3e}   {:.3e}   {:.3e}   {:.3e}   {:.3e}   {:.3e}   {:.3e}   {:.3e}".
              format(i, curD['nTested'], curD['nUntested'], curD['nOpen'], curD['nTestedOpen'], curD['nUntestedOpen'],
                     curD['nClosed'], curD['nTestedClosed'], curD['nUntestedClosed']))
    # Print the header again at the end for easier viewing of text output
    print("{:>4s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s} {:>11s}".
          format("#", "Tested", "Untested", "Open", "TstdOpn", "UntstdOpn", "Closed", "TstdClsd", "UntstClsd"))


def plotReports(locRun, pdfFileName=None, includeGateInfo=True):
    """Function to call all the other main plot-based report functions"""

    # Make plot of mean of score of path taken vs playout number (i.e., vs. time)
    # Also make plot of number of paths still open vs playout number (i.e., vs. time)
    meanResults, nPathsResults = pathTracker(locRun)
    x = [i for i in range(len(meanResults))]
    y1 = [meanResults[i] for i in range(len(meanResults))]
    y2 = sorted(y1)
    y3 = [math.log2(nPathsResults[i]) for i in range(len(nPathsResults))]

    pdf = None
    if pdfFileName is not None: pdf = PdfPages(pdfFileName)  # Open file for saving plots

    figA1 = plt.figure()
    plt.title('mean path score vs playout number')
    plt.scatter(x, y1)
    plt.show(block=False)

    figA1b = plt.figure()
    plt.title('sorted mean path scores')
    plt.scatter(x, y2)
    plt.show(block=False)

    if pdfFileName is not None:
        pdf.savefig(figA1)
        pdf.savefig(figA1b)
    plt.close(figA1)
    plt.close(figA1b)

    if includeGateInfo:
        figA2 = plt.figure()
        plt.title('Log (base 2) of paths open vs playout number')
        plt.scatter(x, y3)
        plt.show(block=False)
        if pdfFileName is not None:
            pdf.savefig(figA2)  # Save number of paths remaining vs playout number
        plt.close(figA2)

    # Make plot of mean of gate status for each variable vs playout number (i.e., vs. time) and also
    # sig/bkd overlay plots for each variable
    if includeGateInfo:
        for i in range(len(locRun.dataset.varNums)):
            figGateCodes = plotGateTracker(locRun, i)
            plt.show(block=False)
            if pdfFileName is not None: pdf.savefig(figGateCodes)
            plt.close(figGateCodes)

    for i in range(len(locRun.dataset.varNums)):
        figV = plotVarCompare(locRun.dataset, i)
        plt.show(block=False)
        if pdfFileName is not None: pdf.savefig(figV)
        plt.close(figV)

    if pdfFileName is not None: pdf.close()  # close pdf file

    return


def pathTracker(locRun):
    """Makes lists of quantities for all playouts.  So, for example, it
    returns the number of paths that remained open as a function of
    playout number.  Typically this would be used to make a plot showing
    how some quantity evolved during the run over many playouts."""
    # Note that as currently written (from my rereading of it long after it was written), this routine doesn't seem to
    # have the flexibility claimed in the docstring.  It tracks mean score and pathsopen only.  The docstring
    # incorrectly makes it sound like you can pass an argument to tell it what to track.
    curDict = locRun.pathStatsColl.pathStatsDict
    meanResults = []
    nPathsResults = []

    for i in locRun.playoutList:
        curPath = i.path

        curPathIdStr = curPath.pathId()[0]
        curStats = curDict.get(curPathIdStr)
        meanScore = curStats.mean
        meanResults.append(meanScore)
        nPathsResults.append(i.pathsCountD['nOpen'])
    return meanResults, nPathsResults


def plotGateTracker(locRun, varNum):
    """plot of gate status vs playout for a given variable"""
    statusList = []
    for i in locRun.playoutList:
        statusString = np.base_repr(gatesStatus(i.nodesList), 4)  # represent as string in base 4.
        currStatus = statusString[
            -1 * (varNum + 1)]  # working our way backwards through the string from -1 to -(varnum+1)
        statusList.append(int(currStatus))  # convert the status character (0,1,2,3) for the gate back to an int.

    x = [i for i in range(len(locRun.playoutList))]
    y = statusList
    myFig = plt.figure()
    plt.title('Gate status vs playout for var ' + str(varNum) + ': ' + locRun.dataset.varNames[varNum])
    plt.ylim(0.0, 3.1)
    plt.scatter(x, y)
    return myFig


def plotVarCompare(locDataset, varNum, okRange=None):
    """Overlays sig and bkd hists for a given varnum in a given dataset.
    Optional argument to specify only including certain range of values.  This
    is so one can exclude markers like -999 from the plots"""

    if okRange is None:
        sig = [row[varNum] for row in locDataset.sigVars]
        bkd = [row[varNum] for row in locDataset.bkdVars]
    else:
        sig = [row[varNum] for row in locDataset.sigVars if (okRange[0] < row[varNum] < okRange[1])]
        bkd = [row[varNum] for row in locDataset.bkdVars if (okRange[0] < row[varNum] < okRange[1])]

    myFig = plt.figure()
    plt.title(str(varNum) + ': ' + locDataset.varNames[varNum])
    # plt.hist(sig, bins='auto', alpha=0.5, label='sig')
    # plt.hist(bkd, bins='auto', alpha=0.5, label='bkd')
    plt.hist([sig, bkd], bins=50, alpha=0.5, density=True, label=['sig', 'bkd'])
    plt.legend(loc='upper right')
    return myFig


def varAcceptRejectStatus(locRun):
    """Given a run object, this function looks at the summary gate statuses for each variable and reports
     which variables are allowed, which are required, and which are not allowed for inclusion in paths.  Note the
     difference between 'allowed', and 'optional'.  Allowed is a sum of optional and required."""
    allowedVars = [i for i, locNode in enumerate(locRun.nodesColl.smry.values()) if locNode.incPathOpen]
    requiredVars = [i for i, locNode in enumerate(locRun.nodesColl.smry.values()) if not locNode.excPathOpen]
    optionalVars = \
        [i for i, locNode in enumerate(locRun.nodesColl.smry.values()) if locNode.incPathOpen and locNode.excPathOpen]
    rejectedVars = [i for i, locNode in enumerate(locRun.nodesColl.smry.values()) if not locNode.incPathOpen]
    return allowedVars, requiredVars, optionalVars, rejectedVars
