
"""
Usage:
Various routines related to using CTree code.
"""
import CTreeMI as ct
import shelve
import pathlib
import dbm.dumb
import numpy as np
import csv


def statsKSG(locDataset, locPath, maxEvts, nEvtsPerPlayout):
    """Make rms estimate from using non-overlapping subsamples of a given dataset"""
    nPlayouts = int(maxEvts/nEvtsPerPlayout)  # max number of playouts limited by num tot events and num per playout
    locRun = ct.run(locDataset, nParallelParm=nPlayouts)  # create a run object to run playouts

    assert locPath[0] >= 0, "{0:45s} {1}".format("statsKSG must be called with a fixed path:", locPath[0])
    ct.playouts(locRun, eventsPerPlayout=nEvtsPerPlayout, pathType=locPath, numPlays=nPlayouts,
                exclusiveEventsOnly=True, sampleRandomly=True)
    firstPlayoutNum = 0
    lastPlayoutNum = nPlayouts - 1
    statsResults = playoutsMeanScore(locRun, firstPlayoutNum, lastPlayoutNum)
    return statsResults


def playoutsMeanScore(locRun, first, last):
    """Utility routine to return mean, stdev, and rms for a given range of playouts in a given run"""
    scores = [_.score for _ in locRun.playoutList[first:last+1]]
    mean, meanErr, rms = ct.meanSigmaFromList(scores, doFit=True)
    return mean, meanErr, rms


def banditBonusDif(nA, nB, cpParm):
    """Returns the difference in bonus in the bandit formula for a given number of visits
    to each side, and a given cpParm.  Useful for studying what value of cpParm makes sense
    for a problem."""
    nTot = nA + nB
    banditA = 2 * cpParm * np.sqrt(2 * np.log(nTot) / nA)
    banditB = 2 * cpParm * np.sqrt(2 * np.log(nTot) / nB)
    return banditA, banditB, banditA - banditB


def shelfRun(filename, mode, key=None, myRunIn=None):
    """Read and write run object (with all its playouts) to shelf file so that you can read them back
    later and continue playouts on the same run object without losing the status of running so far.
    Note that for simplicity, the routine currently only allows you to store one run object per key.
    The key is the name of the job (i.e. HiggsTest).  So, you can't store multiple objects under that
    same key name.  It always just stores the run object under the key myRun.

    filename: the name of the file on disk where you're storing the shelved information.

    mode: read, write, or delete.  Read/write run object from/to the shelf file, or delete run object from shelf file.

    key: the key under which the dictionary for this job should be read/written.  Typically, this will
    be the filename of the job you're running.

    In read mode, pass filename and key.  The returned object will be the filled run object.

    In write mode, pass filename, key, and the run object myRunIn. Whatever had been in the dictionary for this
    job's key will be replaced by the passed-in run object.  Return will be None.

    In delete mode, pass filename and key, but nothing for myRunIn.  Return will be None.
    """
    if myRunIn is None: myRunIn = {}
    myShelf = None

    # Check for path/file and create file if needed
    filenameExtended = filename+'.dir'  # there's also a .dat file, but just need one to confirm existence
    myPath = pathlib.Path(filenameExtended)
    okPath = myPath.exists()
    okFile = myPath.is_file()
    if okPath:  # we at least have a path
        # make sure it's a file, not just a path.  If it is, then you're ready to read/write.
        assert okFile, 'Error: The passed filename is a path not a file.'
    if mode == 'read':
        myDB = dbm.dumb.open(filename, flag='r')
        myShelf = shelve.Shelf(myDB)
    elif mode == 'write':
        myDB = dbm.dumb.open(filename, flag='n')
        myShelf = shelve.Shelf(myDB)
    elif mode == 'delete':
        myDB = dbm.dumb.open(filename, flag='w')
        myShelf = shelve.Shelf(myDB)
    else:
        print('shelfRun called with unknown mode.  No action taken')

    # In read mode, make sure that the shelf has the key that the user passed.
    # If it does, then fill the result object with the corresponding stored run object.
    if mode == 'read':
        if key in myShelf:
            result = myShelf[key]['myRun']
        else:
            print('Given key not found in file.  Now you must create a new run object before doing playouts')
            result = None
    # In write mode, make sure you don't have an empty run object before trying to write it to the shelf dictionary
    elif mode == 'write':
        if isinstance(myRunIn, ct.run):
            myShelf[key] = {'myRun': myRunIn}
        else:
            print('Can not write because user passed no run object to function.')
        result = None
    # In delete mode, make sure the dictionary has the key.  If it does, then delete it.
    elif mode == 'delete':
        if key in myShelf:
            del myShelf[key]
        else:
            print('Can not delete key because key not found in file.')
        result = None
    # Called with invalid mode.
    else:
        print('shelfRun called with unknown mode.  No action taken')
        result = None

    myShelf.close()
    return result


def variableMetric(nodeList):
    """Return a measure of how well a variable is performing"""
    threshPct = nodeList[0].run.playoutList[-1].gateThreshCut
    thresholdValue = nodeList[0].run.getThresholdScore(threshPct, 'percentile')
    resultsList = []
    for i, inode in enumerate(nodeList):
        fracOverThreshInc, errOnFracInc = inode.fracPathsOverThresh(True, thresholdValue)
        fracOverThreshExc, errOnFracExc = inode.fracPathsOverThresh(False, thresholdValue)
        sigDevInc = (fracOverThreshInc - threshPct/100)/errOnFracInc
        sigDevExc = (fracOverThreshExc - threshPct/100)/errOnFracExc

        sigDevInc = min(sigDevInc, 9.99)
        sigDevInc = max(sigDevInc, -9.99)
        sigDevExc = min(sigDevExc, 9.99)
        sigDevExc = max(sigDevExc, -9.99)

        resultsList.append(sigDevInc)
    return resultsList


def makeHiggsDataset(nEvents, normalize=True):
    """Build a dataset object in the format that CTreeMI expects."""

    sigVarsAll = []  # Has weights in it
    with open("storage/data/allevents_signal.csv", "r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            sigVarsAll.append([float(row[-_]) for _ in range(1, len(row) + 1)])

    bkdVarsAll = []  # Has weights in it
    with open("storage/data/allevents_background.csv", "r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            bkdVarsAll.append([float(row[-_]) for _ in range(1, len(row) + 1)])

    # Convert to numpy array
    sigA = np.array(sigVarsAll)
    bkdA = np.array(bkdVarsAll)

    # Choose random selection of events

    # If nEvents is zero, then make sig and bkd samples each equal to all events available in their respective set.
    # The CTreeMI code will later take equal-sized subsets from each according to the requested number of events per
    # playout.  But we will still be drawing from the full sample of events of both types.
    if nEvents == 0:  # take all events
        sigB = sigA
        bkdB = bkdA
    else:  # take requested subset
        sigB = sigA[np.random.choice(sigA.shape[0], nEvents, replace=False)]
        bkdB = bkdA[np.random.choice(bkdA.shape[0], nEvents, replace=False)]

    # Copy weights column
    # Weights were the last column in the file, but you read them in reverse
    sigWts = sigB[:, 0].tolist()
    bkdWts = bkdB[:, 0].tolist()
    # wts have to be list of vectors of length 1
    sigWts = [[_] for _ in sigWts]
    bkdWts = [[_] for _ in bkdWts]

    # remove the weights column from the variables array.
    sigC = np.delete(sigB, 0, 1)
    bkdC = np.delete(bkdB, 0, 1)

    # convert back to list
    sigVars = sigC.tolist()
    bkdVars = bkdC.tolist()

    # set name of dataset and numbers/names of variables
    setName = 'HiggsML Sample'
    varNums = [i for i in range(len(sigVars[0]))]
    # There are 30 variables.
    varNames = ['PRI_jet_all_pt', 'PRI_jet_subleading_phi', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_pt',
                'PRI_jet_leading_phi', 'PRI_jet_leading_eta', 'PRI_jet_leading_pt', 'PRI_jet_num', 'PRI_met_sumet',
                'PRI_met_phi', 'PRI_met', 'PRI_lep_phi', 'PRI_lep_eta', 'PRI_lep_pt', 'PRI_tau_phi', 'PRI_tau_eta',
                'PRI_tau_pt', 'DER_lep_eta_centrality', 'DER_met_phi_centrality', 'DER_pt_ratio_lep_tau',
                'DER_sum_pt', 'DER_pt_tot', 'DER_deltar_tau_lep', 'DER_prodeta_jet_jet', 'DER_mass_jet_jet',
                'DER_deltaeta_jet_jet', 'DER_pt_h', 'DER_mass_vis', 'DER_mass_transverse_met_lep', 'DER_mass_MMC']

    discVars = [7]

    myDataset = ct.dataset(setName, sigVars, bkdVars, sigWts, bkdWts, varNums, varNames, discVars=discVars,
                           normed=normalize, normExclusions=[-999])

    return myDataset
