'''
SequenceViz.py

Visualizes sequential Data as a plot of colored bars of the estimated state
as calculated by the Viterbi algorithm.  Displays a separate grid for each
jobname specified where each row corresponds to a sequence and each column
to a taskid.  By default, the final lap from each taskid is used; however,
a different lap can be specified with the --lap parameter.

Note that this script requires you
to have made a run of bnpy with the custom func learnalg/extras/XViterbi.py
running (use --customFuncPath path/goes/here/XViterbi.py)


Example, showing the final lap:

python SequenceViz.py --Dataset MoCap6 --jobnames defaultjob --taskids 1,3,6
                        --sequences 1,2,4,6

Example, showing the 5th lap for two different jobnames:

python SequenceViz.py --Dataset MoCap6 --jobnames defaultjob,EM --taskids 1
                        --lap 5 --sequences 1,2,4,6
'''
from builtins import *
import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import imp
import sys
import argparse

from matplotlib import pylab

from bnpy.util.StateSeqUtil import convertStateSeq_MAT2list
from bnpy.util.StateSeqUtil import convertStateSeq_list2flat
from bnpy.util.StateSeqUtil import convertStateSeq_flat2list
from bnpy.ioutil import BNPYArgParser
from bnpy.viz.TaskRanker import rankTasksForSingleJobOnDisk
from bnpy.viz.PlotTrace import taskidsHelpMsg


def plotSingleJob(dataset, jobname, taskids='1', lap='final',
                  sequences=[1],
                  showELBOInTitle=False,
                  dispTrue=True,
                  aspectFactor=4.0,
                  specialStateIDs=None,
                  seqNames=None,
                  cmap='Set1',
                  maxT=None,
                  colorManyToOne=False,
                  ):
    '''
    Returns the array of Data corresponding to a single sequence to display

    If dispTrue = True, the true labels will be shown underneath the
      estimated labels
    '''
    # Make sequences zero-indexed
    if isinstance(sequences, str):
        sequences = np.asarray([int(x) for x in args.sequences.split(',')],
                               dtype=np.int32)
    sequences = np.asarray(sequences, dtype=np.int32)
    if np.min(sequences) < 1:
        raise ValueError('Sequences need to be one-index.\n' +
                         'Valid values are 1,2,...N.')
    sequences -= 1

    # Determine the jobpath and taskids
    jobpath = os.path.join(os.path.expandvars('$BNPYOUTDIR'),
                           dataset, jobname)
    if isinstance(taskids, str):
        if taskids.startswith('.'):
            taskids = [taskids]
        else:
            taskids = BNPYArgParser.parse_task_ids(jobpath, taskids)
    elif isinstance(taskids, int):
        taskids = [str(taskids)]

    datasetPrefFile = os.path.join(
        jobpath, taskids[0],
        'args-DatasetPrefs.txt')
    datasetPrefs = dict()
    if os.path.exists(datasetPrefFile):
        with open(datasetPrefFile, 'r') as f:
            for line in f.readlines():
                fields = line.strip().split(' ')
                if len(fields) != 2:
                    continue
                datasetPrefs[fields[0]] = fields[1]

    # Load Data from its python module
    Datamod = imp.load_source(
        dataset,
        os.path.expandvars('$BNPYDATADIR/' + dataset + '.py'))
    if dataset == 'SpeakerDiar':
        if len(sequences) > 1:
            raise ValueError(
                'Joint modeling of several sequences makes no sense')
        Data = Datamod.get_data(meetingNum=sequences[0] + 1,
                                **datasetPrefs)
        jobpath = jobpath.replace(
            'SpeakerDiar', 'SpeakerDiar' + str(sequences[0] + 1))
        sequences[0] = 0

    else:
        Data = Datamod.get_data(**datasetPrefs)

    # Determine the maximum length among any of the sequences to be plotted
    if maxT is None:
        Ts = Data.doc_range[sequences + 1] - Data.doc_range[sequences]
        maxT = np.max(Ts)

    # Define the number of pixels used by vertical space of figure
    NUM_STACK = int(np.ceil(maxT / float(aspectFactor)))
    if dispTrue:
        NUM_STACK /= 2

    f, axes = plt.subplots(len(sequences), len(taskids),
                           sharex='col', sharey='row')

    # For singleton case, make sure that axes is index-able
    if len(sequences) == 1 and len(taskids) == 1:
        axes = [axes]

    for tt, taskidstr in enumerate(taskids):
        if tt == 0 and taskidstr.startswith('.'):
            rankTasksForSingleJobOnDisk(jobpath)

        path = os.path.join(jobpath, taskidstr) + os.path.sep

        # Figure out which lap to use
        if lap == 'final':
            lapsFile = open(path + 'laps-saved-params.txt')
            curLap = lapsFile.readlines()
            curLap = float(curLap[-1])
            lapsFile.close()
        else:
            curLap = int(lap)

        if showELBOInTitle:
            hdists = np.loadtxt(os.path.join(path, 'hamming-distance.txt'))
            hlaps = np.loadtxt(os.path.join(path, 'laps-saved-params.txt'))
            Keffvals = np.loadtxt(os.path.join(path, 'Keff-saved-params.txt'))
            # Determine scalar values to display
            loc = np.argmin(np.abs(hlaps - curLap))
            hdist = hdists[loc]
            Kefffinal = Keffvals[loc]

            try:
                Kvals = np.loadtxt(os.path.join(path, 'K.txt'))
                ELBOscores = np.loadtxt(os.path.join(path, 'evidence.txt'))
                laps = np.loadtxt(os.path.join(path, 'laps.txt'))

                loc = np.argmin(np.abs(laps - curLap))
                ELBO = ELBOscores[loc]
                Kfinal = Kvals[loc]
            except IOError:
                ELBO = 0.0
                Kfinal = Kefffinal

        # Load in the saved Data from $BNPYOUTDIR
        try:
            filename = 'Lap%08.3fMAPStateSeqsAligned.mat' % curLap
            zHatBySeq = scipy.io.loadmat(path + filename)
            key1 = 'zHatBySeqAligned'
            key2 = 'zHatBySeq'
            if key1 in zHatBySeq:
                zHatBySeq = convertStateSeq_MAT2list(zHatBySeq[key1])
            elif key2 in zHatBySeq:
                zHatBySeq = convertStateSeq_MAT2list(zHatBySeq[key2])
            else:
                raise IOError
        except IOError:
            filename = 'Lap%08.3fMAPStateSeqs.mat' % curLap
            zHatBySeq = scipy.io.loadmat(path + filename)
            zHatBySeq = convertStateSeq_MAT2list(zHatBySeq['zHatBySeq'])

        if specialStateIDs is not None:
            zHatBySeq = relabelAllSequences(zHatBySeq, specialStateIDs)

        # Find maximum number of states we need to display
        nSeq = len(zHatBySeq)
        Kmax = np.max([zHatBySeq[i].max() for i in range(nSeq)])
        hasGroundTruth = False

        vmin = 0
        Kignore = 0
        if hasattr(Data, 'TrueParams') and 'Z' in Data.TrueParams:
            hasGroundTruth = True
            Kmax = np.maximum(Data.TrueParams['Z'].max(), Kmax)
            uLabels = np.unique(Data.TrueParams['Z'])
            Kignore = np.sum(uLabels < 0)
            if Kignore > 0:
                for k in range(1, Kignore + 1):
                    print('ignoring state %d  Ttrue = %d' % (
                        -k, np.sum(Data.TrueParams['Z'] == -k)))

            if colorManyToOne:
                # For each state in zHat, find best true sequence
                Zflat = convertStateSeq_list2flat(zHatBySeq, Data)
                ZflatA = -1 * np.ones_like(Zflat)
                for uID in np.unique(Zflat):
                    overlap = np.zeros(uLabels.size)
                    for ii, trueID in enumerate(uLabels):
                        overlap[ii] = np.sum(np.logical_and(
                            Data.TrueParams['Z'] == trueID,
                            Zflat == uID))
                    bestii = overlap.argmax()
                    ZflatA[Zflat == uID] = uLabels[bestii]
                zHatBySeq = convertStateSeq_flat2list(ZflatA, Data)

        # In case there's only one sequence, make sure it's index-able
        for ii, seqNum in enumerate(sequences):
            image = np.tile(zHatBySeq[seqNum], (NUM_STACK, 1))

            # Add the true labels to the image (if they exist)
            if hasGroundTruth and dispTrue:
                start = Data.doc_range[seqNum]
                stop = Data.doc_range[seqNum + 1]
                img_trueZ = np.tile(Data.TrueParams['Z'][start:stop],
                                    (NUM_STACK, 1))
                if dispTrue == 2:
                    image = img_trueZ  # Show only true labels
                else:
                    image = np.vstack((image, img_trueZ))

            image = image[:, :maxT]
            if len(sequences) == 1 or len(taskids) == 1:
                cur_ax = axes[ii + tt]
            else:
                cur_ax = axes[ii, tt]

            if hasattr(cmap, 'N'):
                vmax = cmap.N
            else:
                vmax = Kmax

            cur_ax.imshow(Kignore + image + .0001, interpolation='nearest',
                          vmin=vmin, vmax=vmax,
                          cmap=cmap)
            if tt == 0:
                if seqNames is not None:
                    h = cur_ax.set_ylabel('%s' % (seqNames[ii]), fontsize=13)
                    h.set_rotation(0)

                elif len(sequences) > 4:
                    cur_ax.set_ylabel('%d' % (seqNum + 1), fontsize=13)
                else:
                    cur_ax.set_ylabel('Seq. %d' % (seqNum + 1), fontsize=13)

            if ii == 0:
                if showELBOInTitle:
                    fmtSpec = "ELBO: %.3f  K=%d Keff=%d  "
                    if hdist > 0.01:
                        fmtSpec += "dist=%.2f"
                    elif hdist > 0.001:
                        fmtSpec += "dist=%.3f"
                    else:
                        fmtSpec += "dist=%.4f"
                    title = fmtSpec % (ELBO, Kfinal, Kefffinal, hdist)
                    cur_ax.set_title(title)

            cur_ax.set_xlim([0, maxT])
            cur_ax.set_ylim([0, image.shape[0]])
            cur_ax.set_yticks([])
            # ... end loop over sequences
    return axes, zHatBySeq


def relabelAllSequences(zBySeq, specialStateIDs):
    ''' Relabel all sequences in provided list.

    Returns
    -------
    zBySeq, relabelled so that each label in specialStateIDs
            now corresponds to ids 0, 1, 2, ... L-1
            and all other labels not in that set get ids L, L+1, ...
    '''
    import copy
    zBySeq = copy.deepcopy(zBySeq)
    L = len(specialStateIDs)

    uniqueVals = []
    for z in zBySeq:
        z += 1000
        for kID, kVal in enumerate(specialStateIDs):
            z[z == 1000 + kVal] = -1000 + kID
            uniqueVals = np.union1d(uniqueVals, np.unique(z))

    for z in zBySeq:
        for kID, kVal in enumerate(sorted(uniqueVals)):
            z[z == kVal] = kID

    return zBySeq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--jobnames')
    parser.add_argument('--taskids', type=str, default='1',
                        help=taskidsHelpMsg)

    parser.add_argument('--lap', default='final')
    parser.add_argument('--sequences', default='1')
    args = parser.parse_args()

    if args.jobnames is None:
        raise ValueError('BAD ARGUMENT: String jobname.')
    jobs = args.jobnames.split(',')

    for job in jobs:
        plotSingleJob(dataset=args.dataset,
                      jobname=job,
                      taskids=args.taskids,
                      lap=args.lap,
                      sequences=args.sequences,
                      dispTrue=True)

    plt.show()
