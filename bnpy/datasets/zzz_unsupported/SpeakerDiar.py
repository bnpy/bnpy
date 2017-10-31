'''
SpeakDiar.py

21 audio recordings of academic conferences making up the NIST speaker
diarization dataset, created to asses the ability of different models to
segment speech data into unique speakers.

The 21 recordings are meant to be trained on independently.
Thus, get_data() takes a meetingNum parameter (default 1)
which determines which sequence will be loaded.
The meeting number can be changed with the argument
  --meetingNum 3

Notes
-----
rttm format specification:
http://www.itl.nist.gov/iad/mig/tests/rt/2003-fall/docs/RTTM-format-v13.pdf
'''

import numpy as np
from bnpy.data import GroupXData
import scipy.io
import os
import sys

suffix = '_Nmeans25features_SpNsp'

fileNames = [
    'AMI_20041210-1052_Nmeans25features_SpNsp.mat',
    'AMI_20050204-1206_Nmeans25features_SpNsp.mat',
    'CMU_20050228-1615_Nmeans25features_SpNsp.mat',
    'CMU_20050301-1415_Nmeans25features_SpNsp.mat',
    'CMU_20050912-0900_Nmeans25features_SpNsp.mat',
    'CMU_20050914-0900_Nmeans25features_SpNsp.mat',
    'EDI_20050216-1051_Nmeans25features_SpNsp.mat',
    'EDI_20050218-0900_Nmeans25features_SpNsp.mat',
    'ICSI_20000807-1000_Nmeans25features_SpNsp.mat',
    'ICSI_20010208-1430_Nmeans25features_SpNsp.mat',
    'LDC_20011116-1400_Nmeans25features_SpNsp.mat',
    'LDC_20011116-1500_Nmeans25features_SpNsp.mat',
    'NIST_20030623-1409_Nmeans25features_SpNsp.mat',
    'NIST_20030925-1517_Nmeans25features_SpNsp.mat',
    'NIST_20051024-0930_Nmeans25features_SpNsp.mat',
    'NIST_20051102-1323_Nmeans25features_SpNsp.mat',
    'TNO_20041103-1130_Nmeans25features_SpNsp.mat',
    'VT_20050304-1300_Nmeans25features_SpNsp.mat',
    'VT_20050318-1430_Nmeans25features_SpNsp.mat',
    'VT_20050623-1400_Nmeans25features_SpNsp.mat',
    'VT_20051027-1400_Nmeans25features_SpNsp.mat']


datasetdir = os.path.sep.join(
    os.path.abspath(__file__).split(
        os.path.sep)[
            :-
            1])
if not os.path.isdir(datasetdir):
    raise ValueError('CANNOT FIND DATASET DIRECTORY:\n' + datasetdir)


def get_data(meetingNum=1, **kwargs):
    ''' Load data for specified single sequence.

    Args
    ----
    meetingNum : int
        Identifies which sequence out of the 21 possible to use.
        Must be valid number in range [1,2,3, ... 21].

    Returns
    -------
    Data : GroupXData
        holding only the data for a single sequence.
    '''
    if meetingNum <= 0 or meetingNum > len(fileNames):
        raise ValueError('Bad value for meetingNum: %s' % (meetingNum))

    fName = fileNames[meetingNum - 1].replace(suffix, '')
    matfilepath = os.path.join(datasetdir, 'rawData',
                               'speakerDiarizationData', fName)

    if not os.path.isfile(matfilepath):
        raise ValueError(
            'CANNOT FIND SPEAKDIAR DATASET MAT FILE:\n' + matfilepath)

    Data = GroupXData.read_from_mat(matfilepath)
    Data.summary = \
        'Pre-processed audio data from NIST file %s (meeting %d / 21)' \
        % (fName.replace(suffix, ''), meetingNum)
    Data.name = 'SpeakerDiar' + str(meetingNum)

    Data.fileNames = [fName]
    return Data


def createBetterBNPYDatasetFromMATFiles():
    ''' Create new MAT files that relabel states.

    Post Condition
    --------------
    rawData directory contains files of the form:
        EDI_20050216-1051.mat
    '''
    for file in fileNames:
        matfilepath = os.path.join(os.path.expandvars(
            '$BNPYDATADIR/rawData/speakerDiarizationData'), file)
        SavedVars = scipy.io.loadmat(matfilepath)
        outmatpath = matfilepath.replace(suffix, '')
        print(file.replace(suffix, ''))
        SavedVars['TrueZ'] = \
            relabelStateSeqWithNegativeIDsForNonspeakerIntervals(
            SavedVars['TrueZ'])
        scipy.io.savemat(outmatpath, SavedVars)


def relabelStateSeqWithNegativeIDsForNonspeakerIntervals(Z):
    ''' Relabel provided Z sequence so nonspeaker intervals have neg. ids.

    Returns
    -------
    Znew : 1D array, size of Z
        Znew will have "speaker" states with ids 0, 1, 2, ... K-1
            where the ids are ordered from most to least common.
        and non-speaker states with ids -1 (for silence) and -2 (for overlap)
    '''
    uLabels = np.unique(Z)
    uLabels = np.asarray([u for u in uLabels if u > 0 and u < 10])
    sizes = np.asarray([np.sum(Z == u) for u in uLabels])
    sortIDs = np.argsort(-1 * sizes)
    Znew = np.zeros_like(Z, dtype=np.int32)
    aggFrac = 0
    for rankID, uID in enumerate(uLabels[sortIDs]):
        Znew[Z == uID] = rankID
        size = np.sum(Z == uID)
        frac = size / float(Z.size)
        aggFrac += frac
        print('state %3d: %5d tsteps (%.3f, %.3f)' % (
            rankID, size, frac, aggFrac))
    Znew[Z == 0] = -1
    Znew[Z == 10] = -2
    for uID in [-1, -2]:
        size = np.sum(Znew == uID)
        frac = size / float(Z.size)
        aggFrac += frac
        print('state %3d: %5d tsteps (%.3f, %.3f)' % (
            uID, size, frac, aggFrac))
    assert np.allclose(1.0, aggFrac)
    return Znew


def createBNPYDatasetFromOriginalMATFiles(dataPath):
    for file in fileNames:
        fpath = os.path.join(dataPath, file)
        data = scipy.io.loadmat(fpath)
        X = np.transpose(data['u'])
        TrueZ = data['zsub']
        doc_range = [0, np.size(TrueZ)]
        matfilepath = os.path.join(os.path.expandvars(
            '$BNPYDATADIR/rawData/speakerDiarizationData'), file)
        SaveDict = {'X': X, 'TrueZ': TrueZ, 'doc_range': doc_range}
        scipy.io.savemat(matfilepath, SaveDict)


def plotXPairHistogram(meetingNum=1, dimIDs=[0, 1, 2, 3], numStatesToShow=3):
    from matplotlib import pylab
    Data = get_data(meetingNum=meetingNum)
    TrueZ = Data.TrueParams['Z']
    uniqueLabels = np.unique(TrueZ)
    sizeOfLabels = np.asarray(
        [np.sum(TrueZ == labelID) for labelID in uniqueLabels])
    sortIDs = np.argsort(-1 * sizeOfLabels)
    topLabelIDs = uniqueLabels[sortIDs[:numStatesToShow]]
    Colors = ['k', 'r', 'b', 'm', 'c']
    D = len(dimIDs)
    pylab.subplots(nrows=len(dimIDs), ncols=len(dimIDs))
    for id1, d1 in enumerate(dimIDs):
        for id2, d2 in enumerate(dimIDs):
            pylab.subplot(D, D, id2 + D * id1 + 1)
            if id1 == id2:
                pylab.xticks([])
                pylab.yticks([])
                continue
            pylab.hold('on')
            if id1 < id2:
                order = reversed([x for x in enumerate(topLabelIDs)])
            else:
                order = enumerate(topLabelIDs)
            cur_d1 = np.minimum(d1, d2)
            cur_d2 = np.maximum(d1, d2)
            for kID, labelID in order:
                dataIDs = TrueZ == labelID
                pylab.plot(Data.X[dataIDs, cur_d1],
                           Data.X[dataIDs, cur_d2], '.',
                           color=Colors[kID], markeredgecolor=Colors[kID])
            pylab.ylim([-25, 25])
            pylab.xlim([-25, 25])
            if (id2 > 0):
                pylab.yticks([])
            if (id1 < D - 1):
                pylab.xticks([])

    '''
    # make a color map of fixed colors
    from matplotlib import colors
    cmap = colors.ListedColormap(['white'] + Colors[:3])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    Z = np.zeros(Data.X.shape)
    for kID, labelID in enumerate(topLabelIDs):
        curIDs = TrueZ == labelID
        Z[curIDs, :] = bounds[kID + 1]
    pylab.subplots(nrows=1, ncols=2)
    ax = pylab.subplot(1, 2, 1)
    pylab.imshow(Z.T, interpolation='nearest',
                 cmap=cmap,
                 aspect=Z.shape[0] / float(Z.shape[1]),
                 vmin=bounds[0],
                 vmax=bounds[-1],
                 )
    pylab.yticks([])

    pylab.subplot(1, 2, 2, sharex=ax)
    for d in dimIDs:
        pylab.plot(np.arange(Z.shape[0]), 10 * d + Data.X[:, d] / 25, 'k.-')
    '''
    pylab.show()


def plotBlackWhiteStateSeqForMeeting(meetingNum=1, badUIDs=[-1, -2],
                                     **kwargs):
    ''' Make plot like in Fig. 3 of AOAS paper
    '''
    from matplotlib import pylab

    Data = get_data(meetingNum=args.meetingNum)
    Z = np.asarray(Data.TrueParams['Z'], dtype=np.int32)

    uLabels = np.unique(Z)
    uLabels = np.asarray([u for u in uLabels if u not in badUIDs])
    sizes = np.asarray([np.sum(Z == u) for u in uLabels])
    sortIDs = np.argsort(-1 * sizes)
    Zim = np.zeros((10, Z.size))
    for rankID, uID in enumerate(uLabels[sortIDs]):
        Zim[1 + rankID, Z == uID] = 1
        size = sizes[sortIDs[rankID]]
        frac = size / float(Z.size)
        print('state %3d: %5d tsteps (%.3f)' % (rankID + 1, size, frac))

    for uID in badUIDs:
        size = np.sum(Z == uID)
        frac = size / float(Z.size)
        print('state %3d: %5d tsteps (%.3f)' % (uID, size, frac))

    pylab.imshow(1 - Zim,
                 interpolation='nearest',
                 aspect=Zim.shape[1] / float(Zim.shape[0]) / 3,
                 cmap='bone',
                 vmin=0,
                 vmax=1,
                 origin='lower')
    pylab.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimIDs', default='0,1,2,3')
    parser.add_argument('--meetingNum', type=int, default=1)
    parser.add_argument('--numStatesToShow', type=int, default=4)
    args = parser.parse_args()
    args.dimIDs = [int(x) for x in args.dimIDs.split(',')]
    # plotBlackWhiteStateSeqForMeeting(**args.__dict__)
    plotXPairHistogram(**args.__dict__)
