'''
Monks.py

The dataset was gathered during a period of political turmoil in the cloister.
The true labels (nodeZ) reflect the "faction labels" of each monk:
    0 = Young Turks (rebel group)
    1 = Loyal Opposition (monks who followed tradition and remained loyal),
    2 = Outcasts (Monks who were not accepted by either faction),
    3 = Waverers (Monks who couldn't decide on a group).

Resources
---------
Data Source: http://moreno.ss.uci.edu/sampson.dat
'''

import numpy as np
import scipy.io
import os

from bnpy.data import GraphXData


# monkNames : order copied from the header of sampson.dat
monkNames = [
  'Ramuald',
  'Bonaventure',
  'Ambrose',
  'Berthold',
  'Peter',
  'Louis', 
  'Victor',
  'Winfrid',
  'John_Bosco',
  'Gregory',
  'Hugh',
  'Boniface',
  'Mark',
  'Albert',
  'Amand',
  'Basil',
  'Elias',
  'Simplicius',
  ]

relationLabels = [
  'like_phase1',
  'like_phase2',
  'like_phase3',
  'dislike_phase3',
  'esteem_phase3',
  'disesteem_phase3',
  'posinfluence_phase3',
  'neginfluence_phase3',
  'praise_phase3',
  'negpraise_phase3',
  ]

# Get path to the .mat file with the data
datasetdir = os.path.sep.join(
    os.path.abspath(__file__).split(os.path.sep)[:-1])
if not os.path.isdir(datasetdir):
    raise ValueError('Cannot find Monks dataset directory:\n' + datasetdir)

datfilepath = os.path.join(datasetdir, 'rawData', 'sampson.dat')
if not os.path.isfile(datfilepath):
    raise ValueError('Cannot find Monks dataset file:\n' + datfilepath)


def get_data(relationName='esteem', phase='3', **kwargs):
    DataLines = list()
    with open(datfilepath, 'r') as f:
        doRecord = 0
        for line in f.readlines():
            line = line.strip()
            if doRecord:
                DataLines.append(np.asarray(
                    line.split(' '), dtype=np.int32)>0)
            if line.startswith('DATA:'):
                doRecord = 1
    AdjMatStack = np.vstack(DataLines)
    AdjMatStack[AdjMatStack > 0] = 1
    # AdjMatStack is 180 x 18, where each set of 18 rows
    # corresponds to one of the 10 relations.

    # Crop out set of 18 contig rows
    # specified by the relation keyword 
    matchID = -1
    matchrelLabel = relationName + '_phase' + str(phase)
    for relID, relLabel in enumerate(relationLabels):
      if relLabel == matchrelLabel:
          matchID = relID
          break
    if matchID < 0:
        raise ValueError(
            "Cannot find desired relation: %s" % matchrelLabel)
    AdjMat = AdjMatStack[matchID*18:(matchID+1)*18]

    MonkNameToIDMap = dict()
    for uid, name in enumerate(monkNames):
        MonkNameToIDMap[name] = uid

    MonkIDToLabelIDMap = dict()
    labelfilepath = datfilepath.replace('sampson.dat', 'sampson_labels.txt')
    with open(labelfilepath, 'r') as f:
        header = f.readline()
        LabelNames = header.strip().split()
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                break
            # "John_Bosco 1" >> "John_Bosco", "1"
            keyval = line.split(' ')
            name = keyval[0]
            labelID = int(keyval[1])
            monkID = MonkNameToIDMap[name]
            MonkIDToLabelIDMap[monkID] = labelID
    nodeZ = np.asarray([
        MonkIDToLabelIDMap[MonkNameToIDMap[mName]]
        for mName in monkNames], dtype=np.int32)
        
    Data = GraphXData(AdjMat=AdjMat,
        nodeNames=monkNames, nodeZ=nodeZ)
    Data.summary = get_data_info()
    Data.name = get_short_name()
    Data.relationName = matchrelLabel
    return Data


def get_data_info():
    return 'Sampson Monks dataset'


def get_short_name():
    return 'Monks'

if __name__ == '__main__':
    from matplotlib import pylab;

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--relationName', default='esteem', type=str)
    parser.add_argument('--phase', default='3', type=str)
    args = parser.parse_args()

    print("Loading Sampson Monks dataset with relationName=%s  at phase=%s" % (
        args.relationName, args.phase))
    # Fetch data and plot the adjacency matrix
    Data = get_data(relationName=args.relationName, phase=args.phase)
    Xdisp = np.squeeze(Data.toAdjacencyMatrix())
    sortids = np.argsort(Data.TrueParams['nodeZ'])
    Xdisp = Xdisp[sortids, :]
    Xdisp = Xdisp[:, sortids]
    nodeNames = [Data.nodeNames[s] for s in sortids]

    pylab.imshow(
        Xdisp, cmap='Greys', interpolation='nearest',
        vmin=0, vmax=1)
    pylab.gca().set_yticks(np.arange(len(nodeNames)))
    pylab.gca().set_yticklabels(nodeNames)
    pylab.title('Adj. matrix: %s' % (Data.relationName))
    pylab.ylabel('Monks (sorted by true label)')

    pylab.show()

