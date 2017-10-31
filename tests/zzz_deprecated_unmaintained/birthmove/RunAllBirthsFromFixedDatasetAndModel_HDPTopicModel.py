import argparse
import numpy as np
import os
import sys
import bnpy
import time

from matplotlib import pylab;
from distutils.dir_util import mkpath
from bnpy.birthmove.BirthProposalError import BirthProposalError
from bnpy.birthmove.SCreateFromScratch import createSplitStats
from bnpy.birthmove.SAssignToExisting import assignSplitStats
from bnpy.birthmove.SCreateFromScratch import DefaultLPkwargs
from bnpy.viz.PlotUtil import ConfigPylabDefaults
from bnpy.viz.PrintTopics import printTopWordsFromWordCounts
from bnpy.viz.PrintTopics import plotCompsFromWordCounts, uidsAndCounts2strlist

ConfigPylabDefaults(pylab)

DefaultKwargs = dict(
    nDocTotal=500,
    nWordsPerDoc=400,
    Kinit=1,
    nDocInit=100,
    targetUID=0,
    doInteractiveViz=False,
    dataName='nips',
    Kfresh=10, 
    doShowAfter=1,
    outputdir='/tmp/',
    b_creationProposalName='BregDiv',
    b_includeRemainderTopic=1,
    b_nRefineSteps=3,
    b_initHardCluster=0,
    b_minNumAtomsInDoc=100,
    b_mergeLam=0.1,
    )

outdirPattern = 'targetUID=%d_initHardCluster=%d_' + \
                'includeRemainderTopic=%d_Kfresh=%d'

def main(**kwargs):
    DefaultKwargs.update(kwargs)

    parser = argparse.ArgumentParser()
    for key, val in list(DefaultKwargs.items()):
        try:
            assert np.allclose(int(val), float(val))
            _type = int
        except Exception as e:
            try:
                float(val)
                _type = float
            except:
                _type = str
        parser.add_argument('--' + key, default=val, type=_type)
    args = parser.parse_args()
    dataName = args.dataName

    # Define default kwargs for birth step
    bkwargs = dict()
    for key in args.__dict__:
        if key.startswith('b_'):
            bkwargs[key] = getattr(args, key)

    if dataName.count('BarsK10V900'):
        import BarsK10V900
        Data = BarsK10V900.get_data(
            nDocTotal=args.nDocTotal, nWordsPerDoc=args.nWordsPerDoc)
        Data.alwaysTrackTruth = 1
    else:   
        os.environ['BNPYDATADIR'] = os.path.join(
            os.environ['HOME'], 'git/x-topics/datasets/' + dataName + '/')
        sys.path.append(os.environ['BNPYDATADIR'])
        if dataName.count('nyt'):
            datafile = "/data/liv/nytimes/batches/batch001.ldac"
            Data = bnpy.data.BagOfWordsData.LoadFromFile_ldac(
                datafile, vocab_size=8000,
                vocabfile=os.path.join(os.environ['BNPYDATADIR'], 'vocab.txt'))
        else:
            dataMod = __import__(dataName, fromlist=[])
            Data = dataMod.get_data()

        vocabList = Data.vocabList
        nDoc = np.minimum(args.nDocTotal, Data.nDoc)
        if nDoc < Data.nDoc:
            Data = Data.select_subset_by_mask(list(range(nDoc)))
        Data.name = dataName
        Data.vocabList = vocabList

    nDocInit = np.minimum(Data.nDoc, args.nDocInit)
    InitData = Data.select_subset_by_mask(list(range(nDocInit)))

    # Create and initialize model
    hmodel = bnpy.HModel.CreateEntireModel(
        'moVB', 'HDPTopicModel', 'Mult',
        dict(alpha=0.5, gamma=10), dict(lam=0.1), Data)
    hmodel.init_global_params(
        InitData, K=args.Kinit, initname='kmeansplusplus', seed=5678)

    kwargs = locals()
    kwargs.update(args.__dict__)   
    runBirthForEveryCompInModel(**kwargs)

def runBirthForEveryCompInModel(
        hmodel=None, Data=None,
        bkwargs=dict(),
        nInitSteps=4,
        outputdir='/tmp/',
        args=None,
        **kwargs):
    ''' Attempt birth for every possible comp, and save HTML debug output.
    '''
    nDocTotal = Data.nDoc

    # Use default kwargs for local step
    LPkwargs = dict(**DefaultLPkwargs)

    # Define output directory
    outputdir = os.path.join(
        outputdir,
        '%s_nDoc=%d_K=%d' % (
            args.dataName, nDocTotal, args.Kinit)) 
    print('')
    print('==============')
    print("OUTPUT: ", outputdir)

    for i in range(nInitSteps):    
        LP = hmodel.calc_local_params(Data, **LPkwargs)
        SS = hmodel.get_global_suff_stats(
            Data, LP, doPrecompEntropy=1, doTrackTruncationGrowth=1)
        if i < nInitSteps:
            # Shuffle and remove really small comps
            # except on last step, since we need LP and SS to match up
            bigtosmall = np.argsort(-1 * SS.getCountVec())
            SS.reorderComps(bigtosmall)
            for k in range(SS.K-1, 0, -1):
                Nk = SS.getCountVec()[k]
                if Nk < 10:
                    SS.removeComp(k)
        hmodel.update_global_params(SS)

    # Obtain LP and SS exactly for the current hmodel.
    # Do NOT do any more global steps on hmodel.
    # We rely on this LP to be "fresh" in the proposal ELBO calculations.
    LP = hmodel.calc_local_params(Data, **LPkwargs)
    SS = hmodel.get_global_suff_stats(
        Data, LP, doPrecompEntropy=1, doTrackTruncationGrowth=1)

    for pos, targetUID in [(0, 0)]: #enumerate(SS.uids):
        jobdir = outdirPattern % (
            targetUID, 
            bkwargs['b_initHardCluster'],
            bkwargs['b_includeRemainderTopic'],
            args.Kfresh)
        b_debugOutputDir = os.path.join(outputdir, jobdir)
        mkpath(b_debugOutputDir)
        startuid = 1000 + 100 * targetUID + 1
        newUIDs=np.arange(startuid, startuid+args.Kfresh)

        print(jobdir)
        print('newUIDs: ', newUIDs)
        try:
            xSS, Info = createSplitStats(
                Data, hmodel, LP, curSSwhole=SS,
                targetUID=targetUID,
                batchPos=0,
                newUIDs=newUIDs,
                LPkwargs=LPkwargs,
                returnPropSS=0,
                b_debugOutputDir=b_debugOutputDir,
                **bkwargs)
        except BirthProposalError as e:
            print('  SKIPPED!', str(e))
            continue

        # Construct proposed stats
        propSS = SS.copy()
        propSS.transferMassFromExistingToExpansion(
            uid=targetUID, xSS=xSS)
        assert np.allclose(
            propSS.getCountVec().sum(),
            SS.getCountVec().sum())
        assert np.allclose(
            SS.getCountVec().sum(),
            Data.word_count.sum())

        # Create model via global step from proposed stats
        propModel = hmodel.copy()
        propModel.update_global_params(propSS)
        propLscore = propModel.calc_evidence(SS=propSS)

        # Compare ELBO scores
        curLscore = hmodel.calc_evidence(SS=SS)

        # Decision time: accept or reject
        if propLscore > curLscore:
            print('  ACCEPT!')
        else:
            print('  REJECT!')
        print('     curLscore %.5f' % (curLscore))
        print('    propLscore %.5f' % (propLscore))


if __name__ == '__main__':
    main()
