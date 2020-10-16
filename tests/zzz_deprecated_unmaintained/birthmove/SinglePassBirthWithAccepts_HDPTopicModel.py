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
    nDocPerBatch=10,
    nDocTotal=500,
    nWordsPerDoc=400,
    nFixedInitLaps=0,
    Kinit=1,
    targetUID=0,
    creationProposalName='kmeans',
    doInteractiveViz=False,
    ymin=-2,
    dataName='nips',
    ymax=1.5, 
    Kfresh=10, 
    doShowAfter=1,
    outputdir='/tmp/',
    b_includeRemainderTopic=1,
    b_nRefineSteps=3,
    b_minNumAtomsInDoc=100,
    )

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
                _type = int
            except:
                _type = str
        parser.add_argument('--' + key, default=val, type=_type)
    args = parser.parse_args()
    
    # Dump contents of args into locals
    nDocPerBatch = args.nDocPerBatch
    nDocTotal = args.nDocTotal
    nWordsPerDoc = args.nWordsPerDoc
    nFixedInitLaps = args.nFixedInitLaps
    Kinit = args.Kinit
    creationProposalName = args.creationProposalName
    targetUID = args.targetUID
    doInteractiveViz = args.doInteractiveViz
    Kfresh = args.Kfresh
    dataName = args.dataName
    outputdir = args.outputdir
    print("OUTPUT: ", outputdir)
    nBatch = int(nDocTotal // nDocPerBatch)

    LPkwargs = DefaultLPkwargs

    bkwargs = dict()
    for key in args.__dict__:
        if key.startswith('b_'):
            bkwargs[key] = getattr(args, key)
    print('BIRTH kwargs:')
    for key in bkwargs:
        print(key, bkwargs[key])

    if dataName.count('BarsK10V900'):
        import BarsK10V900
        Data = BarsK10V900.get_data(
            nDocTotal=nDocTotal, nWordsPerDoc=nWordsPerDoc)
        Data.alwaysTrackTruth = 1
    else:   
        os.environ['BNPYDATADIR'] = os.path.join(
            os.environ['HOME'], 'git/x-topics/datasets/' + dataName + '/')
        sys.path.append(os.environ['BNPYDATADIR'])
        dataMod = __import__(dataName, fromlist=[])
        Data = dataMod.get_data()
        vocabList = Data.vocabList
        Data = Data.select_subset_by_mask(list(range(nDocTotal)))
        Data.name = dataName
        Data.vocabList = vocabList
    DataIterator = Data.to_iterator(nBatch=nBatch, nLap=10)
    

    # Use first few docs to initialize!
    initPRNG = np.random.RandomState(5678)
    initDocIDs = DataIterator.IDsPerBatch[0][:4]
    InitData = Data.select_subset_by_mask(initDocIDs)

    hmodel = bnpy.HModel.CreateEntireModel(
        'moVB', 'HDPTopicModel', 'Mult',
        dict(alpha=0.5, gamma=10), dict(lam=0.1), Data)
    hmodel.init_global_params(
        InitData, K=Kinit, initname='kmeansplusplus', seed=5678)
    
    Dbatch = DataIterator.getBatch(0)

    rejectedComps = dict()
    didAccept = True
    nRep = 0
    SS = None
    while didAccept:
        nRep += 1
        print('')
        print('')
        print('rep %d' % (nRep))
        didAccept = False

        LPbatch = hmodel.calc_local_params(Dbatch, **LPkwargs)
        SSbatch = hmodel.get_global_suff_stats(
            Dbatch, LPbatch, doPrecompEntropy=1, doTrackTruncationGrowth=1)

        if SS is None:
            SS = SSbatch
        else:
            uids = SS.uids
            SS = SSbatch
            SS.setUIDs(uids)
        hmodel.update_global_params(SS)

        for pos, targetUID in enumerate(SS.uids):
            if targetUID in rejectedComps:
                continue
            propdir = 'rep=%d_targetUID=%d' % (nRep, targetUID)
            b_debugOutputDir = os.path.join(outputdir, propdir)
            mkpath(b_debugOutputDir)

            startuid = 1000 * nRep + Kfresh * pos
            newUIDs=np.arange(startuid, startuid+Kfresh)
            try:
                xSS, Info = createSplitStats(
                    Dbatch, hmodel, LPbatch, curSSwhole=SS,
                    creationProposalName=creationProposalName,
                    targetUID=targetUID,
                    batchPos=0,
                    newUIDs=newUIDs,
                    LPkwargs=LPkwargs,
                    returnPropSS=0,
                    b_debugOutputDir=b_debugOutputDir,
                    **bkwargs)
            except BirthProposalError as e:
                print('SKIPPED!')
                print(str(e))
                rejectedComps[targetUID] = 1
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
                Dbatch.word_count.sum())

            # Create model via global step from proposed stats
            propModel = hmodel.copy()
            propModel.update_global_params(propSS)
            # Compare ELBO scores
            curLscore = hmodel.calc_evidence(SS=SS)
            propLscore = propModel.calc_evidence(SS=propSS)

            curLdict = hmodel.calc_evidence(SS=SS, todict=1)
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)

            if doInteractiveViz:
                xlabels = uidsAndCounts2strlist(propSS)

                if Data.name.count('Bars'):
                    bnpy.viz.BarsViz.showTopicsAsSquareImages(
                        Info['xSSfake'].WordCounts,
                        vmax=10,
                        xlabels=[
                            '%.0f' % (x)
                            for x in Info['xSSfake'].getCountVec()])
                    bnpy.viz.PlotComps.plotCompsFromHModel(
                        propModel,
                        compsToHighlight=[pos],
                        xlabels=xlabels)
                else:
                    print('TOPIC TO SPLIT')
                    ktarget = SS.uid2k(targetUID)
                    printTopWordsFromWordCounts(
                        SS.WordCounts[ktarget][np.newaxis,:], Data.vocabList)
                    print('NEW TOPICS')
                    printTopWordsFromWordCounts(
                        propSS.WordCounts, Data.vocabList)
                    plotCompsFromWordCounts(
                        propSS.WordCounts, Data.vocabList,
                        xlabels=xlabels,
                        )

            # Decision time: accept or reject
            if propLscore > curLscore:
                print('ACCEPT!')
                hmodel = propModel
                SS = propSS
                didAccept = True
            else:
                print('REJECT!')
                rejectedComps[targetUID] = 1
            print(' curLscore %.5f' % (curLscore))
            print('propLscore %.5f' % (propLscore))

            for key in ['Ldata', 'Lentropy', 'Lalloc', 'LcDtheta']:
                print('  %s : %.5f' % (key, propLdict[key] - curLdict[key]))

            if doInteractiveViz:
                pylab.show(block=False)
                keypress = input("Press any key >>>")
                if keypress == 'embed':
                    from IPython import embed; embed()
                pylab.close('all')

        # Shuffle and remove really small comps
        bigtosmall = np.argsort(-1 * SS.getCountVec())
        SS.reorderComps(bigtosmall)
        for k in range(SS.K-1, 0, -1):
            Nk = SS.getCountVec()[k]
            if Nk < 10:
                SS.removeComp(k)
        hmodel.update_global_params(SS)

    # Final phase: merges
    from bnpy.mergemove import MergePlanner, MergeMove

    print(SS.uids)
    print(' '.join(['%.0f' % (x) for x in SS.getCountVec()]))
    mPairIDs = None
    for step in range(10):
        uids = SS.uids
        if step > 0:
            mPairIDs, PairScoreMat = MergePlanner.preselect_candidate_pairs(
                hmodel, SS,
                randstate=np.random.RandomState(nRep),
                returnScoreMatrix=1,
                M=None,
                mergePairSelection='corr',
                mergeNumExtraCandidates=0)
        LPbatch = hmodel.calc_local_params(Dbatch, **LPkwargs)
        SS = hmodel.get_global_suff_stats(
            Dbatch, LPbatch, 
            doPrecompEntropy=1, doTrackTruncationGrowth=1,
            doPrecompMerge=0, mPairIDs=mPairIDs, mergePairSelection='corr')
        SS.setUIDs(uids)

        hmodel.update_global_params(SS)
        if mPairIDs is not None and len(mPairIDs) > 0:
            curLscore = hmodel.calc_evidence(SS=SS)
            newModel, newSS, newLscore, Info = MergeMove.run_many_merge_moves(
                hmodel, SS, curLscore, mPairIDs=mPairIDs)
            if Info['ELBOGain'] > 0:
                print('ACCEPTED MERGE!')
                from IPython import embed; embed()
                hmodel = newModel
                SS = newSS
            else:
                break

    bnpy.viz.PlotComps.plotCompsFromHModel(
        hmodel,
        xlabels=[
            '%.0f' % (x)
            for x in SS.getCountVec()])
    pylab.show(block=True)


def getColor(key):
    if key.count('total'):
        return 'k'
    elif key.count('data'):
        return 'b'
    elif key.count('entrop'):
        return 'r'
    elif key.count('alloc'):
        return 'c'
    else:
        return 'm'

if __name__ == '__main__':
    main()
