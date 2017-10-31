from builtins import *
import argparse
import numpy as np
import time

import bnpy.deletemove.DLogger as DLogger
from bnpy.ioutil.DataReader import loadDataFromSavedTask, loadLPKwargsFromDisk
from bnpy.ioutil.DataReader import loadKwargsFromDisk
from bnpy.ioutil.ModelReader import loadModelForLap

def parse_list_of_absorbing_comps(kabsorbList, ktarget, K):
    if kabsorbList == 'all':
        kabsorbList = list(range(K))
    elif kabsorbList.count(','):
        kabsorbList = [int(k) for k in kabsorbList.split(',')]
    elif kabsorbList.count('-'):
        kabsorbList = kabsorbList.split('-')
        kabsorbList = list(range(int(kabsorbList[0]), int(kabsorbList[1])+1))
    else:
        kabsorbList = [int(kabsorbList)]
    if ktarget in kabsorbList:
        kabsorbList.remove(ktarget)
    nIntersect = np.intersect1d(kabsorbList, list(range(K))).size
    assert nIntersect == len(kabsorbList)
    return kabsorbList

def tryDeleteProposalForSpecificTarget_DPMixtureModel(
        Data,
        hmodel,
        LPkwargs=dict(),
        ktarget=0,
        verbose=True,
        nUpdateSteps=50,
        **kwargs):
    ''' Execute merge for specific whole dataset

    Returns
    -------
    propModel : HModel
    propSS : SuffStatBag
    propLscore : scalar real
        ELBO score of proposed model
    curModel : HModel
    curSS : SuffStatBag
    curLscore : scalar real
        ELBO score of current model
    '''
    curModel = hmodel.copy()
    propModel = hmodel.copy()

    # Update current
    curLP = curModel.calc_local_params(Data, **LPkwargs)
    curSS = curModel.get_global_suff_stats(Data, curLP, doPrecompEntropy=1)
    curModel.update_global_params(curSS)
    curLscore = curModel.calc_evidence(SS=curSS)

    # Do Delete Proposal
    propResp = np.delete(curLP['resp'], ktarget, axis=1)
    propResp /= propResp.sum(axis=1)[:,np.newaxis]
    assert np.allclose(1.0, propResp.sum(axis=1))
    propLP = dict(resp=propResp)

    propLscoreList = list()
    for step in range(nUpdateSteps):
        if step > 0:
            propLP = propModel.calc_local_params(Data, **LPkwargs)
        propSS = propModel.get_global_suff_stats(
            Data, propLP, doPrecompEntropy=1)
        propModel.update_global_params(propSS)
        propLscore = propModel.calc_evidence(SS=propSS)
        propLscoreList.append(propLscore)
    if verbose:
        print("Deleting cluster %d..." % (ktarget))
        if propLscore - curLscore > 0:
            print("  ACCEPTED")
        else:
            print("  REJECTED")
        print("%.4e  cur ELBO score" % (curLscore))
        print("%.4e prop ELBO score" % (propLscore))
        print("Change in ELBO score: %.4e" % (propLscore - curLscore))
        print("")
    return (
        propModel,
        propSS,
        propLscoreList,
        curModel,
        curSS,
        curLscore)



def tryDeleteProposalForSpecificTarget_HDPTopicModel(
        Data,
        hmodel,
        LPkwargs=dict(),
        ktarget=0,
        kabsorbList=[1],
        verbose=True,
        doPlotComps=True,
        doPlotELBO=True,
        doPlotDocTopicCount=False,
        nELBOSteps=3,
        nUpdateSteps=5,
        d_initTargetDocTopicCount='warm_start',
        d_initWordCounts='none',
        **kwargs):
    ''' Execute merge for specific whole dataset

    Returns
    -------
    propModel : HModel
    propSS : SuffStatBag
    propLscore : scalar real
        ELBO score of proposed model
    curModel : HModel
    curSS : SuffStatBag
    curLscore : scalar real
        ELBO score of current model
    '''
    kabsorbList = parse_list_of_absorbing_comps(
        kabsorbList, ktarget, hmodel.obsModel.K)

    from bnpy.allocmodel.topics.HDPTopicRestrictedLocalStep2 \
        import summarizeRestrictedLocalStep_HDPTopicModel
    curModel = hmodel.copy()
    propModel = hmodel.copy()

    # Update current model
    if verbose:
        print("")
        print("Loading model from disk and performing local step...")
    starttime = time.time()
    curLP = curModel.calc_local_params(Data, **LPkwargs)
    curSS = curModel.get_global_suff_stats(Data, curLP, doPrecompEntropy=1)
    curModel.update_global_params(curSS)
    curLdict = curModel.calc_evidence(SS=curSS, todict=1)
    curLscore = curLdict['Ltotal']
    if verbose:
        print("%5.1f sec to obtain current model, LP, and SS" % (
            time.time() - starttime))

    nontrivialdocIDs = np.flatnonzero(curLP['DocTopicCount'][:, ktarget] > .01)
    sort_mask = np.argsort(-1*curLP['DocTopicCount'][nontrivialdocIDs, ktarget])
    nontrivialdocIDs = nontrivialdocIDs[sort_mask]
    docIDs = nontrivialdocIDs[:5]
    if verbose:
        print("")
        print("Proposing deletion of cluster %d" % (ktarget))
        print("    total mass N_k = %.1f" % (curSS.getCountVec()[ktarget]))
        print("    %d docs with non-trivial mass" % (nontrivialdocIDs.size))
        print("")
        print("Absorbing into %d/%d remaining clusters" % (
            len(kabsorbList), curSS.K-1))
        print(" ".join(['%3d' % (kk) for kk in kabsorbList]))
        print("")

    # Create init observation model for absorbing states
    xObsModel = propModel.obsModel.copy()
    xinitSS = curSS.copy(includeELBOTerms=False, includeMergeTerms=False)
    for k in reversed(np.arange(xObsModel.K)):
        if k not in kabsorbList:
            xinitSS.removeComp(k)
    # Find clusters correlated in appearance with the target
    if curModel.getObsModelName().count('Mult') and d_initWordCounts.count('bycorr'):
        corrVec = calcCorrelationFromTargetToAbsorbingSet(
            curLP['DocTopicCount'], ktarget, kabsorbList)
        bestAbsorbIDs = np.flatnonzero(corrVec >= .001)
        print("absorbIDs with best correlation:")
        print(bestAbsorbIDs)
        for k in bestAbsorbIDs:
            xinitSS.WordCounts[k,:] += curSS.WordCounts[ktarget,:]
    xObsModel.update_global_params(xinitSS)

    # Create init pi vector for absorbing states
    curPiVec = propModel.allocModel.get_active_comp_probs()
    xPiVec = curPiVec[kabsorbList].copy()
    xPiVec /= xPiVec.sum()
    xPiVec *= (curPiVec[kabsorbList].sum() +  curPiVec[ktarget])
    assert np.allclose(np.sum(xPiVec),
        curPiVec[ktarget] + np.sum(curPiVec[kabsorbList]))

    if verbose:
        print("Reassigning target mass among absorbing set...")
    starttime = time.time()
    propLscoreList = list()
    for ELBOstep in range(nELBOSteps):
        xSS, Info = summarizeRestrictedLocalStep_HDPTopicModel(
            Dslice=Data,
            curModel=curModel,
            curLPslice=curLP,
            ktarget=ktarget,
            kabsorbList=kabsorbList,
            curPiVec=curPiVec,
            xPiVec=xPiVec,
            xObsModel=xObsModel,
            nUpdateSteps=nUpdateSteps,
            d_initTargetDocTopicCount=d_initTargetDocTopicCount,
            LPkwargs=LPkwargs)

        if ELBOstep < nELBOSteps - 1:
            # Update the xObsModel
            xObsModel.update_global_params(xSS)
            # TODO: update xPiVec???

        print(" completed step %d/%d after %5.1f sec" % (
            ELBOstep+1, nELBOSteps, time.time() - starttime))

        propSS = curSS.copy()
        propSS.replaceCompsWithContraction(
            replaceSS=xSS,
            replaceUIDs=[curSS.uids[k] for k in kabsorbList],
            removeUIDs=[curSS.uids[ktarget]],
            )
        assert np.allclose(propSS.getCountVec().sum(),
            curSS.getCountVec().sum(),
            atol=0.01,
            rtol=0)
        propModel.update_global_params(propSS)
        propLdict = propModel.calc_evidence(SS=propSS, todict=1)
        propLscore = propModel.calc_evidence(SS=propSS)
        propLscoreList.append(propLscore)

    if verbose:
        print("")
        print("Proposal result:")
        if propLscore - curLscore > 0:
            print("  ACCEPTED")
        else:
            print("  REJECTED")
        print("%.4e  cur ELBO score" % (curLscore))
        print("%.4e prop ELBO score" % (propLscore))
        print("% .4e change in ELBO score" % (propLscore - curLscore))
        print("")
        for key in sorted(curLdict.keys()):
            if key.count('_') or key.count('total'):
                continue
            print("  gain %8s % .3e" % (
                key, propLdict[key] - curLdict[key]))
        print("")
        if docIDs.size > 0:
            np.set_printoptions(suppress=1, precision=2, linewidth=120)
            xLPslice = Info['xLPslice']

            print("BEFORE")
            print("-----")
            print(np.hstack([
                curLP['DocTopicCount'][docIDs,:][:,kabsorbList],
                curLP['DocTopicCount'][docIDs,:][:,ktarget][:,np.newaxis]
                ]))
            print("AFTER")
            print("-----")
            print(xLPslice['DocTopicCount'][docIDs,:])

    if doPlotELBO:
        import bnpy.viz
        from bnpy.viz.PlotUtil import pylab
        bnpy.viz.PlotUtil.ConfigPylabDefaults(pylab)
        iters = np.arange(len(propLscoreList))
        pylab.plot(iters, propLscoreList, 'b-')
        pylab.plot(iters, curLscore*np.ones_like(iters), 'k--')
        pylab.show()

    if doPlotDocTopicCount:
        import bnpy.viz
        from bnpy.viz.PlotUtil import pylab
        bnpy.viz.PlotUtil.ConfigPylabDefaults(pylab)

        kplotList = [x for x in kabsorbList]
        kplotList.append(ktarget)
        for d in docIDs:
            curDTClabels = ['%.1f' % (x) for x in
                curLP['DocTopicCount'][d, kplotList]]
            bnpy.viz.PlotComps.plotCompsFromHModel(
                curModel,
                compListToPlot=kplotList,
                compsToHighlight=[ktarget],
                xlabels=curDTClabels,
                vmin=0,
                vmax=.01)
            fig = pylab.gcf()
            fig.canvas.set_window_title('doc %d BEFORE' % (d))

            propLP = Info['xLPslice']
            propDTClabels = ['%.1f' % (x) for x in
                propLP['DocTopicCount'][d, :]]
            bnpy.viz.PlotComps.plotCompsFromHModel(
                propModel,
                xlabels=propDTClabels,
                vmin=0,
                vmax=.01)
            fig = pylab.gcf()
            fig.canvas.set_window_title('doc %d AFTER' % (d))
            pylab.show(block=False)

        # Plot docs
        dIm = np.zeros((docIDs.size*2, 900))
        dImLabels = list()
        tImLabels = list()
        row = 0
        for ii,d in enumerate(docIDs):
            start = Data.doc_range[d]
            stop = Data.doc_range[d+1]
            wid = Data.word_id[start:stop]
            wct = Data.word_count[start:stop]
            dIm[row, wid] = wct
            dImLabels.append('doc %d' % (d))

            tmask = np.flatnonzero(curLP['resp'][start:stop, ktarget] > .01)
            targetDoc = np.zeros(900)
            dIm[row+docIDs.size, wid[tmask]] = wct[tmask] \
                * curLP['resp'][start + 1*tmask, ktarget]
            tImLabels.append('trgt doc %d' % (d))
            row += 1

        bnpy.viz.BarsViz.showTopicsAsSquareImages(
            dIm,
            ncols=2,
            vmin=0,
            vmax=1,
            xlabels=dImLabels.extend(tImLabels),
            cmap='jet')
        pylab.show()

    if doPlotComps:
        import bnpy.viz
        from bnpy.viz.PlotUtil import pylab
        bnpy.viz.PlotUtil.ConfigPylabDefaults(pylab)

        bnpy.viz.PlotComps.plotCompsFromSS(
                curModel,
                curSS,
                compsToHighlight=[ktarget],
                vmin=0,
                vmax=.01)
        fig = pylab.gcf()
        fig.canvas.set_window_title('BEFORE')

        bnpy.viz.PlotComps.plotCompsFromSS(
                propModel,
                propSS,
                vmin=0,
                vmax=.01)
        fig = pylab.gcf()
        fig.canvas.set_window_title('AFTER')
        pylab.show()
    return (
        propModel,
        propSS,
        propLscoreList,
        curModel,
        curSS,
        curLscore)

def tryDeleteProposalForSavedTask(
        taskoutpath=None,
        lap=None,
        lapFrac=0,
        batchID=None,
        **kwargs):
    ''' Try specific delete proposal for specified taskoutpath

    Post Condition
    --------------
    * Logging messages are printed.
    '''
    if lap is not None:
        lapFrac = lap

    hmodel, lapFrac = loadModelForLap(taskoutpath, lapFrac)
    Data = loadDataFromSavedTask(taskoutpath, batchID=batchID)
    kwargs['LPkwargs'] = loadLPKwargsFromDisk(taskoutpath)

    tryDeleteProposalForSpecificTarget_HDPTopicModel(
        Data, hmodel,
        **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('taskoutpath', type=str)
    parser.add_argument('--lap', type=float, default=None)
    parser.add_argument('--lapFrac', type=float, default=None)
    parser.add_argument('--batchID', type=int, default=None)
    parser.add_argument('--doPlotELBO', type=int, default=0)
    parser.add_argument('--doPlotComps', type=int, default=1)
    parser.add_argument('--ktarget', type=int, default=10)
    parser.add_argument('--kabsorbList', type=str, default='all')
    parser.add_argument('--verbose', type=int, default=True)
    parser.add_argument('--outputdir', type=str, default='/tmp/')
    parser.add_argument('--nUpdateSteps', type=int, default=25)
    parser.add_argument('--nELBOSteps', type=int, default=1)
    parser.add_argument('--d_initWordCounts',
        type=str, default='none')
    parser.add_argument('--d_initTargetDocTopicCount',
        type=str, default="cold_start")
    args = parser.parse_args()

    DLogger.configure(args.outputdir,
        doSaveToDisk=0,
        doWriteStdOut=1)
    tryDeleteProposalForSavedTask(**args.__dict__)

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doPlotELBO', type=int, default=1)
    parser.add_argument('--doPlotComps', type=int, default=0)
    parser.add_argument('--ktarget', type=int, default=10)
    parser.add_argument('--kabsorbList', type=str, default='all')
    parser.add_argument('--initname', type=str, default='truelabelsandjunk')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--nLap', type=int, default=1)
    args = parser.parse_args()
    ktarget = args.ktarget
    kabsorbList = args.kabsorbList

    LPkwargs = dict(
        restartLP=0,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        nCoordAscentItersLP=100,
        convThrLP=0.01)

    import bnpy
    hmodel, Info = bnpy.run('BarsK10V900', 'HDPTopicModel', 'Mult', 'memoVB',
        initname=args.initname,
        nLap=args.nLap,
        K=args.K,
        nBatch=1, nDocTotal=100, nWordsPerDoc=500,
        alpha=0.5,
        gamma=10.0,
        lam=0.1,
        **LPkwargs)
    Data = Info['Data'].getBatch(0)
    if kabsorbList == 'all':
        kabsorbList = range(hmodel.obsModel.K)
    elif kabsorbList.count(','):
        kabsorbList = [int(k) for k in kabsorbList.split(',')]
    elif kabsorbList.count('-'):
        kabsorbList = kabsorbList.split('-')
        kabsorbList = range(int(kabsorbList[0]), int(kabsorbList[1])+1)
    else:
        kabsorbList = [int(kabsorbList)]
    if ktarget in kabsorbList:
        kabsorbList.remove(ktarget)
    nIntersect = np.intersect1d(kabsorbList, range(hmodel.obsModel.K)).size
    assert nIntersect == len(kabsorbList)

    tryDeleteProposalForSpecificTarget_HDPTopicModel(
        Data,
        hmodel,
        LPkwargs=LPkwargs,
        ktarget=ktarget,
        kabsorbList=kabsorbList,
        doPlotComps=args.doPlotComps,
        doPlotELBO=args.doPlotELBO,
        nUpdateSteps=10,)
'''
