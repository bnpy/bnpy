import argparse
import numpy as np
import os
import sys
import bnpy
import time

from bnpy.birthmove.SCreateFromScratch import createSplitStats
from bnpy.birthmove.SAssignToExisting import assignSplitStats
from bnpy.birthmove.SCreateFromScratch import DefaultLPkwargs
from bnpy.viz.PlotUtil import ConfigPylabDefaults
from matplotlib import pylab;

ConfigPylabDefaults(pylab)

DefaultKwargs = dict(
     nDocPerBatch=10,
     nDocTotal=500, nWordsPerDoc=400,
     nFixedInitLaps=0,
     Kinit=1,
     targetUID=0,
     creationProposalName='kmeans',
     doInteractiveViz=False,
     ymin=-2,
     ymax=1.5, 
     Kfresh=10, 
     doShowAfter=1,
     outputdir='/tmp/',
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
    outputdir = args.outputdir
    print("OUTPUT: ", outputdir)
    nBatch = int(nDocTotal // nDocPerBatch)

    LPkwargs = DefaultLPkwargs

    import BarsK10V900
    Data = BarsK10V900.get_data(nDocTotal=nDocTotal, nWordsPerDoc=nWordsPerDoc)
    Data.alwaysTrackTruth = 1
    DataIterator = Data.to_iterator(nBatch=nBatch, nLap=10)
    
    # Use first few docs to initialize!
    initPRNG = np.random.RandomState(5678)
    #initDocIDs = initPRNG.choice(Data.nDoc, 25, replace=False)
    initDocIDs = DataIterator.IDsPerBatch[0][:4]
    InitData = Data.select_subset_by_mask(initDocIDs)

    hmodel = bnpy.HModel.CreateEntireModel(
        'moVB', 'HDPTopicModel', 'Mult',
        dict(alpha=0.5, gamma=10), dict(lam=0.1), Data)
    hmodel.init_global_params(
        InitData, K=Kinit, initname='kmeansplusplus', seed=5678)

    # Do some fixed-truncation local/global steps
    SS = None
    SSmemory = dict()
    nDocsSeenBefore = 0
    for lap in range(nFixedInitLaps):
        for batchID in range(nBatch):
            Dbatch = DataIterator.getBatch(batchID)

            LPbatch = hmodel.calc_local_params(Dbatch, **LPkwargs)
            SSbatch = hmodel.get_global_suff_stats(
                Dbatch, LPbatch, doPrecompEntropy=1, doTrackTruncationGrowth=1)

            if batchID in SSmemory:        
                SS -= SSmemory[batchID]
            SSmemory[batchID] = SSbatch
            if SS is None:
                SS = SSbatch.copy()
            else:
                SS += SSbatch
            hmodel.update_global_params(SS)

            nDocsSeenBefore += SSbatch.nDoc
    
    Lines = dict()
    Lines['xs'] = list()
    for batchID in range(nBatch):
        print('batch %d/%d' % (batchID+1, nBatch))
        Dbatch = DataIterator.getBatch(batchID)

        LPbatch = hmodel.calc_local_params(Dbatch, **LPkwargs)
        SSbatch = hmodel.get_global_suff_stats(
            Dbatch, LPbatch, doPrecompEntropy=1, doTrackTruncationGrowth=1)
        nDocsSeenBefore += SSbatch.nDoc
        if batchID in SSmemory:        
            SS -= SSmemory[batchID]
        SSmemory[batchID] = SSbatch
        if SS is None:
            SS = SSbatch.copy()
        else:
            SS += SSbatch
        
        if batchID == 0:
            xSSbatch, propSSbatch = createSplitStats(
                Dbatch, hmodel, LPbatch, curSSwhole=SS,
                creationProposalName=creationProposalName,
                targetUID=targetUID,
                batchPos=batchID,
                newUIDs=np.arange(100, 100+Kfresh),
                LPkwargs=LPkwargs,
                returnPropSS=1)

            xSSbatch_first = xSSbatch
            LPbatch_first = LPbatch
            xSS = xSSbatch.copy()
            propSS_agg = propSSbatch.copy()
        else:
            xSSbatch, propSSbatch = assignSplitStats(
                Dbatch, hmodel, LPbatch, SS, xSS,
                targetUID=targetUID,
                LPkwargs=LPkwargs,
                returnPropSS=1)
            xSS += xSSbatch
            propSS_agg += propSSbatch

        propSS_whole = propSS_agg.copy()
        for rembatchID in range(batchID+1, nBatch):
            if rembatchID in SSmemory:
                remSSbatch = SSmemory[rembatchID].copy()
                Kextra = propSS_whole.K - SSbatch.K
                if Kextra > 0:
                    remSSbatch.insertEmptyComps(Kextra)
                propSS_whole += remSSbatch

        hmodel.update_global_params(SS)

        if batchID < 32 or (batchID + 1) % 4 == 0:
            curLscore = hmodel.calc_evidence(SS=SS)
            curLbyterm = hmodel.calc_evidence(SS=SS, todict=1)

            propSS = SS.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=xSS)

            for field in ['sumLogPi', 'sumLogPiRemVec']:
                arr = getattr(propSS, field)
                arr_direct = getattr(propSS_whole, field)
                if not np.allclose(arr, arr_direct):
                    print('  Error detected in field: %s' % (field))
                    from IPython import embed; embed()
                print('  SS field %s verified' % (field))

            for field in ['gammalnTheta', 'slackTheta', 'slackThetaRem',
                          'gammalnSumTheta', 'gammalnThetaRem']:
                arr = getattr(propSS._ELBOTerms, field)
                arr_direct = getattr(propSS_whole._ELBOTerms, field)
                if not np.allclose(arr, arr_direct):
                    print('  Error detected in field: %s' % (field))
                    from IPython import embed; embed()
                print('  ELBO field %s verified' % (field))

            propModel = hmodel.copy()
            propModel.update_global_params(propSS)

            propLscore = propModel.calc_evidence(SS=propSS)
            propLbyterm = propModel.calc_evidence(SS=propSS, todict=1)

            assert np.allclose(SS.getCountVec().sum(),
                               propSS.getCountVec().sum())

            print(' curLscore %.3f' % (curLscore))
            print('propLscore %.3f' % (propLscore))
            highlightComps = np.hstack([targetUID, np.arange(Kinit, propSS.K)])
            if propLscore > curLscore:
                print('ACCEPTED!')
            else:
                print('REJECTED <<<<<<<<<< :(')

            if doInteractiveViz:
                bnpy.viz.PlotComps.plotCompsFromHModel(
                    propModel, compsToHighlight=highlightComps)
                pylab.show(block=False)
                keypress = input("Press key to continue >>>")
                if keypress.count('embed'):
                    from IPython import embed; embed()

            Lines['xs'].append(nDocsSeenBefore)
            for key in ['Ldata', 'Lentropy', 'Lalloc', 'LcDtheta']:
                for versionName in ['cur', 'prop']:
                    versionKey = versionName + "-" + key
                    if versionName.count('cur'):
                        val = curLbyterm[key]
                    else:
                        val = propLbyterm[key]
                    try:
                        Lines[versionKey].append(val)
                    except KeyError:
                        Lines[versionKey] = [val]

    pylab.figure(figsize=(6,6)); pylab.hold('on');
    pylab.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

    legendKeys = ['Ldata', 'Lentropy', 'Lalloc', 'LcDtheta', 'Ltotal']
    Lines['cur-Ltotal'] = np.zeros_like(Lines['cur-Ldata'])
    Lines['prop-Ltotal'] = np.zeros_like(Lines['cur-Ldata'])
    Lines['xs'] = np.asarray(Lines['xs'])
    for basekey in legendKeys:
        if basekey.count('total'):
            linewidth= 4
            alpha = 1
        else:
            linewidth = 3
            alpha = 0.5
        for modelkey in ['prop', 'cur']:
            key = modelkey + '-' + basekey
            Lines[key] = np.asarray(Lines[key])
            if key.count('cur'):
                label = basekey
                style = '-'
                if key.count('total') == 0:
                    Lines['cur-Ltotal'] += Lines[key]
            else:
                label = None
                style = '--'
                if key.count('total') == 0:
                    Lines['prop-Ltotal'] += Lines[key]
        diffval = Lines['prop-' + basekey] - Lines['cur-' + basekey]
        pylab.plot(Lines['xs'],
                   diffval,
                   style,
                   color=getColor(key),
                   linewidth=linewidth,
                   alpha=alpha,
                   label=label)
    pylab.gca().set_xscale('log')
    M = int(np.ceil(np.log(nDocTotal)/np.log(2)))
    xticks = np.asarray([2**x for x in range(0, M+1)])
    if xticks.size > 5:
        xticks = xticks[::2] # keep every second
    if xticks[-1] < M:
        xticks = np.append(xticks, M)

    xlims = [1.0/8.0, xticks[-1] * 2]
    pylab.xlim(xlims)
    pylab.xticks(xticks)
    pylab.gca().set_xticklabels(xticks)

    pylab.ylim([args.ymin, args.ymax])
    pylab.plot(xlims, np.zeros_like(xlims), 'k--')
    pylab.xlabel('number of docs processed')
    pylab.ylabel('L gain (proposal - current)')

    good_xs = np.flatnonzero(diffval > 0)
    if good_xs.size > 0:
        xstart = Lines['xs'][good_xs[0]]
        xstop = Lines['xs'][good_xs[-1]] + nDocPerBatch // 2
        pylab.axvspan(xstart, xstop, color='green', alpha=0.2)
        if xstart > nDocPerBatch:
            pylab.axvspan(xticks[0], xstart, color='red', alpha=0.2)
        if xstop < nDocTotal:
            pylab.axvspan(xstop, xticks[-1], color='red', alpha=0.2)
    else:
        pylab.axvspan(xlims[0], xlims[-1], color='red', alpha=0.2)
    pylab.draw()

    lhandles, labels = pylab.gca().get_legend_handles_labels()
    order = [0, 1, 4, 2, 3]
    lhandles = [lhandles[o] for o in order]
    labels = [labels[o] for o in order]
    pylab.legend(lhandles, labels,
                 loc='lower left',
                 ncol=1)
    keys = ['nWordsPerDoc', 'nDocPerBatch', 'nDocTotal', 
            'Kinit', 'targetUID', 'nFixedInitLaps']
    filesuffix = ''
    for key in keys:
        filesuffix += '-%s=%d' % (key, getattr(args, key))
    filename = os.path.join(outputdir, 'ELBOgain')
    pylab.savefig(filename + filesuffix + '.png', bbox_inches='tight', pad_inches=0)

    bnpy.viz.PlotComps.plotCompsFromHModel(
        hmodel, compsToHighlight=[targetUID])
    filename = os.path.join(outputdir, 'BeforeComps')
    pylab.savefig(filename + filesuffix + '.png', bbox_inches='tight', pad_inches=0)

    seedModel = hmodel.copy()
    seedModel.update_global_params(xSSbatch_first)
    bnpy.viz.PlotComps.plotCompsFromHModel(
        seedModel)
    filename = os.path.join(outputdir, 'FirstFreshComps')
    pylab.savefig(filename + filesuffix + '.png', bbox_inches='tight', pad_inches=0)

    bnpy.viz.PlotComps.plotCompsFromHModel(
        propModel, compsToHighlight=highlightComps)
    filename = os.path.join(outputdir, 'AfterComps')
    pylab.savefig(filename + filesuffix + '.png', bbox_inches='tight', pad_inches=0)
    print(filename + filesuffix + '.png', '<<<<<<')

    # Show document subset!
    Dfirst = DataIterator.getBatch(0)
    DocTypeMat = Dfirst.getDocTypeCountMatrix()
    PRNG = np.random.RandomState(0)
    if Dfirst.nDoc > 25:
        relDocs = PRNG.choice(nDocPerBatch, 25, replace=False)
        DocTypeMat = DocTypeMat[relDocs]
    else:
        relDocs = np.arange(Dfirst.nDoc)
    bnpy.viz.BarsViz.showTopicsAsSquareImages(DocTypeMat, vmax=5, cmap='bone_r')
    filename = os.path.join(outputdir, 'FirstDocs')
    pylab.savefig(filename + filesuffix + '.png', bbox_inches='tight', pad_inches=0)

    # Show reassigned subset of document subset!
    relResp = LPbatch_first['resp'][:, targetUID]
    for relID, d in enumerate(relDocs):
        start = Dfirst.doc_range[d]
        stop = Dfirst.doc_range[d+1]
        DocTypeMat[relID, Dfirst.word_id[start:stop]] *= relResp[start:stop]
    bnpy.viz.BarsViz.showTopicsAsSquareImages(DocTypeMat, vmax=5, cmap='bone_r')
    filename = os.path.join(outputdir, 'FirstRelevantDocs')
    pylab.savefig(filename + filesuffix + '.png', bbox_inches='tight', pad_inches=0)



    if args.doShowAfter:
        pylab.show(block=False)
        keypress = input("Press key to continue >>>")
        if keypress.count('embed'):
            from IPython import embed; embed()


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
