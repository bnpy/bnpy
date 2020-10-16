'''
Generic tests for using merge moves during model training with bnpy.
'''
import os
import sys
import numpy as np
import unittest
from nose.plugins.attrib import attr

import bnpy


def arg2name(aArg):
    if isinstance(aArg, dict):
        aName = aArg['name']
    elif isinstance(aArg, str):
        aName = aArg
    return aName


def pprintResult(model, Info, Ktrue=0):
    """ Pretty print the result of a learning algorithm.
    """
    hdist_str = ''
    if 'outputdir' in Info and Info['outputdir'] is not None:
        hdistfile = os.path.join(Info['outputdir'], 'hamming-distance.txt')
        if os.path.exists(hdistfile):
            hdist_str = 'hdist=' + '%.3f' % (float(np.loadtxt(hdistfile)[-1]))

    print(" %25s after %4.1f sec and %4d laps.  ELBO=% 7.5f %s K=%d  Ktrue=%d"\
        % (Info['status'][:25],
           Info['elapsedTimeInSec'],
           Info['lapTrace'][-1],
           Info['evBound'],
           hdist_str,
           model.allocModel.K,
           Ktrue,
           ))


def pprint(val):
    """ Pretty print the provided value.
    """
    if isinstance(val, str):
        print('  %s' % (val[:40]))
    elif hasattr(val, 'items'):
        firstMsg = ''
        msg = ''
        for (k, v) in list(val.items()):
            if k.count('name'):
                firstMsg = str(v)
            else:
                msg += " %s=%s" % (k, str(v))
        print('  ' + firstMsg + ' ' + msg)


def pprintCommandToReproduceError(dataArg, aArg, oArg, algName, **kwargs):
    for key, val in list(dataArg.items()):
        if key == 'name':
            continue
        kwargs[key] = val
    del kwargs['doWriteStdOut']
    del kwargs['doSaveToDisk']
    kwargs['printEvery'] = 1
    kwstr = ' '.join(['--%s %s' % (key, kwargs[key]) for key in kwargs])
    print("python -m bnpy.Run %s %s %s %s %s" % (
        dataArg['name'],
        aArg['name'],
        oArg['name'],
        algName,
        kwstr,
    ))


def is_monotonic(ELBOvec, aArg=None, atol=1e-5, verbose=True):
    ''' Returns True if provided vector monotonically increases, False o.w.

    Returns
    -------
    result : boolean (True or False)
    '''
    if aArg is not None:
        if 'name' in aArg:
            if aArg['name'] == 'HDPTopicModel':
                # ELBO can fluctuate more due to no caching at localstep
                atol = 1e-3

    ELBOvec = np.asarray(ELBOvec, dtype=np.float64)
    assert ELBOvec.ndim == 1
    diff = ELBOvec[1:] - ELBOvec[:-1]
    maskIncrease = diff > 0
    maskWithinTol = np.abs(diff) < atol
    maskOK = np.logical_or(maskIncrease, maskWithinTol)
    isMonotonic = np.all(maskOK)
    if not isMonotonic and verbose:
        print("NOT MONOTONIC!")
        print('  %d violations in vector of size %d. Biggest drop %.8f' \
            % (np.sum(1 - maskOK), ELBOvec.size, diff[diff < 0].max()))
    return isMonotonic


class MergeMoveEndToEndTest(unittest.TestCase):

    """ Defines test exercises for executing bnpy.run on provided dataset.

    Attributes
    ----
    Data : bnpy.data.DataObj
        dataset under testing
    """

    __test__ = False  # Do not execute this abstract module!

    def shortDescription(self):
        return None

    def makeAllKwArgs(self, aArg, obsArg, initArg=dict(),
                      **kwargs):

        allKwargs = dict(
            doSaveToDisk=False,
            doWriteStdOut=False,
            saveEvery=-1,
            printEvery=-1,
            traceEvery=1,
            convergeThr=0.0001,
            doFullPassBeforeMstep=1,
            nLap=300,
            nBatch=2,
            mergeStartLap=2,
            deleteStartLap=2,
            nCoordAscentItersLP=50,
            convThrLP=0.001,
            creationProposalName='randBlocks',
            minBlockSize=10,
            maxBlockSize=50,
            doVizSeqCreate=0,
        )
        allKwargs.update(kwargs)
        allKwargs.update(aArg)
        allKwargs.update(obsArg)
        allKwargs.update(initArg)
        allKwargs.update(self.datasetArg)
        if allKwargs['moves'].count('delete'):
            try:
                MaxSize = 0.5 * int(self.datasetArg['nDocTotal'])
            except KeyError:
                MaxSize = 0.5 * int(self.datasetArg['nObsTotal'])
            allKwargs['dtargetMaxSize'] = int(MaxSize)

        if aArg['name'] == 'HDPTopicModel':
            allKwargs['mergePairSelection'] = 'corrlimitdegree'
        else:
            allKwargs['mergePairSelection'] = 'wholeELBObetter'
        return allKwargs

    def run_MOVBWithMoves(self, aArg, oArg,
                          moves='merge',
                          algName='moVB',
                          nWorkers=0,
                          **kwargs):
        """ Execute single run with merge moves enabled.

        Post Condition
        --------------
        Will raise AssertionError if any bad results detected.
        """
        Ktrue = self.Data.TrueParams['K']
        pprint(aArg)
        pprint(oArg)
        initArg = dict(**kwargs)
        pprint(initArg)
        kwargs = self.makeAllKwArgs(
            aArg, oArg, initArg,
            moves=moves, nWorkers=nWorkers, **kwargs)
        model, Info = bnpy.run(
            self.Data, arg2name(aArg), arg2name(oArg), algName, **kwargs)
        pprintResult(model, Info, Ktrue=Ktrue)

        afterFirstLapMask = Info['lapTrace'] >= 1.0
        evTraceAfterFirstLap = Info['evTrace'][afterFirstLapMask]
        isMonotonic = is_monotonic(evTraceAfterFirstLap,
                                   aArg=aArg)

        try:
            assert isMonotonic
            assert model.allocModel.K == model.obsModel.K
            assert model.allocModel.K == Ktrue

        except AssertionError as e:
            pprintCommandToReproduceError(
                self.datasetArg, aArg, oArg, algName, **kwargs)
            assert isMonotonic
            assert model.allocModel.K == model.obsModel.K
            if not model.allocModel.K == Ktrue:
                print('>>>>>> WHOA! Kfinal != Ktrue <<<<<<')
        return Info

    def run_MOVBWithMoves_SegmentManySeq(
            self, aArg, oArg, moves='merge,delete,shuffle,seqcreate',
            algName='moVB',
            nWorkers=0,
            **kwargs):
        """ Execute single run with all moves enabled.

        Post Condition
        --------------
        Will raise AssertionError if any bad results detected.
        """
        self.Data.alwaysTrackTruth = 1
        Ktrue = np.unique(self.Data.TrueParams['Z']).size

        pprint(aArg)
        pprint(oArg)
        initArg = dict(**kwargs)
        pprint(initArg)

        viterbiPath = os.path.expandvars(
            '$BNPYROOT/bnpy/learnalg/extras/XViterbi.py')
        kwargs = self.makeAllKwArgs(aArg, oArg, initArg,
                                    moves=moves, nWorkers=nWorkers,
                                    customFuncPath=viterbiPath,
                                    doSaveToDisk=1,
                                    doWriteStdOut=1,
                                    printEvery=1,
                                    saveEvery=1000,
                                    **kwargs)

        kwargs['jobname'] += '-creationProposalName=%s' % (
            kwargs['creationProposalName'])
        model, Info = bnpy.run(
            self.Data, arg2name(aArg), arg2name(oArg), algName, **kwargs)
        pprintResult(model, Info, Ktrue=Ktrue)
        try:
            assert model.allocModel.K == model.obsModel.K
            assert model.allocModel.K == Ktrue

        except AssertionError as e:
            pprintCommandToReproduceError(
                self.datasetArg, aArg, oArg, algName, **kwargs)
            assert model.allocModel.K == model.obsModel.K
            if not model.allocModel.K == Ktrue:
                print('>>>>>> WHOA! Kfinal != Ktrue <<<<<<')
        print('')
        return Info

    def run_MOVBWithMoves_SegmentSingleSeq(
            self, aArg, oArg,
            moves='merge,delete,shuffle,seqcreate',
            algName='moVB', nWorkers=0, n=0, **kwargs):
        """ Execute single run with all moves enabled.

        Post Condition
        --------------
        Will raise AssertionError if any bad results detected.
        """
        if hasattr(self.Data, 'nDoc'):
            Data_n = self.Data.select_subset_by_mask(
                [n], doTrackTruth=1, doTrackFullSize=0)
            assert Data_n.nDocTotal == 1
        else:
            # Make GroupXData dataset from XData
            # This code block rearranges rows so that we
            # cycle thru the true labels twice as contig blocks.
            zTrue = self.Data.TrueParams['Z']
            half1 = list()
            half2 = list()
            for uID in np.unique(zTrue):
                dataIDs = np.flatnonzero(zTrue == uID)
                Nk = dataIDs.size
                half1.append(dataIDs[:Nk / 2])
                half2.append(dataIDs[Nk / 2:])
            dIDs_1 = np.hstack([x for x in half1])
            dIDs_2 = np.hstack([x for x in half2])
            dIDs = np.hstack([dIDs_1, dIDs_2])

            Data_n = bnpy.data.GroupXData(
                X=self.Data.X[dIDs],
                doc_range=np.asarray([0, self.Data.nObs]),
                TrueZ=self.Data.TrueParams['Z'][dIDs])

            aArg['name'] = 'HDPHMM'
            aArg['hmmKappa'] = 50
            aArg['alpha'] = 0.5
            aArg['gamma'] = 10.0
            aArg['startAlpha'] = 10.0

        Data_n.name = self.Data.name
        Data_n.alwaysTrackTruth = 1
        if hasattr(self.Data, 'TrueParams'):
            assert hasattr(Data_n, 'TrueParams')
            Ktrue = np.unique(Data_n.TrueParams['Z']).size

        pprint(aArg)
        pprint(oArg)
        initArg = dict(**kwargs)
        pprint(initArg)
        viterbiPath = os.path.expandvars(
            '$BNPYROOT/bnpy/learnalg/extras/XViterbi.py')
        kwargs = self.makeAllKwArgs(aArg, oArg, initArg,
                                    moves=moves, nWorkers=nWorkers,
                                    customFuncPath=viterbiPath,
                                    doSaveToDisk=1,
                                    doWriteStdOut=1,
                                    printEvery=1,
                                    saveEvery=1000,
                                    nBatch=1,
                                    **kwargs)

        kwargs['jobname'] += '-creationProposalName=%s' % (
            kwargs['creationProposalName'])
        model, Info = bnpy.run(
            Data_n, arg2name(aArg), arg2name(oArg), algName, **kwargs)

        if Ktrue == 0:
            pprintResult(model, Info, Ktrue=Ktrue)
        else:
            pprintResult(model, Info, Ktrue=Ktrue)
            try:
                assert model.allocModel.K == model.obsModel.K
                assert model.allocModel.K == Ktrue

            except AssertionError as e:
                pprintCommandToReproduceError(
                    self.datasetArg, aArg, oArg, algName, **kwargs)
                assert model.allocModel.K == model.obsModel.K
                if not model.allocModel.K == Ktrue:
                    print('>>>>>> WHOA! Kfinal != Ktrue <<<<<<')
        print('')

        '''
        from bnpy.viz import SequenceViz
        SequenceViz.plotSingleJob(
            self.Data.name, kwargs['jobname'],
            taskids='1', lap='final',
            sequences=[1],
            showELBOInTitle=False,
            dispTrue=True,
            aspectFactor=4.0,
            specialStateIDs=None,
            cmap='Set1',
            maxT=None,
            )
        SequenceViz.pylab.show(block=1)
        '''
        return Info

    def runMany_MOVBWithMoves(self,
                              initnames=['truelabels',
                                         'repeattruelabels',
                                         'truelabelsandempty'],
                              algName='moVB',
                              nWorkers=0,
                              moves='merge,delete,shuffle'):
        print('')
        for aKwArgs in self.nextAllocKwArgsForVB():
            for oKwArgs in self.nextObsKwArgsForVB():
                Info = dict()
                for iname in initnames:
                    if iname.count('junk') or iname.count('empty'):
                        initKextra = 1
                    else:
                        initKextra = 0
                    Info[iname] = self.run_MOVBWithMoves(
                        aKwArgs, oKwArgs,
                        moves=moves,
                        algName=algName,
                        nWorkers=nWorkers,
                        initKextra=initKextra,
                        initname=iname)

    def test_MOVBWithMerges(self):
        self.runMany_MOVBWithMoves(moves='merge')

    def test_MOVBWithDeletes(self):
        self.runMany_MOVBWithMoves(moves='delete')

    def test_MOVBWithMergeDeletes(self):
        self.runMany_MOVBWithMoves(moves='merge,delete')

    def test_MOVBWithShuffleMergeDeletes(self):
        self.runMany_MOVBWithMoves(moves='shuffle,merge,delete')

    def test_MOVBWithMerges_0ParallelWorkers(self):
        self.runMany_MOVBWithMoves(moves='merge', algName='pmoVB',
                                   nWorkers=0)

    def test_MOVBWithMerges_2ParallelWorkers(self):
        self.runMany_MOVBWithMoves(moves='merge', algName='pmoVB',
                                   nWorkers=2)

    def test_MOVBCreateDestroy_SingleSeq(self):
        print('')
        argDict = parseCmdLineArgs()
        for aKwArgs in self.nextAllocKwArgsForVB():
            for oKwArgs in self.nextObsKwArgsForVB():
                Info = dict()
                for iPattern in argDict['initnameVals'].split(','):
                    fields = iPattern.split('-')
                    initargs = dict()
                    for kvstr in fields:
                        kvpair = kvstr.split('=')
                        key = kvpair[0]
                        val = kvpair[1]
                        initargs[key] = val
                    initargs.update(argDict)
                    initargs['jobname'] = 'nosetest-initname=%s-K=%s' % (
                        initargs['initname'], initargs['K'])
                    self.run_MOVBWithMoves_SegmentSingleSeq(
                        aKwArgs, oKwArgs,
                        moves='merge,delete,shuffle,seqcreate',
                        **initargs)
                    print('')
                    print('')
                print('')
                print('')
                return

    def test_MOVBCreateDestroy_ManySeq(self):
        print('')
        argDict = parseCmdLineArgs()
        for aKwArgs in self.nextAllocKwArgsForVB():
            for oKwArgs in self.nextObsKwArgsForVB():
                Info = dict()
                for iPattern in argDict['initnameVals'].split(','):
                    fields = iPattern.split('-')
                    initargs = dict()
                    for kvstr in fields:
                        kvpair = kvstr.split('=')
                        key = kvpair[0]
                        val = kvpair[1]
                        initargs[key] = val
                    initargs.update(argDict)
                    initargs['jobname'] = 'nosetest-initname=%s-K=%s' % (
                        initargs['initname'], initargs['K'])
                    self.run_MOVBWithMoves_SegmentManySeq(
                        aKwArgs, oKwArgs,
                        moves='merge,delete,shuffle,seqcreate',
                        **initargs)
                    print('')
                    print('')
                print('')
                print('')
                return

    def interactivetest_findBestCut_SingleSeq(self, n=0, **kwargs):
        """ Interactively try out findBestCut.

        Post Condition
        --------------
        Will raise AssertionError if any bad results detected.
        """
        print('')
        argDict = parseCmdLineArgs()
        for aArg in self.nextAllocKwArgsForVB():
            for oArg in self.nextObsKwArgsForVB():
                for iPattern in argDict['initnameVals'].split(','):
                    fields = iPattern.split('-')
                    for kvstr in fields:
                        kvpair = kvstr.split('=')
                        key = kvpair[0]
                        val = kvpair[1]
                        argDict[key] = val
                    break
        if hasattr(self.Data, 'nDoc'):
            Data_n = self.Data.select_subset_by_mask(
                [n], doTrackTruth=1, doTrackFullSize=0)
            assert Data_n.nDocTotal == 1
        else:
            # Make GroupXData dataset from XData
            # This code block rearranges rows so that we
            # cycle thru the true labels twice as contig blocks.
            zTrue = self.Data.TrueParams['Z']
            half1 = list()
            half2 = list()
            for uID in np.unique(zTrue):
                dataIDs = np.flatnonzero(zTrue == uID)
                Nk = dataIDs.size
                half1.append(dataIDs[:Nk / 2])
                half2.append(dataIDs[Nk / 2:])
            dIDs_1 = np.hstack([x for x in half1])
            dIDs_2 = np.hstack([x for x in half2])
            dIDs = np.hstack([dIDs_1, dIDs_2])

            Data_n = bnpy.data.GroupXData(
                X=self.Data.X[dIDs],
                doc_range=np.asarray([0, self.Data.nObs]),
                TrueZ=self.Data.TrueParams['Z'][dIDs])

            aArg['name'] = 'HDPHMM'
            aArg['hmmKappa'] = 50
            aArg['alpha'] = 0.5
            aArg['gamma'] = 10.0
            aArg['startAlpha'] = 10.0

        Data_n.name = self.Data.name
        Data_n.alwaysTrackTruth = 1
        assert hasattr(Data_n, 'TrueParams')

        # Create and initialize model
        hmodel = bnpy.HModel.CreateEntireModel(
            'VB', aArg['name'], oArg['name'], aArg, oArg, Data_n)
        print(argDict)
        hmodel.init_global_params(Data_n, **argDict)

        # Run initial segmentation
        LP_n = hmodel.calc_local_params(Data_n)
        Z_n = LP_n['resp'].argmax(axis=1)
        Ztrue = np.asarray(Data_n.TrueParams['Z'], dtype=np.int32)
        Ktrue = np.max(Ztrue) + 1

        # Explore the findBestCut idea
        from matplotlib import pylab
        from bnpy.init.SeqCreateProposals import findBestCutForBlock
        while True:
            keypress = input("Enter start stop stride >>> ")
            fields = keypress.split(" ")
            if len(fields) < 2:
                break
            a = int(fields[0])
            b = int(fields[1])
            if len(fields) > 2:
                stride = int(fields[2])
            else:
                stride = 3
            m = findBestCutForBlock(Data_n, hmodel, a=a, b=b, stride=stride)
            print("Best Cut: [a=%d, m=%d, b=%d]" % (a, m, b))

            Kcur = LP_n['resp'].shape[1]
            Kmax_cur = np.maximum(Kcur, Ktrue)
            CMap = bnpy.util.StateSeqUtil.makeStateColorMap(
                nTrue=Kmax_cur, nExtra=2)

            Z_n_mod = Z_n.copy()
            Z_n_mod[a:m] = Kmax_cur
            Z_n_mod[m:b] = Kmax_cur + 1

            imshowArgs = dict(interpolation='nearest',
                              aspect=Z_n.size / 1.0,
                              cmap=CMap,
                              vmin=0, vmax=Kmax_cur + 1)

            pylab.close()
            pylab.subplots(nrows=3, ncols=1)
            ax = pylab.subplot(3, 1, 1)
            pylab.imshow(Ztrue[np.newaxis, :], **imshowArgs)
            pylab.yticks([])

            pylab.subplot(3, 1, 2, sharex=ax)
            pylab.imshow(Z_n[np.newaxis, :], **imshowArgs)
            pylab.yticks([])

            pylab.subplot(3, 1, 3, sharex=ax)
            pylab.imshow(Z_n_mod[np.newaxis, :], **imshowArgs)
            pylab.yticks([])

            L = b - a
            amin = np.maximum(0, a - L / 5)
            bmax = np.minimum(Z_n.size - 1, b + L / 5)
            pylab.xlim([amin, bmax])

            pylab.show(block=0)


def parseCmdLineArgs():
    cmdlineArgList = sys.argv[1:]
    argList = list()
    for aa, arg in enumerate(cmdlineArgList):
        if arg.startswith('-'):
            continue
        elif arg.count('.py'):
            continue
        argList.append(arg)

    assert len(argList) % 2 == 0
    argDict = dict()
    argDict['initnameVals'] = 'initname=randcontigblocks-K=1'
    for ii in range(0, len(argList), 2):
        key = argList[ii]
        val = argList[ii + 1]
        argDict[key] = val
    return argDict
