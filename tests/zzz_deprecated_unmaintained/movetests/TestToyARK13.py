import numpy as np
import unittest
from collections import OrderedDict

import bnpy
from MergeMoveEndToEndTest import MergeMoveEndToEndTest
from bnpy.init.SingleSeqStateCreator import initSingleSeq_SeqAllocContigBlocks


class Test(MergeMoveEndToEndTest):
    __test__ = True

    def setUp(self):
        """ Create the dataset
        """
        self.datasetArg = dict(
            name='ToyARK13',
            nDocTotal=26,
            T=400,
        )
        datasetMod = __import__(self.datasetArg['name'], fromlist=[])
        self.Data = datasetMod.get_data(**self.datasetArg)

    def nextObsKwArgsForVB(self):
        for sF in [0.1]:
            for ECovMat in ['eye']:
                kwargs = OrderedDict()
                kwargs['name'] = 'AutoRegGauss'
                kwargs['ECovMat'] = ECovMat
                kwargs['sF'] = sF
                kwargs['VMat'] = ECovMat
                kwargs['sV'] = sF
                yield kwargs

    def nextAllocKwArgsForVB(self):
        alpha = 0.5
        startAlpha = 10
        for gamma in [10.0]:
            for hmmKappa in [0, 50.0]:
                kwargs = OrderedDict()
                kwargs['name'] = 'HDPHMM'
                kwargs['gamma'] = gamma
                kwargs['alpha'] = alpha
                kwargs['hmmKappa'] = hmmKappa
                kwargs['startAlpha'] = startAlpha
                yield kwargs

    def test_initStateSeq(self):
        for aKwArgs in self.nextAllocKwArgsForVB():
            for oKwArgs in self.nextObsKwArgsForVB():
                hmodel = bnpy.HModel.CreateEntireModel(
                    'VB', aKwArgs['name'], oKwArgs['name'],
                    aKwArgs, oKwArgs, self.Data)
                SS = None
                for n in range(self.Data.nDoc):
                    SS, _, _ = initSingleSeq_SeqAllocContigBlocks(
                        n, self.Data, hmodel,
                        SS=SS,
                        verbose=2)
                assert SS.K >= self.Data.TrueParams['K']
