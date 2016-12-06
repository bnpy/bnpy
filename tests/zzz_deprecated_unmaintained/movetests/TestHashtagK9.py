import numpy as np
import unittest
from collections import OrderedDict

import bnpy
from MergeMoveEndToEndTest import MergeMoveEndToEndTest


class Test(MergeMoveEndToEndTest):
    __test__ = True

    def setUp(self):
        """ Create the dataset
        """
        self.datasetArg = dict(
            name='HashtagK9',
            nObsTotal=10000,
        )
        datasetMod = __import__(self.datasetArg['name'], fromlist=[])
        self.Data = datasetMod.get_data(**self.datasetArg)

    def nextObsKwArgsForVB(self):
        for sF in [0.5]:
            for ECovMat in ['covdata']:
                kwargs = OrderedDict()
                kwargs['name'] = 'DiagGauss'
                kwargs['ECovMat'] = ECovMat
                kwargs['sF'] = sF
                yield kwargs

    def nextAllocKwArgsForVB(self):
        for gamma in [5.0]:
            kwargs = OrderedDict()
            kwargs['name'] = 'DPMixtureModel'
            kwargs['gamma0'] = gamma
            yield kwargs
