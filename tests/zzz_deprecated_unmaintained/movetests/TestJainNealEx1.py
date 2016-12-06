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
            name='JainNealEx1',
            nPerState=100,
            nObsTotal=500,
        )
        datasetMod = __import__(self.datasetArg['name'], fromlist=[])
        self.Data = datasetMod.get_data(**self.datasetArg)

    def nextObsKwArgsForVB(self):
        for lam in [0.5, 0.01]:
            kwargs = OrderedDict()
            kwargs['name'] = 'Bern'
            kwargs['lam1'] = lam
            kwargs['lam0'] = lam
            yield kwargs

    def nextAllocKwArgsForVB(self):
        for gamma in [1.0, 50.0]:
            kwargs = OrderedDict()
            kwargs['name'] = 'DPMixtureModel'
            kwargs['gamma0'] = gamma
            yield kwargs
