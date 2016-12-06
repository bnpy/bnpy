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
            name='AsteriskK8',
            nObsTotal=16000,
        )
        datasetMod = __import__(self.datasetArg['name'], fromlist=[])
        self.Data = datasetMod.get_data(**self.datasetArg)

        import AsteriskK8
        self.Data = AsteriskK8.get_data(nObsTotal=16000)

    def nextObsKwArgsForVB(self):
        for sF in [0.5, 5.0]:
            for ECovMat in ['eye', 'covdata']:
                kwargs = OrderedDict()
                kwargs['name'] = 'Gauss'
                kwargs['ECovMat'] = ECovMat
                kwargs['sF'] = sF
                yield kwargs

    def nextAllocKwArgsForVB(self):
        for gamma in [1.0, 50.0]:
            kwargs = OrderedDict()
            kwargs['name'] = 'DPMixtureModel'
            kwargs['gamma0'] = gamma
            yield kwargs
