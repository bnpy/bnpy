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
            name='AdmixAsteriskK8',
            nDocTotal=200,
            nObsPerDoc=100,
        )
        datasetMod = __import__(self.datasetArg['name'], fromlist=[])
        self.Data = datasetMod.get_data(**self.datasetArg)

    def nextObsKwArgsForVB(self):
        for sF in [0.5]:
            for ECovMat in ['covdata']:
                kwargs = OrderedDict()
                kwargs['name'] = 'Gauss'
                kwargs['ECovMat'] = ECovMat
                kwargs['sF'] = sF
                yield kwargs

    def nextAllocKwArgsForVB(self):
        alpha = 0.5
        for gamma in [5.0]:
            kwargs = OrderedDict()
            kwargs['name'] = 'HDPTopicModel'
            kwargs['gamma'] = gamma
            kwargs['alpha'] = alpha
            yield kwargs
