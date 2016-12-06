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
            name='BarsK10V900',
            nDocTotal=100,
            nWordsPerDoc=100,
        )
        datasetMod = __import__(self.datasetArg['name'], fromlist=[])
        self.Data = datasetMod.get_data(**self.datasetArg)

    def nextObsKwArgsForVB(self):
        for lam in [0.1, 0.01]:
            kwargs = OrderedDict()
            kwargs['name'] = 'Mult'
            kwargs['lam'] = lam
            yield kwargs

    def nextAllocKwArgsForVB(self):
        alpha = 0.5
        for gamma in [1.0, 50.0]:
            kwargs = OrderedDict()
            kwargs['name'] = 'HDPTopicModel'
            kwargs['gamma'] = gamma
            kwargs['alpha'] = alpha
            yield kwargs
