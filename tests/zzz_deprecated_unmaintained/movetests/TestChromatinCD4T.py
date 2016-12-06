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
        import scipy.io
        LVars = scipy.io.loadmat(
            '/data/liv/biodatasets/CD4TCellLine/batches/batch07.mat')
        self.Data = bnpy.data.GroupXData(**LVars)
        self.Data.name = 'ChromatinCD4T'
        self.datasetArg = dict(
            name=self.Data.name,
            nDocTotal=1,
            T=self.Data.doc_range[1],
        )

    def nextObsKwArgsForVB(self):
        for lam in [0.1]:
            kwargs = OrderedDict()
            kwargs['name'] = 'Bern'
            kwargs['lam1'] = lam
            kwargs['lam0'] = 3 * lam
            yield kwargs

    def nextAllocKwArgsForVB(self):
        alpha = 0.5
        startAlpha = 10
        for gamma in [10.0]:
            for hmmKappa in [50.0, 0]:
                kwargs = OrderedDict()
                kwargs['name'] = 'HDPHMM'
                kwargs['gamma'] = gamma
                kwargs['alpha'] = alpha
                kwargs['hmmKappa'] = hmmKappa
                kwargs['startAlpha'] = startAlpha
                yield kwargs
