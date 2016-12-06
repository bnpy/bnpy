import numpy as np
import unittest
from collections import OrderedDict

import bnpy
from AbstractEndToEndTest import AbstractEndToEndTest


class TestEndToEnd(AbstractEndToEndTest):
    __test__ = True

    def setUp(self):
        """ Create the dataset
        """
        rng = np.random.RandomState(0)
        X = np.asarray(rng.rand(100, 2) > 0.75,
                       dtype=np.float64)
        self.Data = bnpy.data.XData(X=X)
        self.possibleAllocModelNames = ["FiniteMixtureModel",
                                        "DPMixtureModel",
                                        ]
        self.possibleObsModelNames = ["Bern",
                                      ]
        self.possibleInitNames = ["randexamples",
                                  "randexamplesbydist",
                                  ]

        self.possibleLearnAlgsForAllocModel = dict(
            FiniteMixtureModel=["EM", "VB", "soVB", "moVB"],
            DPMixtureModel=["VB", "soVB", "moVB"],
        )

    def nextObsKwArgsForVB(self, aName):
        for oName in self.possibleObsModelNames:
            for lam0 in [0.01, 1.0, 10.0]:
                for lam1 in [0.01, 0.5]:
                    kwargs = OrderedDict()
                    kwargs['name'] = oName
                    kwargs['lam1'] = lam1
                    kwargs['lam0'] = lam0
                    yield kwargs

    def nextInitKwArgs(self, aName, oName):
        for iName in self.possibleInitNames:
            for K in [1, 2, 10]:
                kwargs = OrderedDict()
                kwargs['initname'] = iName
                kwargs['K'] = K
                yield kwargs
