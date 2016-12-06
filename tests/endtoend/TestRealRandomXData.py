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
        X = rng.rand(100, 2)
        self.Data = bnpy.data.XData(X=X)

        self.possibleAllocModelNames = ["FiniteMixtureModel",
                                        "DPMixtureModel",
                                        ]
        self.possibleObsModelNames = ["Gauss",
                                      "DiagGauss",
                                      "ZeroMeanGauss",
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
            for sF in [0.5, 1.0, 5.0]:
                for ECovMat in ['eye', 'covdata']:
                    kwargs = OrderedDict()
                    kwargs['name'] = oName
                    kwargs['ECovMat'] = ECovMat
                    kwargs['sF'] = sF
                    yield kwargs

    def nextInitKwArgs(self, aName, oName):
        for iName in self.possibleInitNames:
            for K in [1, 2, 10]:
                kwargs = OrderedDict()
                kwargs['initname'] = iName
                kwargs['K'] = K
                yield kwargs
