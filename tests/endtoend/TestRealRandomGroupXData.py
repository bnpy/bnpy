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
        doc_range = [0, 20, 40, 50, 100]
        self.Data = bnpy.data.GroupXData(X=X, doc_range=doc_range)

        self.possibleAllocModelNames = ["FiniteMixtureModel",
                                        "FiniteTopicModel",
                                        "HDPTopicModel",
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
            FiniteTopicModel=["VB", "soVB", "moVB"],
            HDPTopicModel=["VB", "soVB", "moVB"],
        )

    def nextAllocKwArgsForVB(self):
        for aName in self.possibleAllocModelNames:
            kwargs = OrderedDict()
            kwargs['name'] = aName
            if aName == 'FiniteMixtureModel':
                for gamma in [0.1, 1.0, 9.9]:
                    kwargs['gamma'] = gamma
                    yield kwargs
            elif aName == 'DPMixtureModel':
                for gamma0 in [1.0, 9.9]:
                    kwargs['gamma0'] = gamma0
                    yield kwargs
            elif aName == 'FiniteTopicModel':
                for alpha in [0.1, 0.5, 22]:
                    kwargs['alpha'] = alpha
                    yield kwargs
            elif aName == 'HDPTopicModel':
                for alpha in [0.1, 0.5]:
                    for gamma in [1.0, 5.0]:
                        kwargs['gamma'] = gamma
                        yield kwargs

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
            for K in [5, 10]:
                kwargs = OrderedDict()
                kwargs['initname'] = iName
                kwargs['K'] = K
                yield kwargs
