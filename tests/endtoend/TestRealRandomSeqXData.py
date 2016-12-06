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
        X = rng.rand(17, 3)
        doc_range = [0, 1, 3, 17]
        self.Data = bnpy.data.GroupXData(X=X, doc_range=doc_range)

        self.possibleAllocModelNames = ["FiniteHMM",
                                        "HDPHMM",
                                        ]
        self.possibleObsModelNames = ["Gauss",
                                      "DiagGauss",
                                      ]
        self.possibleInitNames = ["randexamples",
                                  "kmeans",
                                  ]

        self.possibleLearnAlgsForAllocModel = dict(
            FiniteHMM=["EM", "VB", "soVB", "moVB"],
            HDPHMM=["VB", "soVB", "moVB"],
        )

    def nextObsKwArgsForVB(self, aName):
        for oName in self.possibleObsModelNames:
            for nu in [5, 10]:
                for ECovMat in ['covdata']:
                    kwargs = OrderedDict()
                    kwargs['name'] = oName
                    kwargs['ECovMat'] = ECovMat
                    kwargs['nu'] = nu
                    yield kwargs

    def nextInitKwArgs(self, aName, oName):
        for iName in self.possibleInitNames:
            for K in [3, 11]:
                kwargs = OrderedDict()
                kwargs['initname'] = iName
                kwargs['K'] = K
                yield kwargs
