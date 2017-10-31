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
        vocab_size = 25
        nDoc = 10
        for d in range(nDoc):
            NU_d = rng.choice(vocab_size)
            NU_d = np.maximum(1, NU_d)
            word_id_d = rng.choice(vocab_size, size=NU_d, replace=False)
            word_ct_d = rng.choice(list(range(1, 10)), size=NU_d, replace=True)
            if d == 0:
                word_id = word_id_d
                word_ct = word_ct_d
                doc_range = np.asarray([0, NU_d], dtype=np.int32)
            else:
                word_id = np.hstack([word_id, word_id_d])
                word_ct = np.hstack([word_ct, word_ct_d])
                doc_range = np.hstack([doc_range, doc_range[-1] + NU_d])

        self.Data = bnpy.data.BagOfWordsData(word_id=word_id,
                                        word_count=word_ct,
                                        doc_range=doc_range,
                                        vocab_size=vocab_size)

        self.possibleAllocModelNames = ["FiniteMixtureModel",
                                        "DPMixtureModel",
                                        "FiniteTopicModel",
                                        "HDPTopicModel",
                                        ]
        self.possibleObsModelNames = ["Mult",
                                      ]
        self.possibleInitNames = ["randexamples",
                                  "randomlikewang",
                                  "randomfromprior",
                                  ]

        self.possibleLearnAlgsForAllocModel = dict(
            FiniteMixtureModel=["EM", "VB", "soVB", "moVB"],
            DPMixtureModel=["VB", "soVB", "moVB"],
            FiniteTopicModel=["VB", "soVB", "moVB"],
            HDPTopicModel=["VB", "soVB", "moVB"],
        )

    def nextObsKwArgsForVB(self, aName):
        for oName in self.possibleObsModelNames:
            for lam in [0.01, 1.0, 10.0]:
                kwargs = OrderedDict()
                kwargs['name'] = oName
                kwargs['lam'] = lam
                yield kwargs

    def nextInitKwArgs(self, aName, oName):
        for iName in self.possibleInitNames:
            for K in [1, 2, 10]:
                kwargs = OrderedDict()
                kwargs['initname'] = iName
                kwargs['K'] = K
                kwargs['initMinWordsPerDoc'] = 0
                yield kwargs

    def makeInitKwArgs(self, initname):
        kwargs = dict(
            initname=initname,
            initMinWordsPerDoc=0,
            K=3,
        )
        return kwargs
