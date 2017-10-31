import unittest
import scipy.io
import numpy as np

import bnpy

runViterbiAlg = bnpy.allocmodel.hmm.HMMUtil.runViterbiAlg
runViterbiAlg_forloop = bnpy.allocmodel.hmm.HMMUtil.runViterbiAlg_forloop

INPUTmatfile = 'kmurphy-viterbi-reference/ViterbiTestInput.mat'
OUTPUTmatfile = 'kmurphy-viterbi-reference/ViterbiTestOutput.mat'


def pprintZ(zHat):
    print(zHat[:10] + 1)
    print(zHat[10:20] + 1)
    print(zHat[20:30] + 1)
    print(zHat[-10:] + 1)


class TestViterbi(unittest.TestCase):

    def setUp(self):
        # Load input problem: logSoftEv, logPiTrans, logPiInit
        Q = scipy.io.loadmat(INPUTmatfile)
        Q['logPiInit'] = np.squeeze(Q['logPiInit'])
        self.MATdict = Q

        # Load expected solution (as computed by K Murphy's ref implementation)
        OUTdict = scipy.io.loadmat(OUTPUTmatfile)
        self.zTrue = OUTdict['zHat'] - 1  # Transform to 0-indexed array

    def test_runViterbiAlg(self):
        print('')
        Q = self.MATdict
        zHat = runViterbiAlg(Q['logEvidence'], Q['logPiInit'], Q['logPiTrans'])
        pprintZ(zHat)
        assert np.allclose(zHat, self.zTrue)

    def test_runViterbiAlg_forloop(self):
        print('')
        Q = self.MATdict
        zHat = runViterbiAlg_forloop(
            Q['logEvidence'],
            Q['logPiInit'],
            Q['logPiTrans'])
        pprintZ(zHat)
        assert np.allclose(zHat, self.zTrue)
