import numpy as np
import scipy.io
import unittest

from bnpy.util import as1D
from bnpy.util import StateSeqUtil as SSU


class TestStateSeqUtil(unittest.TestCase):

    def setUp(self):
        pass

    def shortDescription(self):
        pass

    def test_alignEstStateSeqToTrue__Zest_equals_Ztrue(self):
        ''' Verify alignment works when both sequences match exactly
        '''
        zEst = [0, 0, 0, 1, 1, 1, 1, 1]
        zTru = [0, 0, 0, 1, 1, 1, 1, 1]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zTru)

        zEst = [1, 1, 1, 0, 0, 0, 0, 0]
        zTru = [0, 0, 0, 1, 1, 1, 1, 1]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zTru)

        zEst = [2, 2, 2, 0, 0, 0, 0, 0]
        zTru = [0, 0, 0, 1, 1, 1, 1, 1]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zTru)

        zEst = [2, 2, 2, 0, 0, 0, 0, 0]
        zTru = [3, 3, 3, 1, 1, 1, 1, 1]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zTru)

    def test_alignEstStateSeqToTrue__Kest_equals_Ktrue(self):
        ''' Verify alignment works when both sequences have same number of states
        '''
        print('')

        zEst = [0, 0, 0, 1, 1, 1, 1, 1]
        zTru = [0, 0, 1, 1, 1, 1, 1, 0]
        zExp = [0, 0, 0, 1, 1, 1, 1, 1]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zExp)

        zEst = [1, 1, 1, 0, 0, 0, 0, 0]
        zTru = [0, 0, 1, 1, 1, 1, 1, 0]
        zExp = [0, 0, 0, 1, 1, 1, 1, 1]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zExp)

        zEst = [0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1]
        zTru = [1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0]
        zExp = [0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zExp)

        zEst = [2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        zTru = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0]
        zExp = [1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zExp)

    def test_alignEstStateSeqToTrue__Kest_lt_Ktrue(self):
        ''' Verify alignment works when est sequence has fewer states than true
        '''
        print('')

        zEst = [0, 0, 0, 0, 0, 0, 0, 0]
        zTru = [0, 0, 1, 1, 1, 1, 1, 0]
        zExp = [1, 1, 1, 1, 1, 1, 1, 1]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zExp)

        zEst = [0, 0, 0, 0, 0, 0, 0, 0]
        zTru = [0, 0, 1, 1, 1, 1, 1, 2]
        zExp = [1, 1, 1, 1, 1, 1, 1, 1]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zExp)

        zEst = [0, 1, 1, 1, 0, 0, 0, 1]
        zTru = [0, 0, 1, 1, 1, 1, 1, 2]
        zExp = [1, 0, 0, 0, 1, 1, 1, 0]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)

        distA = SSU.calcHammingDistance(zA, zTru)
        distExp = SSU.calcHammingDistance(zExp, zTru)

    def test_alignEstStateSeqToTrue__Kest_gt_Ktrue(self):
        ''' Verify alignment works when est sequence has more states than true
        '''
        print('')

        zEst = [0, 0, 0, 1, 1, 2, 0, 0]
        zTru = [0, 0, 0, 0, 0, 0, 0, 0]
        zExp = [0, 0, 0, 1, 1, 2, 0, 0]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zExp)

        zEst = [2, 2, 2, 1, 1, 0, 2, 2]
        zTru = [0, 0, 0, 0, 0, 0, 0, 0]
        zExp = [0, 0, 0, 1, 1, 2, 0, 0]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        distA = SSU.calcHammingDistance(zA, zTru)
        distExp = SSU.calcHammingDistance(zExp, zTru)
        assert distA == distExp

        zEst = [0, 0, 1, 2, 3, 4, 5, 6]
        zTru = [0, 0, 0, 1, 1, 1, 2, 2]
        zExp = [0, 0, 5, 1, 3, 4, 2, 6]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        distA = SSU.calcHammingDistance(zA, zTru)
        distExp = SSU.calcHammingDistance(zExp, zTru)
        assert distA == distExp

        zEst = [6, 6, 0, 5, 4, 3, 2, 1]
        zTru = [0, 0, 0, 1, 1, 1, 2, 2]
        zExp = [0, 0, 5, 1, 3, 4, 2, 6]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        distA = SSU.calcHammingDistance(zA, zTru)
        distExp = SSU.calcHammingDistance(zExp, zTru)
        assert distA == distExp

    def test_alignEstStateSeqToTrue__Kest_gt_Ktrue_someempty(self):
        ''' Verify alignment works when est sequence has more states than true
        '''
        print('')

        zEst = [1, 1, 1, 2, 2, 1, 1, 1]
        zTru = [0, 0, 0, 0, 0, 0, 0, 0]
        zExp = [0, 0, 0, 2, 2, 0, 0, 0]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zExp)
        hdist = SSU.calcHammingDistance(zA, zTru)
        print(hdist)
        assert hdist == 2 / float(len(zTru))

        zEst = [2, 2, 2, 3, 4, 5, 2, 2]
        zTru = [0, 0, 0, 0, 0, 0, 0, 0]
        zExp = [0, 0, 0, 3, 4, 5, 0, 0]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        assert np.allclose(zA, zExp)
        hdist = SSU.calcHammingDistance(zA, zTru)
        print(hdist)
        assert hdist == 3  / float(len(zTru))

        zEst = [2, 2, 2, 3, 4, 5, 2, 2]
        zTru = [1, 1, 0, 0, 0, 0, 0, 0]
        zExp = [0, 0, 0, 4, 5, 1, 0, 0]
        zA = SSU.alignEstimatedStateSeqToTruth(zEst, zTru)
        print(zA)
        assert np.allclose(zA, zExp)
        hdist = SSU.calcHammingDistance(zA, zTru)
        assert hdist == 5  / float(len(zTru))
