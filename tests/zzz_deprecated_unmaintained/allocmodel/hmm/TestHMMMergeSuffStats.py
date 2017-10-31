'''
TestHMMMergeSuffStats

Verifies that the sufficient statistics involved in the allocation model
are treated properly when merging.
'''

import numpy as np
import unittest
import bnpy

M_K2 = np.asarray([
    [100, 200],
    [40, 30],
])
mM_K2 = np.asarray([[370]])

N_K4 = np.asarray([1, 2, 3, 4])
M_K4 = np.asarray([
    [10, 21, 30, 41],
    [11, 20, 31, 40],
    [10, 21, 30, 40],
    [11, 20, 30, 40],
])

# Expected result for merging 1st & 2nd rows
mN_K4_12 = np.asarray([3, 3, 4])
mM_K4_12 = np.asarray([
    [62, 61, 81],
    [31, 30, 40],
    [31, 30, 40],
])

# Expected result for merging 2nd and 4th rows
mN_K4_24 = np.asarray([1, 6, 3])
mM_K4_24 = np.asarray([
    [10, 62, 30],
    [22, 120, 61],
    [10, 61, 30],
])

# Expected result for merging 1st & 3rd rows
mM_K4_13 = np.asarray([
    [80, 42, 81],
    [42, 20, 40],
    [41, 20, 40],
])

# Expected result for merging 3rd & 4th rows
mM_K4_34 = np.asarray([
    [10, 21, 71],
    [11, 20, 71],
    [21, 41, 140],
])

# Expected result for merging 3rd & 4th rows, then 1st&2nd rows
mM_K4_3412 = np.asarray([
    [62, 142],
    [62, 140],
])


class TestHMMMergeSuffStats(unittest.TestCase):

    def shortDescription(self):
        pass

    def setUp(self):
        self.origSS_K2 = bnpy.suffstats.SuffStatBag(K=2)
        self.origSS_K2.setField('M', M_K2, dims=('K', 'K'))

        self.origSS_K4 = bnpy.suffstats.SuffStatBag(K=4)
        self.origSS_K4.setField('M', M_K4, dims=('K', 'K'))
        self.origSS_K4.setField('N', N_K4, dims=('K'))

    def test_K2__expected_merges_have_same_sum(self):
        sum = self.origSS_K2.M.sum()
        sum2 = mM_K2.sum()
        assert np.allclose(sum, sum2)

    def test_K4__expected_merges_have_same_sum(self):
        sum = self.origSS_K4.M.sum()
        sum12 = mM_K4_12.sum()
        sum13 = mM_K4_13.sum()
        sum34 = mM_K4_34.sum()
        sum24 = mM_K4_34.sum()

        sum3412 = mM_K4_3412.sum()
        assert np.allclose(sum, sum12)
        assert np.allclose(sum, sum34)
        assert np.allclose(sum, sum13)
        assert np.allclose(sum, sum24)
        assert np.allclose(sum, sum3412)

    def test_K2__merge_equals_expected(self):
        print('')
        propSS = self.origSS_K2.copy()
        propSS.mergeComps(0, 1)

        print(propSS.M)
        print(mM_K2)
        assert np.allclose(propSS.M, mM_K2)

    def test_K4_12__merge_equals_expected(self):
        print('')
        propSS = self.origSS_K4.copy()
        propSS.mergeComps(0, 1)

        print(propSS.M)
        print(mM_K4_12)
        assert np.allclose(propSS.M, mM_K4_12)

        print(propSS.N)
        print(propSS.N.shape)

        assert propSS.N.shape[0] == 3
        assert np.allclose(propSS.N, mN_K4_12)

    def test_K4_13__merge_equals_expected(self):
        print('')
        propSS = self.origSS_K4.copy()
        propSS.mergeComps(0, 2)

        print(propSS.M)
        print(mM_K4_13)
        assert np.allclose(propSS.M, mM_K4_13)

    def test_K4_24__merge_equals_expected(self):
        print('')
        propSS = self.origSS_K4.copy()
        propSS.mergeComps(1, 3)

        print(propSS.M)
        print(mM_K4_24)
        assert np.allclose(propSS.M, mM_K4_24)
        assert propSS.N.shape[0] == 3
        assert np.allclose(propSS.N, mN_K4_24)

    def test_K4_34__merge_equals_expected(self):
        print('')
        propSS = self.origSS_K4.copy()
        propSS.mergeComps(2, 3)

        print(propSS.M)
        print(mM_K4_34)
        assert np.allclose(propSS.M, mM_K4_34)

    def test_K4_34then12__merge_equals_expected(self):
        ''' Verify merge followed by another merge works as expected
        '''
        print('')
        propSS = self.origSS_K4.copy()
        propSS.mergeComps(2, 3)
        propSS.mergeComps(0, 1)

        print(propSS.M)
        print(mM_K4_3412)
        assert np.allclose(propSS.M, mM_K4_3412)
