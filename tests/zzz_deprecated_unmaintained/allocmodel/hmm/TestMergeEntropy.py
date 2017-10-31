import unittest
import scipy.io
import numpy as np
np.set_printoptions(precision=2, suppress=1)

import bnpy
from bnpy.allocmodel.hmm import HMMUtil

import ToyHMMK4


class TestMergeEntropyCalc_EndToEnd(unittest.TestCase):

    ''' This test suite verifies that when using HDPHMM and SuffStats
        to track the entropy terms, we can exactly calculate the entropy
    '''

    def shortDescription(self):
        return None

    def setUp(self):
        ''' Create a valid Data - model - LP - SS configuration
        '''
        # Make toy data
        Data = ToyHMMK4.get_data(12345, T=15, nDocTotal=3)
        self.Data = Data

        hmodel, Info = bnpy.run(Data, 'HDPHMM', 'Gauss', 'VB',
                                nLap=1, K=6, initname='randexamplesbydist',
                                alpha=0.5, gamma=5.0,
                                ECovMat='eye', sF=1.0, kappa=1e-5,
                                doWriteStdOut=False, doSaveToDisk=False)
        LP = hmodel.calc_local_params(Data, limitMemoryLP=0)
        assert 'mHtable' not in LP

        self.mPairIDs = [(0, 1), (2, 3), (4, 5), (1, 5), (3, 4)]
        SS = hmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1,
                                          doPrecompMergeEntropy=1,
                                          mPairIDs=self.mPairIDs)
        hmodel.update_global_params(SS)
        self.hmodel = hmodel
        self.origLP = LP
        self.origSS = SS.copy()

    def test__ELBOfromSS_equals_ELBOfromLP(self):
        ''' Verify that ELBO calculation is identical regardless of method used

            Method 1: directly from local parameters, no precomputed ELBO terms
            Method 2: use precomputed tables in SuffStatsBag
        '''
        print('')
        ELBOfromSS = self.hmodel.calc_evidence(SS=self.origSS)
        plainSS = self.hmodel.get_global_suff_stats(self.Data, self.origLP)
        ELBOfromLP = self.hmodel.calc_evidence(self.Data, plainSS, self.origLP)
        assert ELBOfromSS == ELBOfromLP
        assert np.allclose(ELBOfromSS, ELBOfromLP)
        assert not plainSS.hasELBOTerms()

    def test__mergeELBOfromSS_equals_mergeELBOfromLP(self):
        ''' Loop over all tracked merge pairs, and verify ELBO is same
            regardless of which method is used
        '''
        for mID in range(len(self.mPairIDs)):
            self.verify__mergeELBOfromSS_equals_mergeELBOfromLP(mID)

    def verify__mergeELBOfromSS_equals_mergeELBOfromLP(self, mID=0):
        ''' Verify identical ELBO value from two methods for single merge pair

        Method 1: Using tables in suff stats
        Method 2: Using direct construction of the local parameters
        '''
        print('')

        kA = self.mPairIDs[mID][0]
        kB = self.mPairIDs[mID][1]
        print('@@@ kA=%d kB=%d' % (kA, kB))

        # Method 1: using suff stats tables
        #
        propSS = self.origSS.copy()
        propSS.mergeComps(kA, kB)
        propM = self.hmodel.copy()
        propM.update_global_params(propSS)
        propELBO_fromSS = propM.calc_evidence(SS=propSS)
        assert np.isfinite(propELBO_fromSS)
        print('propELBO via manipulation of tables stored in SS')
        print(propELBO_fromSS)

        # Method 2 : direct construction of resp, respPair for candidate
        #
        Knew = self.origSS.K - 1
        newLP = HMMUtil.construct_LP_forMergePair(
            self.Data,
            self.origLP,
            kA,
            kB)
        newSS = self.hmodel.get_global_suff_stats(self.Data, newLP)
        newM = self.hmodel.copy()
        newM.update_global_params(newSS)
        propELBO_fromLP = newM.calc_evidence(self.Data, newSS, newLP)
        assert np.isfinite(propELBO_fromLP)
        print('propELBO via direct construction of LP')
        print(propELBO_fromLP)

        assert np.allclose(propELBO_fromLP, propELBO_fromSS)
        assert np.allclose(newSS.N, propSS.N)
        assert np.allclose(newSS.TransStateCount, propSS.TransStateCount)

    def test__twomergemoveELBOfromSS_lt_twomergemoveELBOfromLP(self):
        ''' Test many possible events of two merges attempted in succession.

            Requirement: The components involved in both merges must be distinct!
        '''
        M = len(self.mPairIDs)
        # Try all possible tuples of merge pairs
        for mID in range(M):
            for mID2 in range(M):
                if mID == mID2:
                    continue
                if self.mPairIDs[mID][0] in self.mPairIDs[mID2]:
                    continue
                if self.mPairIDs[mID][1] in self.mPairIDs[mID2]:
                    continue
                self.verify__twomergemoveELBOfromSS_lt_twomergemoveELBOfromLP(
                    mID,
                    mID2)

    def verify__twomergemoveELBOfromSS_lt_twomergemoveELBOfromLP(
            self, mID=0, mID2=1):
        ''' Verify ELBO calc when two merges are attempted in succession.

        Requirement: The components involved in both merges must be distinct!

        Method 1: Using tables in suff stats
        Method 2: Using direct construction of the local parameters
        '''
        print('')

        kA = self.mPairIDs[mID][0]
        kB = self.mPairIDs[mID][1]
        kA2 = self.mPairIDs[mID2][0]
        kB2 = self.mPairIDs[mID2][1]

        assert len(np.unique([kA, kB, kA2, kB2])) == 4
        print('@@@ kA=%d kB=%d kA2=%d kB2=%d' % (kA, kB, kA2, kB2))

        if kB < kA2:
            kA2 -= 1
        if kB < kB2:
            kB2 -= 1

        # Method 1: using suff stats tables
        #
        propSS = self.origSS.copy()
        propSS.mergeComps(kA, kB)
        propSS.mergeComps(kA2, kB2)

        propM = self.hmodel.copy()
        propM.update_global_params(propSS)
        propELBO_fromSS = propM.calc_evidence(SS=propSS)
        assert np.isfinite(propELBO_fromSS)
        print('propELBO via manipulation of tables stored in SS')
        print(propELBO_fromSS)

        # Method 2 : direct construction of resp, respPair for candidate
        #
        Knew = self.origSS.K - 1
        newLP = HMMUtil.construct_LP_forMergePair(
            self.Data,
            self.origLP,
            kA,
            kB)
        newLP = HMMUtil.construct_LP_forMergePair(self.Data, newLP, kA2, kB2)
        newSS = self.hmodel.get_global_suff_stats(self.Data, newLP)
        newM = self.hmodel.copy()
        newM.update_global_params(newSS)
        propELBO_fromLP = newM.calc_evidence(self.Data, newSS, newLP)
        assert np.isfinite(propELBO_fromLP)
        print('propELBO via direct construction of LP')
        print(propELBO_fromLP)

        assert propELBO_fromLP >= propELBO_fromSS
        assert np.allclose(newSS.N, propSS.N)
        assert np.allclose(newSS.TransStateCount, propSS.TransStateCount)

        print('Htable from suff stats... should have some zero entries')
        print(propSS.getELBOTerm('Htable'))

    def test__merge_yields_valid_suff_stats_object(self):
        ''' Loop over each possible merge pair that we've tracked,
            and verify that we can merge successfully with that pair
        '''
        for mID in range(len(self.mPairIDs)):
            self.verify__merge_yields_valid_suff_stats_object(mID)

    def verify__merge_yields_valid_suff_stats_object(self, mID=0):
        ''' Verify that a single merge pair yields a valid SuffStatBag

            "Valid" means that we test for
            * expected K
            * expected shapes of various fields
        '''
        print('')

        kA = self.mPairIDs[mID][0]
        kB = self.mPairIDs[mID][1]
        propSS = self.origSS.copy()
        propSS.mergeComps(kA, kB)

        Korig = self.origSS.K
        Knew = Korig - 1
        assert propSS.K == Knew

        print(self.origSS.getELBOTerm('Hstart'))
        print(propSS.getELBOTerm('Hstart'))

        assert propSS.getELBOTerm('Hstart').ndim == 1
        assert propSS.getELBOTerm('Hstart').size == Knew
        assert propSS.getELBOTerm('Htable').shape == (Knew, Knew)

        origHtable = self.origSS.getELBOTerm('Htable')
        propHtable = propSS.getELBOTerm('Htable')
        assert np.allclose(origHtable[kB + 1:, kB + 1:],
                           propHtable[kB:, kB:])

        Mtable = self.origSS.getMergeTerm('Htable')[mID]
        newHtable = HMMUtil.calc_Htable_forMergePair_fromTables(origHtable,
                                                                Mtable, kA, kB)

        print('newHTable calculated by SuffStatBag.mergeComps')
        print(propHtable)

        print('newHtable from HMMUtil')
        print(newHtable)
        assert np.allclose(propHtable,
                           newHtable
                           )

        Mnew = propSS.mPairIDs.shape[0]
        assert propSS.getMergeTerm('Hstart').size == Mnew

        assert propSS.getMergeTerm('Htable').shape[0] == Mnew
        assert propSS.getMergeTerm('Htable').shape[1] == 2
        assert propSS.getMergeTerm('Htable').shape[2] == Knew


class TestMergeEntropyCalc_SingleTimeSlice(unittest.TestCase):

    ''' This test suite verifies that we can calculate the entropy
        after a merge effectively, for a single time-slice of respPair
    '''

    def shortDescription(self):
        return None

    def setUp(self):
        respPair = np.asarray([
            [.01, .02, .03, .14],
            [.11, .02, .13, .04],
            [.01, .12, .03, .04],
            [.11, .02, .13, .04],
        ])
        self.respPair = respPair[np.newaxis, :, :].copy()

    def test_Lentropy__directMethod_equals_tableMethod(self):
        ''' Test for all pairs (kA, kB) whether L_entropy is same from two methods
        '''
        print('')
        K = self.respPair.shape[1]
        for kA in range(K):
            for kB in range(kA + 1, K):
                self.verify_Lentropy__directMethod_equals_tableMethod(kA, kB)

    def verify_Lentropy__directMethod_equals_tableMethod(self, kA, kB):
        ''' Worker method that does both methods for a single pair (kA, kB)
        '''
        respPair = self.respPair
        # Method 1/2
        # Directly construct s, sigma for merge,
        # then compute entropy from these local parameters
        L_direct = HMMUtil.calc_sumHtable_forMergePair__fromResp(
            respPair,
            kA,
            kB)

        # Method 2/2
        # Compute table of scalars for original
        # Also compute O(K) replacement entries for table required by merge
        # then, just do some table operations and sums to get final L_entropy
        Htable_orig = HMMUtil.calc_Htable(respPair)
        Mtable = HMMUtil.calc_sub_Htable_forMergePair(respPair, kA, kB)
        L_table = HMMUtil.calc_sumHtable_forMergePair__fromTables(
            Htable_orig, Mtable, kA, kB)
        assert np.allclose(L_table, L_direct)
        print(kA, kB, L_table, L_direct)


class TestMergeEntropyCalc_FullSequence(TestMergeEntropyCalc_SingleTimeSlice):

    ''' This test suite verifies that we can calculate the entropy
        after a merge effectively, for a single time-slice of respPair
    '''

    def setUp(self):
        self.T = 50
        self.K = 5
        PRNG = np.random.RandomState(12345)
        self.s = PRNG.rand(self.T, self.K, self.K)
        self.s /= self.s.sum(axis=2).sum(axis=1)[:, np.newaxis, np.newaxis]
        self.respPair = self.s

    # all other tests inherited from parent class!


class TestIsNotTheSame(unittest.TestCase):

    ''' This is just a quick test to verify that we can't calculate entropy
        like we've done before, and we need to use the respPair method

        Just a quick test to void some fears I had that we were making this too
        hard on ourselves
    '''

    def shortDescription(self):
        return None

    def setUp(self):
        PRNG = np.random.RandomState(101)
        self.s = np.zeros((3, 5, 5))
        self.s[0] = PRNG.rand(5, 5)
        self.s[1] = PRNG.rand(5, 5)
        self.s[2] = PRNG.rand(5, 5)
        self.s /= self.s.sum(axis=2).sum(axis=1)[:, np.newaxis, np.newaxis]

        self.resp = np.zeros((4, 5))
        self.resp[0] = np.sum(self.s[0, :, :], axis=0)
        self.resp[1] = np.sum(self.s[0, :, :], axis=1)
        self.resp[2] = np.sum(self.s[1, :, :], axis=1)
        self.resp[3] = np.sum(self.s[2, :, :], axis=1)

    def test__respPair_sums_to_one(self):
        print('')
        print(self.s[0])
        assert np.allclose(1.0, self.s.sum(axis=2).sum(axis=1))

    def test_Htrad_equals_Hhmm(self):
        Htrad = -1 * np.sum(bnpy.util.NumericUtil.calcRlogR(self.resp))
        Hhmm = bnpy.allocmodel.hmm.HMMUtil.calcEntropyFromResp(
            self.resp,
            self.s)
        print(Htrad)
        print(Hhmm)
        assert not np.allclose(Htrad, Hhmm)


class TestSimpleFacts(unittest.TestCase):

    ''' This is a quick test to verify a few simple facts for a single slice
        of the respPair matrix. So here, respPair is a KxK array, not TxKxK.

        Things we verify:
        * how to programmatically construct a merge of respPair
        * some simple bounds on the entropy
    '''

    def shortDescription(self):
        return None

    def setUp(self):
        self.respPair = np.asarray([
            [.01, .02, .03, .14],
            [.11, .02, .13, .04],
            [.01, .12, .03, .04],
            [.11, .02, .13, .04],
        ])
        respPair = self.respPair

        self.sigma = respPair / respPair.sum(axis=1)[:, np.newaxis]
        self.H_KxK = self.respPair * np.log(self.sigma)
        self.H_total = np.sum(self.respPair * np.log(self.sigma))

        # Merge 1and2 into A
        self.ArespPair = respPair[1:, 1:].copy()
        self.ArespPair[0, 0] += respPair[0, 0]
        self.ArespPair[:, 0] += respPair[1:, 0]
        self.ArespPair[0, :] += respPair[0, 1:]
        self.Asigma = self.ArespPair / \
            self.ArespPair.sum(axis=1)[:, np.newaxis]

        # Merge 3and4 into B
        self.BrespPair = respPair[:-1, :-1].copy()
        self.BrespPair[-1, -1] += respPair[-1, -1]
        self.BrespPair[-1, :] += respPair[-1, :-1]
        self.BrespPair[:, -1] += respPair[:-1, -1]
        self.Bsigma = self.BrespPair / \
            self.BrespPair.sum(axis=1)[:, np.newaxis]

    def test_print_A(self):
        print('')
        print('     BEFORE merge of states 1&2')
        print(self.respPair)
        print('')
        print('     AFTER  merge of states 1&2')
        print(self.ArespPair)

    def test_print_B(self):
        print('')
        print('     BEFORE merge of states 3&4')
        print(self.respPair)
        print('')
        print('     AFTER  merge of states 3&4')
        print(self.BrespPair)

    def test_sums_to_one(self):
        print('')
        print(self.respPair)
        print(self.respPair.sum())
        assert np.allclose(1.0, np.sum(self.respPair))
        assert np.allclose(1.0, np.sum(self.sigma, axis=1))

        assert np.allclose(1.0, np.sum(self.ArespPair))
        assert np.allclose(1.0, np.sum(self.Asigma, axis=1))

    def test_entropy_decreases_after_merge(self):
        print('')
        H_orig = -1 * np.sum(self.respPair * np.log(self.sigma))
        H_A = -1 * np.sum(self.ArespPair * np.log(self.Asigma))
        H_B = -1 * np.sum(self.BrespPair * np.log(self.Bsigma))
        assert H_orig >= H_A
        assert H_orig >= H_B

    def test_entropy_bounded_by_slogs(self):
        ''' Verify that -1 * s log s >= -1 * s log (sigma)
        '''
        print('')
        H_orig = -1 * np.sum(self.respPair * np.log(self.sigma))
        H_A = -1 * np.sum(self.ArespPair * np.log(self.Asigma))
        H_B = -1 * np.sum(self.BrespPair * np.log(self.Bsigma))

        G_orig = -1 * np.sum(self.respPair * np.log(self.respPair))
        G_A = -1 * np.sum(self.ArespPair * np.log(self.ArespPair))
        G_B = -1 * np.sum(self.BrespPair * np.log(self.BrespPair))
        assert G_orig >= H_orig
        assert G_B >= H_B
        assert G_A >= H_A
