import numpy as np
import unittest
import sys
from bnpy.allocmodel.mix.DPMixtureModel import calcCachedELBOGap_SinglePair
from bnpy.allocmodel.mix.DPMixtureModel import calcCachedELBOTerms_SinglePair
from bnpy.suffstats.SuffStatBag import SuffStatBag
try:
    from matplotlib import pylab
    doViz = True
except ImportError:
    doViz = False

rEPS = 1e-40


def makeNewResp_Exact(resp, delCompID, targetCompID):
    """ Create new resp matrix by following hard merge procedure.

    Returns
    -------
    respNew : 2D array, size N x K-1
        all mass formerly assigned to delCompID transferred to targetCompID.
    """
    if targetCompID >= delCompID:
        raise ValueError('INDEXING SUCKS!')
    respNew = np.delete(resp, delCompID, axis=1)
    respNew[:, targetCompID] += resp[:, delCompID]
    respNew = np.maximum(respNew, rEPS)
    return respNew


def calcRlogR(R):
    """

    Returns
    -------
    H : 2D array, size N x K
        each entry is positive.
    """
    return -1 * R * np.log(R)


def pprintR(R):
    """ Pretty printing of 2D array R
    """
    if R.size < 5:
        print(R[:, :10])
    else:
        print(R[:3, :10])
        print(R[-3:, :10])


def pprintRandL(R, L, Rmsg=''):
    """ Pretty printing of 2D array R and corresponding entropy.
    """

    strR = ''
    nRows = np.minimum(3, R.shape[0])
    for r in range(nRows):
        strR += '  '.join(['%.4f' % (x) for x in R[r, :10]]) + '\n'

    strR = strR.replace('[', ' ')
    strR = strR.replace(']', ' ')

    lines = strR.split('\n')
    print('--------------------')
    print('%10s | R %s' % ('Lentropy', Rmsg))
    print('--------------------')
    for lID, line in enumerate(lines):
        if lID == 0:
            line = '% 10.3f' % (L) + '   ' + line
        else:
            line = '%13s' % (' ') + line
        print(line)


class MyTestN1K4(unittest.TestCase):

    """ Unit test for calculation of (bounds on) entropy of compound moves.

    We are particularly interested in what to do with "non target"
    portion for a delete move to guarantee exact calculation.

    Attributes
    ----------
    Horig : 1D array, size K
        this is a per-state scalar for entropy
    Lorig : scalar
        this is the entropy (scalar) of the original model

    """

    def shortDescription(self):
        return None

    def setUp(self, K=4, N=1,
              dtargetMinResp=0.01,
              nMoves=3,
              Rsource='random'):
        """ Create original R and a several compound hard merge proposals.
        """
        rng = np.random.RandomState(101)

        if Rsource == 'random':
            R = 1.0 / (K - nMoves) + rng.rand(N, K)
            R[:, -nMoves:] = dtargetMinResp
            assert R.sum(axis=1).min() > 1.0
        elif Rsource == 'toydata':
            raise NotImplementedError('TODO')
            # TODO run for a few iters on toy data to get "realistic"
            # responsibility matrix from a junk-y initialization.

        R = np.maximum(R, rEPS)
        R /= R.sum(axis=1)[:, np.newaxis]
        assert np.all(R[:, -nMoves:] <= dtargetMinResp)

        self.K = K
        self.R = R
        self.Rnew1 = makeNewResp_Exact(R, K - 1, 0)
        self.Rnew2 = makeNewResp_Exact(self.Rnew1, K - 2, 0)
        self.Rnew3 = makeNewResp_Exact(self.Rnew2, K - 3, 0)

        assert np.all(self.Rnew1[:, (K - 2):] <= dtargetMinResp)
        assert np.all(self.Rnew2[:, (K - 3):] <= dtargetMinResp)

        self.Horig = calcRlogR(self.R).sum(axis=0)
        self.H1 = calcRlogR(self.Rnew1).sum(axis=0)
        self.H2 = calcRlogR(self.Rnew2).sum(axis=0)
        self.H3 = calcRlogR(self.Rnew3).sum(axis=0)

        self.Lorig = np.sum(self.Horig)
        self.L1 = np.sum(self.H1)
        self.L2 = np.sum(self.H2)
        self.L3 = np.sum(self.H3)

        Reps = dtargetMinResp
        sumH1meps = N * (1 - Reps) * np.log(1 - Reps)
        self.L1_lb = self.Lorig - self.Horig[-1] + sumH1meps
        self.L2_lb = self.L1_lb - self.Horig[-2] + sumH1meps
        self.L3_lb = self.L2_lb - self.Horig[-3] + sumH1meps

        Nvec = R.sum(axis=0)
        self.L1_lb2 = self.Lorig - self.Horig[-1] - Nvec[-1]
        self.L2_lb2 = self.L1_lb2 - self.Horig[-2] - Nvec[-2]
        self.L3_lb2 = self.L2_lb2 - self.Horig[-3] - Nvec[-3]

        self.H1_lb = np.delete(self.Horig, -1, axis=0)
        self.H1_lb[0] += sumH1meps
        self.H2_lb = np.delete(self.H1_lb, -1, axis=0)
        self.H2_lb[0] += sumH1meps

        self.H1_lb2 = np.delete(self.Horig, -1, axis=0)
        self.H1_lb2[0] -= Nvec[-1]
        self.H2_lb2 = np.delete(self.H1_lb2, -1, axis=0)
        self.H2_lb2[0] -= Nvec[-2]

        SS = SuffStatBag(K=K, D=0)
        SS.setField("N", Nvec, dims=("K"))
        SS.setELBOTerm("ElogqZ", -1 * self.Horig, dims=("K"))
        self.SSorig = SS

        self.gap1 = self.L1 - self.Lorig
        self.gap1_lb2 = calcCachedELBOGap_SinglePair(
            SS, 0, K - 1, delCompID=K - 1)

        # Adjust suff stats to reflect first move
        ELBOTerms = calcCachedELBOTerms_SinglePair(
            SS,
            0,
            K -
            1,
            delCompID=K -
            1)
        SS1 = SS.copy()
        SS1.mergeComps(0, K - 1)
        SS1.setELBOTerm("ElogqZ", ELBOTerms["ElogqZ"], dims=("K"))
        self.SS1 = SS1

        # Adjust suff stats to reflect second move
        ELBOTerms = calcCachedELBOTerms_SinglePair(
            SS1,
            0,
            K -
            2,
            delCompID=K -
            2)
        SS2 = SS1.copy()
        SS2.mergeComps(0, K - 2)
        SS2.setELBOTerm("ElogqZ", ELBOTerms["ElogqZ"], dims=("K"))
        self.SS2 = SS2

        self.gap2 = self.L2 - self.Lorig
        self.gap2_lb2 = self.gap1_lb2 + calcCachedELBOGap_SinglePair(
            SS1, 0, K - 2, delCompID=K - 2)

    def test_entropy_gt_zero(self):
        """ Verify that all entropy calculations yield positive values.
        """
        assert np.all(self.Horig > -1e-10)
        assert np.all(self.H1 > -1e-10)
        assert np.all(self.H2 > -1e-10)

    def test_entropy_drops_from_old_to_new(self):
        """ Verify that entropy drops as more moves are performed.

        Each successive candidate should have lower entropy than before.
        """
        print('')
        print('Lorig = % 7.3f' % (self.Lorig))
        print('L1    = % 7.3f' % (self.L1))
        print('L2    = % 7.3f' % (self.L2))
        print('L3    = % 7.3f' % (self.L3))

        assert self.Lorig > self.L1
        assert self.L1 > self.L2
        assert self.L2 > self.L3

    def test_pprint_proposals(self):
        """ Display the original R and each proposal, prettily.
        """
        print('')
        pprintRandL(self.R, self.Lorig, 'original')
        pprintRandL(self.Rnew1, self.L1, 'after 1 delete')
        pprintRandL(self.Rnew2, self.L2, 'after 2 deletes')
        pprintRandL(self.Rnew3, self.L3, 'after 3 deletes')

    def test_memoized_way_to_compute_bound(self):
        """ Verify that H1_lb (memoized) equals L1 (computed directly)

        This will certify that we can use bound without touching local params.
        """
        print('')
        print('L1_lb        = % 7.5f' % (self.L1_lb))
        print('H1_lb.sum()  = % 7.5f' % (self.H1_lb.sum()))
        assert np.allclose(self.L1_lb, self.H1_lb.sum())

        print('')
        print('L2_lb        = % 7.5f' % (self.L2_lb))
        print('H2_lb.sum()  = % 7.5f' % (self.H2_lb.sum()))
        assert np.allclose(self.L2_lb, self.H2_lb.sum())

        print('')
        print('L1_lb2       = % 7.5f' % (self.L1_lb2))
        print('H1_lb2.sum() = % 7.5f' % (self.H1_lb2.sum()))
        H = self.SS1.getELBOTerm('ElogqZ').sum()
        print('SS1.getELBO  = % 7.5f' % (-1 * H))
        assert np.allclose(self.L1_lb2, self.H1_lb2.sum())
        assert np.allclose(self.L1_lb2, -1 * H)

        print('')
        print('L2_lb2       = % 7.5f' % (self.L2_lb2))
        print('H2_lb2.sum() = % 7.5f' % (self.H2_lb2.sum()))
        H = self.SS2.getELBOTerm('ElogqZ').sum()
        print('SS2.getELBO  = % 7.5f' % (-1 * H))

    def test_entropy_bounds_hold(self):
        """ Verify "binary entropy" trick is correct in our calculations.

        Using the binary entropy trick:
            \sum_{k=1}^K H'_k >= \sum_{k=1}^K H_k + sum1mepslog1meps
        """
        print('')
        print('L1     = % 7.5f' % (self.L1))
        print('L1_lb2 = % 7.5f' % (self.L1_lb2))
        print('L1_lb  = % 7.5f' % (self.L1_lb))

        assert self.L1 > self.L1_lb
        assert self.L1 > self.L1_lb2

        print('')
        print('L2     = % 7.5f' % (self.L2))
        print('L2_lb2 = % 7.5f' % (self.L2_lb2))
        print('L2_lb  = % 7.5f' % (self.L2_lb))
        assert self.L2 > self.L2_lb
        assert self.L2 > self.L2_lb2

        print('')
        print('L3     = % 7.5f' % (self.L3))
        print('L3_lb2 = % 7.5f' % (self.L3_lb2))
        print('L3_lb  = % 7.5f' % (self.L3_lb))
        assert self.L3 > self.L3_lb
        assert self.L3 > self.L3_lb2

    def test_gap_bounds_hold(self):
        """ Verify gap calculation is correct.
        """
        print('')
        print("GAP L_move1 - L_orig")
        print('gap1      = % 7.5f' % (self.gap1))
        print('gap1_lb2  = % 7.5f' % (self.gap1_lb2))
        assert self.gap1 > self.gap1_lb2

        print("GAP L_move2 - L_orig")
        print('gap2      = % 7.5f' % (self.gap2))
        print('gap2_lb2  = % 7.5f' % (self.gap2_lb2))
        assert self.gap2 > self.gap2_lb2


class MyTestN1K7(MyTestN1K4):

    def setUp(self):
        super(type(self), self).setUp(K=7, N=1)


class MyTestN50K4(MyTestN1K4):

    def setUp(self):
        super(type(self), self).setUp(K=4, N=50)


class MyTestN1000K11(MyTestN1K4):

    def setUp(self):
        super(type(self), self).setUp(K=11, N=1000)
