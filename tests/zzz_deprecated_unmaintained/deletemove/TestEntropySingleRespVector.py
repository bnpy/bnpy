import numpy as np
import unittest
import sys

try:
    from matplotlib import pylab
    doViz = True
except ImportError:
    doViz = False


rEPS = 1e-40


def makeNewResp_Exact(resp):
    """ Create new resp matrix that exactly obeys required constraints.
    """
    respNew = resp[:, 1:].copy()
    respNew /= respNew.sum(axis=1)[:, np.newaxis]
    respNew = np.maximum(respNew, rEPS)
    return respNew


def makeNewResp_Approx(resp):
    """ Create new resp matrix that exactly obeys required constraints.
    """
    respNew = resp[:, 1:].copy()
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


class TestK2(unittest.TestCase):

    def shortDescription(self):
        return None

    def setUp(self, K=2, respOther=0, dtargetMinResp=0.01):
        respOther = np.maximum(np.asarray(respOther), rEPS)
        rSum = dtargetMinResp + respOther.sum()
        rVals = np.linspace(rEPS, 1 - rSum, 100)
        N = rVals.size
        resp = np.zeros((N, K))
        resp[:, 0] = dtargetMinResp
        if K > 2:
            resp[:, 2] = rVals
        if K > 3:
            resp[:, 3:] = respOther[np.newaxis, :]
        resp[:, 1] = 1.0 - resp[:, 0] - resp[:, 2:].sum(axis=1)
        resp = np.maximum(resp, rEPS)
        resp /= resp.sum(axis=1)[:, np.newaxis]

        self.K = K
        self.R = resp
        self.Rnew_Exact = makeNewResp_Exact(resp)
        self.Rnew_Approx = makeNewResp_Approx(resp)
        self.rVals = rVals

    def test_entropy_gt_zero(self):
        """ Verify that all entropy calculations yield positive values.
        """
        H = calcRlogR(self.R)
        Hnew_exact = calcRlogR(self.Rnew_Exact)
        Hnew_approx = calcRlogR(self.Rnew_Approx)

        assert np.all(H > -1e-10)
        assert np.all(Hnew_exact > -1e-10)
        assert np.all(Hnew_approx > -1e-10)

    def test_entropy_drops_from_old_to_new(self):
        """ Verify that entropy of original is higher than candidate
        """
        H = np.sum(calcRlogR(self.R), axis=1)
        Hnew_exact = np.sum(calcRlogR(self.Rnew_Exact), axis=1)

        assert np.all(H > Hnew_exact)

    def plot_entropy_vs_rVals(self):
        if not doViz:
            self.skipTest("Required module matplotlib unavailable.")
        H = np.sum(calcRlogR(self.R), axis=1)
        Hnew_exact = np.sum(calcRlogR(self.Rnew_Exact), axis=1)
        Hnew_approx = np.sum(calcRlogR(self.Rnew_Approx), axis=1)

        rVals = self.rVals

        np.set_printoptions(precision=4, suppress=True)
        print('')
        print('--- rVals')
        print(rVals[:3], rVals[-3:])

        print('--- R original')
        print(self.R[:3])
        print(self.R[-3:, :])

        print('--- R proposal')
        print(self.Rnew_Exact[:3])
        print(self.Rnew_Exact[-3:, :])

        pylab.plot(rVals, H, 'k-', label='H original')
        pylab.plot(rVals, Hnew_exact, 'b-', label='H proposal exact')
        pylab.plot(rVals, Hnew_approx, 'r-', label='H proposal approx')
        pylab.legend(loc='best')
        pylab.xlim([rVals.min() - .01, rVals.max() + .01])
        ybuf = 0.05 * H.max()
        pylab.ylim([ybuf, H.max() + ybuf])
        pylab.show(block=True)


class TestK3(TestK2):

    def setUp(self):
        super(TestK3, self).setUp(K=3, respOther=0)


class TestK4(TestK2):

    def setUp(self):
        super(TestK4, self).setUp(K=4, respOther=[0.97])
