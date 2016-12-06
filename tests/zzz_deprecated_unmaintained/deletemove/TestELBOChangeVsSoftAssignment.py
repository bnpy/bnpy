import numpy as np
import unittest

import bnpy

DPSpec = dict(gamma0=10.0)
GaussSpec = dict(ECovMat='eye', sF=0.5, nu=0, kappa=1e-8)


class Test(unittest.TestCase):

    def setUp(self, N=100, Data=None, likelihood='Gauss',
              **kwargs):
        GaussSpec.update(kwargs)
        DPSpec.update(kwargs)
        if Data is None:
            Data = makeGaussianDataWithTwoComps(N=N)
        self.Data = Data
        self.oneresp = 1e-50 * np.ones((2 * N, 2))
        self.oneresp[:, 0] = 1.0

        self.trueresp = 1e-50 * np.ones((2 * N, 2))
        self.trueresp[:N, 0] = 1.0
        self.trueresp[N:, 1] = 1.0

        amodel = bnpy.allocmodel.DPMixtureModel('VB', DPSpec)
        if likelihood.count('ZeroMean'):
            omodel = bnpy.obsmodel.ZeroMeanGaussObsModel(
                'VB', Data=self.Data, **GaussSpec)
        else:
            omodel = bnpy.obsmodel.GaussObsModel(
                'VB', Data=self.Data, **GaussSpec)

        self.hmodel = bnpy.HModel(amodel, omodel)

    def test_true_better_than_one(self):
        Ltrue = self.evalELBOFromResp(self.trueresp)
        Lone = self.evalELBOFromResp(self.oneresp)
        assert Ltrue > Lone

    def evalELBOFromResp(self, resp):
        LP = dict(resp=resp)
        SS = self.hmodel.get_global_suff_stats(
            self.Data, LP, doPrecompEntropy=1)
        self.hmodel.update_global_params(SS)
        ELBO = self.hmodel.calc_evidence(SS=SS)
        return ELBO

    def evalELBODictFromResp(self, resp):
        LP = dict(resp=resp)
        SS = self.hmodel.get_global_suff_stats(
            self.Data, LP, doPrecompEntropy=1)
        self.hmodel.update_global_params(SS)
        ELBOdict = self.hmodel.calc_evidence(SS=SS, todict=1)
        return ELBOdict


def strToRange(s):
    """ Convert string to range
    """
    return [int(i) for i in s.split(',')]


def strToFloatArray(s):
    """ Convert string to range
    """
    return np.asarray([float(i) for i in s.split(',')])


def makeGaussianDataWithTwoComps(N=50, D=2,
                                 likelihood='Gauss',
                                 sigmaA=1, sigmaB=1,
                                 muA=3, muB=-3,
                                 XA=None, XB=None,
                                 **kwargs):
    """ Make synthetic data with two components.
    """
    PRNG = np.random.RandomState(0)
    if XA is not None and XB is not None:
        XA = strToFloatArray(XA)[:, np.newaxis]
        XB = strToFloatArray(XB)[:, np.newaxis]
    elif D == 2:
        SigmaA = np.diag([sigmaA, sigmaA])
        SigmaB = np.diag([sigmaB, sigmaB])
        XA = PRNG.multivariate_normal([muA, 0], SigmaA, N)
        XB = PRNG.multivariate_normal([muB, 0], SigmaB, N)
    else:
        randVals = PRNG.randn(2 * N, 1)
        XA = np.sqrt(sigmaA) * randVals[::2] + muA
        XB = np.sqrt(sigmaB) * randVals[1::2] + muB

    X = np.vstack([XA, XB])
    Data = bnpy.data.XData(X)
    return Data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=str, default='50')
    parser.add_argument('--D', type=int, default=2)
    parser.add_argument('--likelihood', type=str, default='Gauss')
    parser.add_argument('--sigmaA', type=float, default=1)
    parser.add_argument('--sigmaB', type=float, default=1)
    parser.add_argument('--muA', type=float, default=3)
    parser.add_argument('--muB', type=float, default=-3)
    parser.add_argument('--XA', type=str, default=None)
    parser.add_argument('--XB', type=str, default=None)

    parser.add_argument('--gamma0', type=float, default=DPSpec['gamma0'])
    parser.add_argument('--ECovMat', type=str, default=GaussSpec['ECovMat'])
    parser.add_argument('--sF', type=float, default=GaussSpec['sF'])
    parser.add_argument('--kappa', type=float, default=GaussSpec['kappa'])
    args = parser.parse_args()

    from matplotlib import pylab

    for N in strToRange(args.N):
        args.N = N
        epsvals = np.linspace(0, 1, 100)
        ELBOvals = np.zeros_like(epsvals)
        ELBOdict = [None for ii in range(epsvals.size)]
        Data = makeGaussianDataWithTwoComps(**args.__dict__)
        myTest = Test("setUp")
        myTest.setUp(Data=Data, **args.__dict__)
        for ii, epsval in enumerate(epsvals):
            resp = (1 - epsval) * myTest.oneresp + epsval * myTest.trueresp
            ELBOvals[ii] = myTest.evalELBOFromResp(resp)
            ELBOdict[ii] = myTest.evalELBODictFromResp(resp)
            ELBOsum = np.sum([ELBOdict[ii][k] for k in ELBOdict[ii]])
            assert np.allclose(ELBOsum, ELBOvals[ii])

        pylab.subplots(nrows=3, ncols=1, figsize=(7, 9))
        figH = pylab.subplot(3, 1, 1)
        bnpy.viz.PlotComps.plotCompsFromHModel(
            myTest.hmodel, Data=myTest.Data, figH=figH, Colors=['c', 'm'])
        pylab.title('N = %d  gamma=%.2f'
                    % (args.N, args.gamma0))
        if args.D == 1:
            pylab.xlim([-10, 10])

        ax2 = pylab.subplot(3, 1, 2)
        pylab.plot(epsvals, ELBOvals, 'k.-')
        pylab.ylabel('ELBO')
        goodIDs = np.flatnonzero(ELBOvals[1:] > ELBOvals[0])
        if goodIDs.size > 0:
            pylab.title('K=2 preferred for eps > %s' % (epsvals[goodIDs[0]]))
        else:
            pylab.title('K=1 preferred')

        pylab.subplot(3, 1, 3, sharex=ax2)
        pylab.plot(epsvals, [d['Lentropy'] for d in ELBOdict], 'r.-')
        pylab.plot(epsvals, [d['Lglobal'] for d in ELBOdict], 'b.-')
        pylab.plot(epsvals, [d['Ldata'] for d in ELBOdict], 'g.-')
        pylab.legend(['Lentropy', 'Lalloc', 'Ldata'], loc='center right')
        pylab.xlabel('eps')
        pylab.ylabel('ELBO')
        pylab.xlim([-.01, 1.01])

    pylab.show(block=1)
