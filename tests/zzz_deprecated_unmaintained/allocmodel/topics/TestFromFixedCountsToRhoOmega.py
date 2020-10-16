'''
Basic unittest that will instantiate a fixed DocTopicCount
(assumed the same for all documents, for simplicity),
and then examine how inference of rho/omega procedes given this
fixed set of counts.

Conclusions
-----------
rho/omega objective seems to be convex in rho/omega,
and very flat with respect to omega (so up to 6 significant figs,
    same objective obtained by omega that differ by say 100 or 200)
'''

import argparse
import sys
import numpy as np
from scipy.optimize import approx_fprime
import warnings
import unittest

from bnpy.util import digamma
from bnpy.allocmodel.topics import OptimizerRhoOmega
from bnpy.util.StickBreakUtil import rho2beta
from bnpy.util.StickBreakUtil import create_initrho, create_initomega
from bnpy.allocmodel.topics.HDPTopicUtil import \
    calcELBO_FixedDocTopicCountIgnoreEntropy
np.set_printoptions(precision=3, suppress=False, linewidth=140)


def np2flatstr(xvec, fmt='%9.3f', Kmax=10):
    strList = [(fmt % (x)) for x in xvec[:Kmax]]
    return ' '.join(strList)


def pprintResult(Results):
    K = Results[0]['rho'].size
    rhoinitstr = 'init : '
    rhoeststr = 'final: '
    ominitstr = 'init : '
    omeststr = 'final: '
    betastr = 'final: '
    for Info in Results:
        rho_init = Info['init'][:K]
        omega_init = Info['init'][K:]
        rho = Info['rho']
        omega = Info['omega']
        rhoinitstr += np2flatstr(rho_init, fmt=' %.4f') + '   '
        rhoeststr += np2flatstr(rho, fmt=' %.4f') + '   '
        ominitstr += np2flatstr(omega_init, fmt='%6.2f ') + '   '
        omeststr += np2flatstr(omega, fmt='%6.2f ') + '   '
        betaK = Info['betaK']
        betastr += np2flatstr(betaK, fmt=' %.4f') + '   '

    print('                             @ nDoc %d' % (Info['nDoc']))
    print('>>>>>> rho: ')
    print(rhoinitstr)
    print(rhoeststr)
    print('>>>>>> omega: ')
    print(ominitstr)
    print(omeststr)
    print('>>>>>> beta: ')
    print(betastr)
    print('')


class Test(unittest.TestCase):

    def shortDescription(self):
        return None

    def testMany_FixedCount_GlobalStepOnce(self):
        Nd = 100
        for n1 in [10, 20, 30, 40, 50]:
            DocTopicCount_d = np.asarray([n1, Nd - n1])
            self.test_FixedCount_GlobalStepOnce(
                DocTopicCount_d=DocTopicCount_d,
                alpha=0.5, gamma=10.0)

    def testMany_FixedCount_GlobalStepToConvergence(self):
        Nd = 100
        for n1 in [10, 20, 30, 40, 50]:
            DocTopicCount_d = np.asarray([n1, Nd - n1])
            self.test_FixedCount_GlobalStepToConvergence(
                DocTopicCount_d=DocTopicCount_d,
                alpha=0.5, gamma=10.0)

    def test_FixedCount_GlobalStepOnce(self,
                                       K=2,
                                       gamma=10.0,
                                       alpha=5.0,
                                       DocTopicCount_d=[100. / 2, 100 / 2]):
        ''' Given fixed counts, run one global update to rho/omega.

        Verify that regardless of initialization,
        the recovered beta value is roughly the same.
        '''
        print('')
        DocTopicCount_d = np.asarray(DocTopicCount_d, dtype=np.float64)

        print('------------- alpha %6.3f gamma %6.3f' % (
            alpha, gamma))
        print('------------- DocTopicCount [%s]' % (
            np2flatstr(DocTopicCount_d, fmt='%d'),
        ))
        print('------------- DocTopicProb  [%s]' % (
            np2flatstr(DocTopicCount_d / DocTopicCount_d.sum(), fmt='%.3f'),
        ))
        Nd = np.sum(DocTopicCount_d)
        theta_d = DocTopicCount_d + alpha * 1.0 / (K + 1) * np.ones(K)
        thetaRem = alpha * 1 / (K + 1)
        assert np.allclose(theta_d.sum() + thetaRem, alpha + Nd)
        digammaSum = digamma(theta_d.sum() + thetaRem)
        Elogpi_d = digamma(theta_d) - digammaSum
        ElogpiRem = digamma(thetaRem) - digammaSum
        for nDoc in [1, 10, 100, 1000]:
            sumLogPi = nDoc * np.hstack([Elogpi_d, ElogpiRem])

            # Now, run inference from many inits to find optimal rho/omega
            Results = list()
            for initrho in [None, 1, 2, 3]:
                initomega = None
                if isinstance(initrho, int):
                    PRNG = np.random.RandomState(initrho)
                    initrho = PRNG.rand(K)
                    initomega = 100 * PRNG.rand(K)
                rho, omega, f, Info = OptimizerRhoOmega.\
                    find_optimum_multiple_tries(
                        alpha=alpha,
                        gamma=gamma,
                        sumLogPi=sumLogPi,
                        nDoc=nDoc,
                        initrho=initrho,
                        initomega=initomega,
                    )
                betaK = rho2beta(rho, returnSize='K')
                Info.update(nDoc=nDoc, alpha=alpha, gamma=gamma,
                            rho=rho, omega=omega, betaK=betaK)
                Results.append(Info)
            pprintResult(Results)
            beta1 = Results[0]['betaK']
            for i in range(1, len(Results)):
                beta_i = Results[i]['betaK']
                assert np.allclose(beta1, beta_i, atol=0.0001, rtol=0)

    def test_FixedCount_GlobalStepToConvergence(self,
                                                gamma=10.0,
                                                alpha=5.0,
                                                nDocRange=[1, 10, 100],
                                                DocTopicCount_d=[
                                                    500,
                                                    0,
                                                    300,
                                                    200],
                                                doRestart=1):
        ''' Given fixed counts, run rho/omega inference to convergence.

        Verify that regardless of initialization,
        the recovered beta value is roughly the same.
        '''
        print('')
        DocTopicCount_d = np.asarray(DocTopicCount_d, dtype=np.float64)
        print('Fixed DocTopicCount [%s]' % (
            np2flatstr(DocTopicCount_d, fmt='%5d'),
        ))
        print('Est DocTopicProb    [%s]' % (
            np2flatstr(DocTopicCount_d / DocTopicCount_d.sum(), fmt='%.3f'),
        ))
        for nDoc in nDocRange:
            print('nDoc = %d' % (nDoc))

            rho, omega = learn_rhoomega_fromFixedCounts(
                DocTopicCount_d=DocTopicCount_d, nDoc=nDoc,
                alpha=alpha, gamma=gamma)

            if doRestart:
                print('restart with 2x smaller omega')
                rho2, omega2 = learn_rhoomega_fromFixedCounts(
                    DocTopicCount_d=DocTopicCount_d, nDoc=nDoc,
                    alpha=alpha, gamma=gamma,
                    initrho=rho / 2, initomega=omega / 2)
                assert np.allclose(rho, rho2, atol=0.0001, rtol=0)


def learn_rhoomega_fromFixedCounts(DocTopicCount_d=None,
                                   nDoc=0,
                                   alpha=None, gamma=None,
                                   initrho=None, initomega=None):
    Nd = np.sum(DocTopicCount_d)
    K = DocTopicCount_d.size
    if initrho is None:
        rho = OptimizerRhoOmega.create_initrho(K)
    else:
        rho = initrho
    if initomega is None:
        omega = OptimizerRhoOmega.create_initomega(K, nDoc, gamma)
    else:
        omega = initomega

    evalELBOandPrint(
        rho=rho, omega=omega,
        DocTopicCount=np.tile(DocTopicCount_d, (nDoc, 1)),
        alpha=alpha, gamma=gamma,
        msg='init',
    )
    betaK = rho2beta(rho, returnSize="K")
    prevbetaK = np.zeros_like(betaK)
    iterid = 0
    while np.sum(np.abs(betaK - prevbetaK)) > 0.000001:
        iterid += 1
        theta_d = DocTopicCount_d + alpha * betaK
        thetaRem = alpha * (1 - np.sum(betaK))
        assert np.allclose(theta_d.sum() + thetaRem, alpha + Nd)
        digammaSum = digamma(theta_d.sum() + thetaRem)
        Elogpi_d = digamma(theta_d) - digammaSum
        ElogpiRem = digamma(thetaRem) - digammaSum
        sumLogPi = nDoc * np.hstack([Elogpi_d, ElogpiRem])

        rho, omega, f, Info = OptimizerRhoOmega.\
            find_optimum_multiple_tries(
                alpha=alpha,
                gamma=gamma,
                sumLogPi=sumLogPi,
                nDoc=nDoc,
                initrho=rho,
                initomega=omega,
                approx_grad=1,
            )
        prevbetaK = betaK.copy()
        betaK = rho2beta(rho, returnSize="K")
        if iterid < 5 or iterid % 10 == 0:
            evalELBOandPrint(
                rho=rho, omega=omega,
                DocTopicCount=np.tile(DocTopicCount_d, (nDoc, 1)),
                alpha=alpha, gamma=gamma,
                msg=str(iterid),
            )
    return rho, omega


def evalELBOandPrint(DocTopicCount=None, alpha=None, gamma=None,
                     rho=None, omega=None, msg=''):
    ''' Check on the objective.
    '''
    L = calcELBO_FixedDocTopicCountIgnoreEntropy(
        DocTopicCount=DocTopicCount,
        alpha=alpha,
        gamma=gamma,
        rho=rho,
        omega=omega)
    nDoc = DocTopicCount.shape[0]
    betaK = rho2beta(rho, returnSize='K')
    betastr = np2flatstr(betaK, fmt="%.4f")
    omstr = np2flatstr(omega, fmt="%6.2f")
    print('%10s % .6e beta %s | omega %s' % (
        msg, L / float(nDoc), betastr, omstr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DocTopicCount_d', default='90,9,1')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--nDocRange', type=str, default='1,10')
    parser.add_argument('--doRestart', type=int, default=0)
    args = parser.parse_args()

    args.nDocRange = [int(nD) for nD in args.nDocRange.split(',')]
    args.DocTopicCount_d = [
        float(Ndk) for Ndk in args.DocTopicCount_d.split(',')]
    myTest = Test("test_FixedCount_GlobalStepToConvergence")
    myTest.test_FixedCount_GlobalStepToConvergence(
        **args.__dict__)
