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
import os
import numpy as np
from scipy.optimize import approx_fprime
import warnings
import unittest
import joblib
from matplotlib import pylab

from bnpy.util import digamma, as1D, argsort_bigtosmall_stable, is_sorted_bigtosmall
from bnpy.allocmodel.topics import OptimizerRhoOmegaBetter
from bnpy.util.StickBreakUtil import rho2beta
from bnpy.viz.PrintTopics import vec2str
from bnpy.allocmodel.topics.HDPTopicUtil import \
    calcELBO_IgnoreTermsConstWRTrhoomegatheta
np.set_printoptions(precision=4, suppress=1, linewidth=140)

def reorder_rho(rho, bigtosmallIDs):
    betaK = rho2beta(rho, returnSize='K')
    newbetaK = betaK[bigtosmallIDs]
    return OptimizerRhoOmegaBetter.beta2rho(newbetaK, rho.size), newbetaK

def calcAvgPiFromDocTopicCount(DocTopicCount):
    estPi = DocTopicCount / DocTopicCount.sum(axis=1)[:,np.newaxis]
    avgPi = np.sum(estPi, axis=0) / DocTopicCount.shape[0]
    return avgPi

def mapToNewPos(curposIDs, bigtosmall):
    ''' Convert list of old ids to new positions after bigtosmall reordering.

    Example
    -------
    >>> curposIDs = [0, 2, 4]
    >>> N = [11, 9, 3, 1, 5]
    >>> bigtosmall = argsort_bigtosmall_stable(N)
    >>> print bigtosmall
    [0 1 4 2 3]
    >>> newposIDs = mapToNewPos(curposIDs, bigtosmall)
    >>> print newposIDs
    [0, 3, 2]
    '''
    newposIDs = np.zeros_like(curposIDs)
    for posID in range(len(curposIDs)):
        newposIDs[posID] = np.flatnonzero(bigtosmall == curposIDs[posID])[0]
    return newposIDs.tolist()

def learn_rhoomega_fromFixedCounts(DocTopicCount=None,
                                   nDoc=0,
                                   canShuffleInit='byUsage',
                                   canShuffle=None,
                                   maxiter=5,
                                   warmStart_rho=1,
                                   alpha=None, gamma=None,
                                   initrho=None, initomega=None, **kwargs):
    assert nDoc == DocTopicCount.shape[0]
    K = DocTopicCount.shape[1]

    didShuffle = 0
    if canShuffleInit:
        if canShuffleInit.lower().count('byusage'):
            print('INITIAL SORTING BY USAGE')
            avgPi = calcAvgPiFromDocTopicCount(DocTopicCount)
            bigtosmall = argsort_bigtosmall_stable(avgPi)
        elif canShuffleInit.lower().count('bycount'):
            print('INITIAL SORTING BY COUNT')
            bigtosmall = argsort_bigtosmall_stable(DocTopicCount.sum(axis=0))
        elif canShuffleInit.lower().count('random'):
            print('INITIAL SORTING RANDOMLY')
            PRNG = np.random.RandomState(0)
            bigtosmall = np.arange(K)
            PRNG.shuffle(bigtosmall)
        else:
            bigtosmall = np.arange(K)
        # Now, sort.
        if not np.allclose(bigtosmall, np.arange(K)):
            DocTopicCount = DocTopicCount[:, bigtosmall]
            didShuffle = 1
    avgPi = calcAvgPiFromDocTopicCount(DocTopicCount)
    sortedids = argsort_bigtosmall_stable(avgPi)
    if canShuffleInit.lower().count('byusage'):
        assert np.allclose(sortedids, np.arange(K))

    # Find UIDs of comps to track
    emptyUIDs = np.flatnonzero(DocTopicCount.sum(axis=0) < 0.0001)
    if emptyUIDs.size >= 3:
        firstEmptyUID = emptyUIDs.min()
        lastEmptyUID = emptyUIDs.max()
        middleEmptyUID = emptyUIDs[len(emptyUIDs)/2]
        trackEmptyUIDs = [firstEmptyUID, middleEmptyUID, lastEmptyUID]
        emptyLabels = ['first', 'middle', 'last']
    elif emptyUIDs.size == 2:
        trackEmptyUIDs = [emptyUIDs.min(), emptyUIDs.max()]
        emptyLabels = ['first', 'last']
    elif emptyUIDs.size == 1:
        firstEmptyUID = emptyUIDs.min()
        trackEmptyUIDs = [firstEmptyUID]
        emptyLabels = ['first']
    else:
        trackEmptyUIDs = []
        emptyLabels = []

    trackActiveUIDs = list()
    activeLabels = list()
    # Track the top 5 active columns of DocTopicCount
    for pos in range(0, np.minimum(5, K)):
        if sortedids[pos] not in emptyUIDs:
            trackActiveUIDs.append(sortedids[pos])
            activeLabels.append('max+%d' % (pos))
    # Find the minnonemptyID
    for pos in range(K-1, 0, -1):
        curid = sortedids[pos]
        if curid not in emptyUIDs:
            break
    minnonemptyPos = pos
    # Track the 5 smallest active columns of DocTopicCount
    nBeyond5 = np.minimum(5, K - len(emptyUIDs) - 5)
    for i in range(-1 * (nBeyond5-1), 1):
        trackActiveUIDs.append(sortedids[minnonemptyPos + i])
        activeLabels.append('min+%d' % (-1 * i))

    assert np.all(avgPi[trackActiveUIDs] > 0)
    assert np.allclose(avgPi[trackEmptyUIDs], 0.0)
    assert is_sorted_bigtosmall(avgPi[trackActiveUIDs])

    nDocToDisplay = np.minimum(nDoc, 10)

    # Initialize rho
    if initrho is None:
        rho = OptimizerRhoOmegaBetter.make_initrho(K, nDoc, gamma)
    else:
        if didShuffle:
            rho, _ = reorder_rho(initrho, bigtosmall)
        else:
            rho = initrho
    # Initialize omega
    if initomega is None:
        omega = OptimizerRhoOmegaBetter.make_initomega(K, nDoc, gamma)
    else:
        omega = initomega
    # ELBO value of initial state
    Ltro = evalELBOandPrint(
        rho=rho, omega=omega,
        nDoc=nDoc,
        DocTopicCount=DocTopicCount,
        alpha=alpha, gamma=gamma,
        msg='init',
    )
    Snapshots = dict()
    Snapshots['DTCSum'] = list()
    Snapshots['DTCUsage'] = list()
    Snapshots['beta'] = list()
    Snapshots['Lscore'] = list()
    Snapshots['activeLabels'] = activeLabels
    Snapshots['emptyLabels'] = emptyLabels
    Snapshots['pos_trackActive'] = list()
    Snapshots['pos_trackEmpty'] = list()
    Snapshots['beta_trackActive'] = list()
    Snapshots['beta_trackEmpty'] = list()
    Snapshots['count_trackActive'] = list()
    Snapshots['count_trackEmpty'] = list()
    Snapshots['beta_trackRem'] = list()

    LtroList = list()
    LtroList.append(Ltro)
    betaK = rho2beta(rho, returnSize="K")
    iterid = 0
    prevbetaK = np.zeros_like(betaK)
    prevrho = rho.copy()
    while np.sum(np.abs(betaK - prevbetaK)) > 0.0000001:
        iterid += 1
        if iterid > maxiter:
            break
        # Take Snapshots of Learned Params
        Snapshots['Lscore'].append(Ltro)
        Snapshots['DTCSum'].append(DocTopicCount.sum(axis=0))
        Snapshots['DTCUsage'].append((DocTopicCount > 0.001).sum(axis=0))
        Snapshots['beta'].append(betaK)
        Snapshots['pos_trackActive'].append(trackActiveUIDs)
        Snapshots['pos_trackEmpty'].append(trackEmptyUIDs)
        Snapshots['beta_trackActive'].append(betaK[trackActiveUIDs])
        Snapshots['beta_trackEmpty'].append(betaK[trackEmptyUIDs])
        Snapshots['beta_trackRem'].append(1.0 - betaK.sum())
        Snapshots['count_trackActive'].append(
            DocTopicCount.sum(axis=0)[trackActiveUIDs])
        Snapshots['count_trackEmpty'].append(
            DocTopicCount.sum(axis=0)[trackEmptyUIDs])

        # Sort by beta
        didShuffle = 0
        tlabel = '_t'
        if iterid > 1 and canShuffle and canShuffle.lower().count('bybeta'):
            bigtosmall = argsort_bigtosmall_stable(betaK)
            if not np.allclose(bigtosmall, np.arange(K)):
                trackActiveUIDs = mapToNewPos(trackActiveUIDs, bigtosmall)
                trackEmptyUIDs = mapToNewPos(trackEmptyUIDs, bigtosmall)
                rho, betaK = reorder_rho(rho, bigtosmall)
                DocTopicCount = DocTopicCount[:, bigtosmall]
                didShuffle = 1
                tlabel = '_ts'
        # Update theta
        sumLogPiActiveVec, sumLogPiRemVec, LP = DocTopicCount_to_sumLogPi(
            rho=rho, omega=omega, 
            DocTopicCount=DocTopicCount,
            alpha=alpha, gamma=gamma,
            **kwargs)
        # Show ELBO with freshly-optimized theta value.
        Ltro = evalELBOandPrint(
            rho=rho, omega=omega,
            DocTopicCount=DocTopicCount,
            theta=LP['theta'],
            thetaRem=LP['thetaRem'],
            nDoc=nDoc,
            sumLogPiActiveVec=sumLogPiActiveVec,
            sumLogPiRemVec=sumLogPiRemVec,
            alpha=alpha, gamma=gamma, f=None,
            msg=str(iterid) + tlabel,
        )
        LtroList.append(Ltro)
        if not LtroList[-1] >= LtroList[-2]:
            if didShuffle:
                print('NOT MONOTONIC! just after theta update with SHUFFLE!')
            else:
                print('NOT MONOTONIC! just after theta standard update')

        didELBODrop = 0
        if canShuffle:
            if canShuffle.lower().count('bysumlogpi'):
                bigtosmall = argsort_bigtosmall_stable(
                    sumLogPiActiveVec)
            elif canShuffle.lower().count('bycounts'):
                bigtosmall = argsort_bigtosmall_stable(
                    DocTopicCount.sum(axis=0))
            elif canShuffle.lower().count('byusage'):
                estPi = DocTopicCount / DocTopicCount.sum(axis=1)[:,np.newaxis]
                avgPi = np.sum(estPi, axis=0)
                bigtosmall = argsort_bigtosmall_stable(avgPi)
            else:
                bigtosmall = np.arange(K)
            if not np.allclose(bigtosmall, np.arange(K)):
                trackActiveUIDs = mapToNewPos(trackActiveUIDs, bigtosmall)
                trackEmptyUIDs = mapToNewPos(trackEmptyUIDs, bigtosmall)
                rho, betaK = reorder_rho(rho, bigtosmall)
                sumLogPiActiveVec = sumLogPiActiveVec[bigtosmall]
                DocTopicCount = DocTopicCount[:,bigtosmall]
                LP['theta'] = LP['theta'][:, bigtosmall]
                didShuffle = 1
                # Show ELBO with freshly-optimized rho value.
                Ltro = evalELBOandPrint(
                    rho=rho, omega=omega,
                    DocTopicCount=DocTopicCount,
                    theta=LP['theta'],
                    thetaRem=LP['thetaRem'],
                    nDoc=nDoc,
                    sumLogPiActiveVec=sumLogPiActiveVec,
                    sumLogPiRemVec=sumLogPiRemVec,
                    alpha=alpha, gamma=gamma, f=None,
                    msg=str(iterid) + "_ss",
                )
                LtroList.append(Ltro)
                if not LtroList[-1] >= LtroList[-2]:
                    print('NOT MONOTONIC! just after %s shuffle update!' % (
                        canShuffle))
                    didELBODrop = 1

        prevrho[:] = rho
        # Update rhoomega
        if warmStart_rho:
            initrho = rho
        else:
            initrho = None
        rho, omega, f, Info = OptimizerRhoOmegaBetter.\
            find_optimum_multiple_tries(
                alpha=alpha,
                gamma=gamma,
                sumLogPiActiveVec=sumLogPiActiveVec,
                sumLogPiRemVec=sumLogPiRemVec,
                nDoc=nDoc,
                initrho=initrho,
                initomega=omega,
                approx_grad=1,
                do_grad_omega=0,
            )
        prevbetaK[:] = betaK
        betaK = rho2beta(rho, returnSize="K")
        # Show ELBO with freshly-optimized rho value.
        Ltro = evalELBOandPrint(
            rho=rho, omega=omega,
            DocTopicCount=DocTopicCount,
            theta=LP['theta'],
            thetaRem=LP['thetaRem'],
            nDoc=nDoc,
            sumLogPiActiveVec=sumLogPiActiveVec,
            sumLogPiRemVec=sumLogPiRemVec,
            alpha=alpha, gamma=gamma, f=f,
            msg=str(iterid) + "_r",
        )
        LtroList.append(Ltro)
        if not LtroList[-1] >= LtroList[-2]:
            print('NOT MONOTONIC! just after rho update!')

        if didELBODrop:
            if LtroList[-1] >= LtroList[-3]:
                print('Phew. Combined update of sorting then optimizing rho OK')
            else:
                print('WHOA! Combined update of sorting then' + \
                    ' optimizing rho beta NOT MONOTONIC')

    Snapshots['Lscore'].append(Ltro)
    Snapshots['DTCSum'].append(DocTopicCount.sum(axis=0))
    Snapshots['DTCUsage'].append((DocTopicCount > 0.001).sum(axis=0))
    Snapshots['beta'].append(betaK)
    Snapshots['pos_trackActive'].append(trackActiveUIDs)
    Snapshots['pos_trackEmpty'].append(trackEmptyUIDs)
    Snapshots['beta_trackActive'].append(betaK[trackActiveUIDs])
    Snapshots['beta_trackEmpty'].append(betaK[trackEmptyUIDs])
    Snapshots['beta_trackRem'].append(1.0 - betaK.sum())
    Snapshots['count_trackActive'].append(
        DocTopicCount.sum(axis=0)[trackActiveUIDs])
    Snapshots['count_trackEmpty'].append(
        DocTopicCount.sum(axis=0)[trackEmptyUIDs])

    print('\nEmpty cluster ids (%d of %d)' % (
        len(trackEmptyUIDs), len(emptyUIDs)))
    print('-----------------')
    print(' '.join(['% 10d' % (x) for x in trackEmptyUIDs]))
    

    print('\nSelected active clusters to track')
    print('---------------------------------')
    print(' '.join(['% 10d' % (x) for x in trackActiveUIDs]))
    print(' '.join(['% .3e' % (x) for x in avgPi[trackActiveUIDs]]))

    print('\nDocTopicCount for %d of %d docs' % (nDocToDisplay, nDoc))
    print('---------------------------------')
    for n in range(nDocToDisplay):
        print(' '.join([
            '% 9.2f' % (x) for x in DocTopicCount[n, trackActiveUIDs]]))

    print('\nFinal sumLogPiActiveVec')
    print('---------------------------------')
    print(' '.join(['% .3e' % (x) for x in sumLogPiActiveVec[trackActiveUIDs]]))

    print('is sumLogPiActiveVec sorted?', \
        is_sorted_bigtosmall(sumLogPiActiveVec))
    return rho, omega, Snapshots


def evalELBOandPrint(nDoc=None,
                     theta=None, thetaRem=None,
                     DocTopicCount=None,
                     sumLogPiActiveVec=None,
                     sumLogPiRemVec=None,
                     alpha=None, gamma=None,
                     rho=None, omega=None, msg='', f=None, **kwargs):
    ''' Check on the objective.
    '''
    L = calcELBO_IgnoreTermsConstWRTrhoomegatheta(
        nDoc=nDoc,
        alpha=alpha,
        gamma=gamma,
        DocTopicCount=DocTopicCount,
        theta=theta,
        thetaRem=thetaRem,
        sumLogPi=sumLogPiActiveVec,
        sumLogPiRemVec=sumLogPiRemVec,
        rho=rho,
        omega=omega)

    if sumLogPiActiveVec is None:
        sumLogPiActiveVec, sumLogPiRemVec, LP = DocTopicCount_to_sumLogPi(
            rho=rho, omega=omega,
            DocTopicCount=DocTopicCount, alpha=alpha, gamma=gamma)

    Lrhoomega = OptimizerRhoOmegaBetter.negL_rhoomega_viaHDPTopicUtil(
        nDoc=nDoc,
        alpha=alpha,
        gamma=gamma,
        sumLogPiActiveVec=sumLogPiActiveVec,
        sumLogPiRemVec=sumLogPiRemVec,
        rho=rho,
        omega=omega)
    if f is None:
        f = Lrhoomega
    assert np.allclose(f, Lrhoomega)
    print("%10s Ltro= % .8e   Lro= % .5e  fro= % .5e" % (
        msg, L, Lrhoomega, f))
    return L

def DocTopicCount_to_sumLogPi(
        rho=None, omega=None, 
        betaK=None, DocTopicCount=None, alpha=None, gamma=None, **kwargs):
    '''

    Returns
    -------
    f : scalar
    '''
    K = rho.size
    if betaK is None:
        betaK = rho2beta(rho, returnSize="K")
    theta = DocTopicCount + alpha * betaK[np.newaxis,:]
    thetaRem = alpha * (1 - np.sum(betaK))
    assert np.allclose(theta.sum(axis=1) + thetaRem,
        alpha + DocTopicCount.sum(axis=1))
    digammaSum = digamma(theta.sum(axis=1) + thetaRem)
    ElogPi = digamma(theta) - digammaSum[:,np.newaxis]
    ElogPiRem = digamma(thetaRem) - digammaSum
    sumLogPiActiveVec = np.sum(ElogPi, axis=0)
    sumLogPiRemVec = np.zeros(K)
    sumLogPiRemVec[-1] = np.sum(ElogPiRem)

    LP = dict(
        ElogPi=ElogPi,
        ElogPiRem=ElogPiRem,
        digammaSumTheta=digammaSum,
        theta=theta,
        thetaRem=thetaRem)
    return sumLogPiActiveVec, sumLogPiRemVec, LP

def f_DocTopicCount(
        rho=None, omega=None, 
        betaK=None, DocTopicCount=None, alpha=None, gamma=None, **kwargs):
    ''' Evaluate the objective f for rho/omega optimization.

    Returns
    -------
    f : scalar
    '''
    K = rho.size
    sumLogPiActiveVec, sumLogPiRemVec, LP = DocTopicCount_to_sumLogPi(
        rho=rho, omega=omega, 
        DocTopicCount=DocTopicCount, alpha=alpha, gamma=gamma,
        **kwargs)

    f = OptimizerRhoOmegaBetter.negL_rhoomega(
        rho=rho, omega=omega,
        sumLogPiActiveVec=sumLogPiActiveVec,
        sumLogPiRemVec=sumLogPiRemVec,
        alpha=alpha, gamma=gamma,
        nDoc=DocTopicCount.shape[0],
        approx_grad=1)
    return f

def makeDocTopicCount(
        nDoc=10, K=5, Kempty=0,
        seed=0, minK_d=1, maxK_d=5, **kwargs):
    '''

    Returns
    -------
    DocTopicCount : 2D array, nDoc x K
    '''
    PRNG = np.random.RandomState(seed)
    DocTopicCount = np.zeros((nDoc, K))
    for d in range(nDoc):
        # Pick one to five random columns to be left in each doc
        maxK_d = np.minimum(K - Kempty, maxK_d)
        K_d = PRNG.choice(maxK_d, size=1)
        K_d = np.maximum(K_d, minK_d)
        ks = PRNG.choice(K - Kempty, size=K_d, replace=False)
        for k in ks:
            DocTopicCount[d,k] = 1
    DocTopicCount *= 100 * PRNG.rand(nDoc, K)
    return DocTopicCount

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumppath', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--maxiter', type=int, default=5)
    parser.add_argument('--canShuffle', type=str, default='None')
    parser.add_argument('--canShuffleInit', type=str, default='None')
    parser.add_argument('--useSavedInit_rho', type=int, default=1)
    parser.add_argument('--useSavedInit_omega', type=int, default=0)
    parser.add_argument('--warmStart_rho', type=int, default=1)

    parser.add_argument('--doInteractive', type=int, default=1)
    parser.add_argument('--savename', type=str, default=None)

    parser.add_argument('--nDoc', type=int, default=10)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--Kempty', type=int, default=0)

    args = parser.parse_args()
    K = args.K
    savename = args.savename

    if savename:
        print('')
        print('')
        print('>>>>>>>>>>', savename)

    # If provided, load DocTopicCount from saved file.
    if args.dumppath and os.path.exists(args.dumppath):
        print("Loading DocTopicCount from file: ", args.dumppath)
        LVars = joblib.load(args.dumppath)
        assert 'DocTopicCount' in LVars
        if 'rho' in LVars:
            if args.useSavedInit_rho:
                LVars['initrho'] = LVars['rho']
            del LVars['rho']
        if 'omega' in LVars:
            if args.useSavedInit_omega:
                LVars['initomega'] = LVars['omega']
            del LVars['omega']
        args.__dict__.update(LVars)
        args.__dict__['K'] = LVars['DocTopicCount'].shape[1]
    else:
        args.__dict__['DocTopicCount'] = makeDocTopicCount(**args.__dict__)

    # Do the hard work
    _, _, Snapshots = learn_rhoomega_fromFixedCounts(**args.__dict__)

    if savename:
        with open('Lscore_' + savename + '.txt', 'w') as f:
            for Lscore in Snapshots['Lscore']:
                f.write("%.8f\n" % (Lscore))

    nrows = 2
    ncols = 3
    FONTSIZE = 20

    pylab.subplots(figsize=(20, 9), nrows=nrows, ncols=ncols)

    PosMat = np.vstack(Snapshots['pos_trackActive']) + 1
    ax1 = pylab.subplot(nrows, ncols, 1)
    pylab.plot( PosMat[:, :5], '.-' )
    pylab.ylabel('position', fontsize=FONTSIZE)
    pylab.title('largest active', fontsize=FONTSIZE)
    ax1.get_yaxis().set_major_locator(pylab.MaxNLocator(integer=True))
    
    BetaMat = np.vstack(Snapshots['beta_trackActive'])
    ax4 = pylab.subplot(nrows, ncols, 4, sharex=ax1)
    pylab.plot( BetaMat[:, :5], '.-' )
    pylab.ylabel('beta', fontsize=FONTSIZE)

    nActive = len(Snapshots['activeLabels'])
    if nActive > 5:
        nSmall = nActive - 5
        ax2 = pylab.subplot(nrows, ncols, 2, sharex=ax1, sharey=ax1)
        pylab.plot( PosMat[:, -nSmall:], '.-' )
        pylab.title('smallest active', fontsize=FONTSIZE)
        ax2.get_yaxis().set_major_locator(pylab.MaxNLocator(integer=True))

        ax5 = pylab.subplot(nrows, ncols, 5, sharex=ax4)
        pylab.plot( BetaMat[:, -nSmall:], '.-' )
    else:
        ax2 = pylab.subplot(nrows, ncols, 2, sharex=ax1)
        pylab.title('smallest active', fontsize=FONTSIZE)
        ax2.get_yaxis().set_major_locator(pylab.MaxNLocator(integer=True))

        ax5 = pylab.subplot(nrows, ncols, 5, sharex=ax4)


    if len(Snapshots['emptyLabels']) > 0:
        ax3 = pylab.subplot(nrows, ncols, 3, sharex=ax2, sharey=ax2)
        EmptyPosMat = np.vstack(Snapshots['pos_trackEmpty']) + 1
        pylab.plot( EmptyPosMat, '.-' )
        pylab.title('empty', fontsize=FONTSIZE)

        ax6 = pylab.subplot(nrows, ncols, 6, sharex=ax4, sharey=ax5)
        EmptyBetaMat = np.vstack(Snapshots['beta_trackEmpty'])
        pylab.plot(EmptyBetaMat, '.-' )
    else:
        ax3 = pylab.subplot(nrows, ncols, 3, sharex=ax1, sharey=ax2)
        pylab.title('empty', fontsize=FONTSIZE)

    
    ax6 = pylab.subplot(nrows, ncols, 6, sharex=ax4, sharey=ax5)
    pylab.plot(Snapshots['beta_trackRem'], 'k--' )
    pylab.show(block=False)

    if savename:
        pylab.savefig('BetaTracePlot_' + savename + ".png", pad_inches=0)
        pylab.savefig('BetaTracePlot_' + savename + ".eps", pad_inches=0)
    if args.doInteractive:
        keyinput = input("Press any key to quit >>>")
        if keyinput.count('embed'):
            from IPython import embed;
            embed()
    #myTest = Test("test_FixedCount_GlobalStepToConvergence")
    #myTest.test_FixedCount_GlobalStepToConvergence(
    #    **args.__dict__)
