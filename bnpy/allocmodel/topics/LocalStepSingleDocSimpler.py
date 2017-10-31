from builtins import *
import numpy as np
from scipy.special import digamma, gammaln

from bnpy.util.OptimizerForPi import \
    estimatePiForDoc_frankwolfe, estimatePiForDoc_graddescent

def calcLocalParamsWithELBOTraceForSingleDoc(
        cts_U=None,
        logLik_UK=None,
        alphaEbeta_K=None,
        initDocTopicProb_K=None,
        initDocTopicCount_K=None,
        convThrLP=0.001,
        nCoordAscentItersLP=100,
        restartLP=0,
        restartNumItersLP=2):
    '''

    Returns
    -------
    LP : dict with fields
        * resp
        * DocTopicCount
        * theta
    Ltrace : list
    '''
    U, K = logLik_UK.shape

    explogLik_UK = logLik_UK.copy()
    maxlogLik_U = np.max(explogLik_UK, axis=1)
    explogLik_UK -= maxlogLik_U[:,np.newaxis]
    np.exp(explogLik_UK, out=explogLik_UK)

    # Perform valid initialization for DocTopicProb_k
    if initDocTopicCount_K is not None:
        DocTopicCount_K = initDocTopicCount_K.copy()
        DocTopicProb_K = DocTopicCount_K + alphaEbeta_K
        digamma(DocTopicProb_K, out=DocTopicProb_K)
        np.exp(DocTopicProb_K, out=DocTopicProb_K)
    elif initDocTopicProb_K is not None:
        DocTopicCount_K = np.zeros(K)
        DocTopicProb_K = initDocTopicProb_K.copy()
    else:
        # Default initialization!
        DocTopicCount_K = np.zeros(K)
        DocTopicProb_K = alphaEbeta_K.copy()

    # Initialize sumResp
    sumResp_U = np.zeros(U)
    np.dot(explogLik_UK, DocTopicProb_K, out=sumResp_U)

    Ltrace = np.zeros(nCoordAscentItersLP)
    prevDocTopicCount_K = DocTopicCount_K.copy()
    for riter in range(nCoordAscentItersLP):
        # # Update DocTopicCount
        np.dot(cts_U / sumResp_U, explogLik_UK,
               out=DocTopicCount_K)
        DocTopicCount_K *= DocTopicProb_K
        # # Update DocTopicProb
        np.add(DocTopicCount_K, alphaEbeta_K,
            out=DocTopicProb_K)
        digamma(DocTopicProb_K, out=DocTopicProb_K)
        np.exp(DocTopicProb_K, out=DocTopicProb_K)
        # # Update sumResp
        np.dot(explogLik_UK, DocTopicProb_K, out=sumResp_U)
        # # Compute ELBO
        Ltrace[riter] = calcELBOForSingleDocFromCountVec(
            DocTopicCount_K=DocTopicCount_K,
            cts_U=cts_U,
            sumResp_U=sumResp_U,
            alphaEbeta_K=alphaEbeta_K)
        # # Check for convergence
        maxDiff = np.max(np.abs(
            prevDocTopicCount_K - DocTopicCount_K))
        if maxDiff < convThrLP:
            break
        # Track previous DocTopicCount
        prevDocTopicCount_K[:] = DocTopicCount_K

    if restartLP:
        # Traverse active topics in random order
        PRNG = np.random.RandomState(0)
        activeTopicIDs_A = np.flatnonzero(DocTopicCount_K >= 1.0)
        PRNG.shuffle(activeTopicIDs_A)

        nOrigActive = activeTopicIDs_A.size
        nActive = nOrigActive
        curLval = Ltrace[riter]
        for k in activeTopicIDs_A:
            if nActive < 2:
                break
            propDocTopicCount_K, propLval = restartProposalForSingleDoc(
                ktarget=k,
                DocTopicCount_K=DocTopicCount_K,
                cts_U=cts_U,
                explogLik_UK=explogLik_UK,
                alphaEbeta_K=alphaEbeta_K,
                restartNumItersLP=restartNumItersLP)
            if propLval > curLval:
                nActive = nActive - 1
                curLval = propLval
                DocTopicCount_K = propDocTopicCount_K
                riter += 1
                if riter < Ltrace.size:
                    Ltrace[riter] = curLval
                else:
                    Ltrace = np.append(Ltrace, curLval)

        nAccept = nOrigActive - nActive
        print("%d/%d restarts accepted" % (nAccept, nOrigActive))

    # Correct ELBO trace for missing additive constant, indep. of topic cts
    Ltrace[riter+1:] = Ltrace[riter]
    Ltrace += np.inner(cts_U, maxlogLik_U)
    return DocTopicCount_K, Ltrace

def calcELBOForSingleDocFromCountVec(
        DocTopicCount_K=None,
        cts_U=None,
        sumResp_U=None,
        alphaEbeta_K=None,
        logLik_UK=None,
        L_max=0.0):
    ''' Compute ELBO for single doc as function of doc-topic counts.

    Returns
    -------
    L : scalar float
        equals ELBO as function of local parameters of single document
        up to an additive constant independent of DocTopicCount
    '''
    theta_K = DocTopicCount_K + alphaEbeta_K
    logPrior_K = digamma(theta_K)
    L_theta = np.sum(gammaln(theta_K)) - np.inner(DocTopicCount_K, logPrior_K)
    explogPrior_K = np.exp(logPrior_K)
    if sumResp_U is None:
        maxlogLik_U = np.max(logLik_UK, axis=1)
        explogLik_UK = logLik_UK - maxlogLik_U[:,np.newaxis]
        np.exp(explogLik_UK, out=explogLik_UK)
        sumResp_U = np.dot(explogLik_UK, explogPrior_K)
        L_max = np.inner(cts_U, maxlogLik_U)
    L_resp = np.inner(cts_U, np.log(sumResp_U))
    return L_theta + L_resp + L_max

def restartProposalForSingleDoc(
        ktarget=0,
        DocTopicCount_K=None,
        curLval=None,
        cts_U=None,
        explogLik_UK=None,
        alphaEbeta_K=None,
        restartNumItersLP=2,
        **kwargs):
    ''' Execute restart proposal at specific cluster.

    Returns
    -------
    propDocTopicCount_K : new
    propLval : scalar real
    '''
    propDocTopicCount_K = DocTopicCount_K.copy()
    propDocTopicCount_K[ktarget] = 0

    # Init DocTopicProb
    propDocTopicProb_K = np.add(propDocTopicCount_K, alphaEbeta_K)
    digamma(propDocTopicProb_K, out=propDocTopicProb_K)
    np.exp(propDocTopicProb_K, out=propDocTopicProb_K)
    # Init sumResp
    propsumResp_U = np.dot(explogLik_UK, propDocTopicProb_K)
    # Go go gadget for loop!
    for riter in range(restartNumItersLP):
        # Update DocTopicCount
        np.dot(cts_U / propsumResp_U, explogLik_UK, out=propDocTopicCount_K)
        propDocTopicCount_K *= propDocTopicProb_K
        # Determine corresponding DocTopicProb
        np.add(propDocTopicCount_K, alphaEbeta_K, out=propDocTopicProb_K)
        digamma(propDocTopicProb_K, out=propDocTopicProb_K)
        np.exp(propDocTopicProb_K, out=propDocTopicProb_K)
        # Determing corresponding sumResp
        np.dot(explogLik_UK, propDocTopicProb_K, out=propsumResp_U)
    if propDocTopicCount_K[ktarget] > 0.1:
        return DocTopicCount_K, -np.inf
    propLval = calcELBOForSingleDocFromCountVec(
        DocTopicCount_K=propDocTopicCount_K,
        cts_U=cts_U,
        alphaEbeta_K=alphaEbeta_K,
        sumResp_U=propsumResp_U)
    return propDocTopicCount_K, propLval

def calcELBOForInterpolatedDocTopicCounts(
        DTCA_K, DTCB_K, cts_U,
        logLik_UK=None,
        alphaEbeta_K=None,
        nGrid=100):
    wgrid_G = np.linspace(0, 1.0, nGrid)
    fgrid_G = np.zeros(nGrid)
    for ii in range(nGrid):
        DTC_K = wgrid_G[ii] * DTCB_K + (1.0 - wgrid_G[ii]) * DTCA_K
        fgrid_G[ii] = calcELBOForSingleDocFromCountVec(
            DocTopicCount_K=DTC_K,
            cts_U=cts_U,
            logLik_UK=logLik_UK,
            alphaEbeta_K=alphaEbeta_K,
            )
    return fgrid_G, wgrid_G

def isMonotonicIncreasing(Ltrace):
    return np.diff(Ltrace).min() > -1e-8

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--initname", type=str, default="truelabels")
    args = parser.parse_args()

    import bnpy
    pylab = bnpy.viz.PlotUtil.pylab
    bnpy.viz.PlotUtil.ConfigPylabDefaults(pylab)

    import BarsK10V900
    Data = BarsK10V900.get_data(nDocTotal=50, nWordsPerDoc=500)

    nCoordAscentItersLP = 200
    convThrLP = .00001
    hmodel, Info = bnpy.run(
        Data, 'HDPTopicModel', 'Mult', 'VB',
        initname=args.initname,
        K=args.K,
        gamma=10.0,
        alpha=0.5,
        nLap=1,
        initDocTopicCount_d='setDocProbsToEGlobalProbs',
        restartLP=1,
        nCoordAscentItersLP=nCoordAscentItersLP,
        convThrLP=convThrLP)

    # Extract relevant values from hmodel
    K = hmodel.obsModel.K
    alphaEbeta_K = hmodel.allocModel.alpha_E_beta()
    alpha = hmodel.allocModel.alpha
    LP = hmodel.obsModel.calc_local_params(Data, None)
    topics_KV = hmodel.obsModel.getTopics()

    pylab.figure(figsize=(12,6));
    for d in range(Data.nDoc):
        start = Data.doc_range[d]
        stop = Data.doc_range[d+1]
        logLik_UK = LP['E_log_soft_ev'][start:stop].copy()
        ids_U = Data.word_id[start:stop].copy()
        cts_U = Data.word_count[start:stop].copy()

        pylab.clf();
        ax = pylab.subplot(1, 2, 1);

        bestL = -np.inf
        worstL = +np.inf
        PRNG = np.random.RandomState(101 * d + 1)
        for randiter in range(50):
            randlabel = 'rand + 1'
            randProb_K = 1.0 + PRNG.rand(K)
            randProb_K /= randProb_K.sum()
            if randiter == 0:
                label = randlabel
            else:
                label = None
            randDTC_K, randLtrace = calcLocalParamsWithELBOTraceForSingleDoc(
                initDocTopicProb_K=randProb_K,
                logLik_UK=logLik_UK,
                cts_U=cts_U,
                alphaEbeta_K=alphaEbeta_K,
                convThrLP=convThrLP,
                nCoordAscentItersLP=nCoordAscentItersLP)
            pylab.plot(randLtrace, 'r--', label=label)
            if randLtrace[-1] > bestL:
                bestL = randLtrace[-1]
                bestDTC_K = randDTC_K
            if randLtrace[-1] < worstL:
                worstL = randLtrace[-1]
                worstDTC_K = randDTC_K
            assert isMonotonicIncreasing(randLtrace)

        print("BEST of ", randlabel)
        print(' '.join(['%6.1f' % (x) for x in bestDTC_K]))
        print("WORST of ", randlabel)
        print(' '.join(['%6.1f' % (x) for x in worstDTC_K]))

        PRNG = np.random.RandomState(701 * d + 7)
        for randiter in range(50):
            randlabel = 'rand'
            randProb_K = PRNG.rand(K)
            randProb_K /= randProb_K.sum()
            if randiter == 0:
                label = randlabel
            else:
                label = None
            randDTC_K, randLtrace = calcLocalParamsWithELBOTraceForSingleDoc(
                initDocTopicProb_K=randProb_K,
                logLik_UK=logLik_UK,
                cts_U=cts_U,
                alphaEbeta_K=alphaEbeta_K,
                convThrLP=convThrLP,
                nCoordAscentItersLP=nCoordAscentItersLP)
            pylab.plot(randLtrace, 'm--', label=label)
            if randLtrace[-1] > bestL:
                bestL = randLtrace[-1]
                bestDTC_K = randDTC_K
            if randLtrace[-1] < worstL:
                worstL = randLtrace[-1]
                worstDTC_K = randDTC_K
            assert isMonotonicIncreasing(randLtrace)

        print("BEST of ", randlabel)
        print(' '.join(['%6.1f' % (x) for x in bestDTC_K]))
        print("WORST of ", randlabel)
        print(' '.join(['%6.1f' % (x) for x in worstDTC_K]))

        fwpi_K, _, _ = estimatePiForDoc_frankwolfe(
            ids_U=ids_U,
            cts_U=cts_U,
            topics_KV=topics_KV,
            alpha=alpha,
            seed=d)
        fwDTC_K, fwLtrace = calcLocalParamsWithELBOTraceForSingleDoc(
            initDocTopicProb_K=fwpi_K,
            logLik_UK=logLik_UK,
            cts_U=cts_U,
            alphaEbeta_K=alphaEbeta_K,
            convThrLP=convThrLP,
            nCoordAscentItersLP=nCoordAscentItersLP)
        pylab.plot(fwLtrace, 'b-', label='frankwolfeMAP', linewidth=2);
        assert isMonotonicIncreasing(fwLtrace)

        natpi_K, _, _ = estimatePiForDoc_graddescent(
            ids_U=ids_U,
            cts_U=cts_U,
            topics_KV=topics_KV,
            alpha=alpha,
            )
        natDTC_K, natLtrace = calcLocalParamsWithELBOTraceForSingleDoc(
            initDocTopicProb_K=natpi_K,
            logLik_UK=logLik_UK,
            cts_U=cts_U,
            alphaEbeta_K=alphaEbeta_K,
            convThrLP=convThrLP,
            nCoordAscentItersLP=nCoordAscentItersLP)
        pylab.plot(natLtrace, 'g-', label='naturalMAP', linewidth=2);
        assert isMonotonicIncreasing(natLtrace)

        restartDTC_K, restartLtrace = calcLocalParamsWithELBOTraceForSingleDoc(
            logLik_UK=logLik_UK,
            cts_U=cts_U,
            alphaEbeta_K=alphaEbeta_K,
            convThrLP=convThrLP,
            nCoordAscentItersLP=nCoordAscentItersLP,
            restartLP=1,
            restartNumItersLP=10)
        assert isMonotonicIncreasing(restartLtrace)

        priorDTC_K, priorLtrace = calcLocalParamsWithELBOTraceForSingleDoc(
            logLik_UK=logLik_UK,
            cts_U=cts_U,
            alphaEbeta_K=alphaEbeta_K,
            convThrLP=convThrLP,
            nCoordAscentItersLP=nCoordAscentItersLP)
        assert isMonotonicIncreasing(priorLtrace)

        if restartLtrace[-1] > priorLtrace[-1]:
            pylab.plot(restartLtrace, '.-',
                label='prior+restarts',
                color='gold',
                linewidth=2);
        pylab.plot(priorLtrace, 'k-', label='prior', linewidth=2);

        pylab.ylabel('train ELBO')
        pylab.xlabel('iterations');
        pylab.title('local inference for doc %d' % (d));
        pylab.legend(loc='lower right')
        pylab.show(block=False)

        print("init = prior")
        print(' '.join(['%6.1f' % (x) for x in priorDTC_K]))
        print("init = frankwolfeMAP")
        print(' '.join(['%6.1f' % (x) for x in fwDTC_K]))
        print("init = naturalMAP")
        print(' '.join(['%6.1f' % (x) for x in natDTC_K]))

        if priorLtrace[-1] > bestL:
            bestL = priorLtrace[-1]
            bestDTC_K = priorDTC_K
        if fwLtrace[-1] > bestL:
            bestL = fwLtrace[-1]
            bestDTC_K = fwDTC_K
        if natLtrace[-1] > bestL:
            bestL = natLtrace[-1]
            bestDTC_K = natDTC_K

        if priorLtrace[-1] < worstL:
            worstL = priorLtrace[-1]
            worstDTC_K = priorDTC_K
        if fwLtrace[-1] < worstL:
            worstL = fwLtrace[-1]
            worstDTC_K = fwDTC_K
        if natLtrace[-1] < worstL:
            worstL = natLtrace[-1]
            worstDTC_K = natDTC_K

        fgrid, wgrid = calcELBOForInterpolatedDocTopicCounts(
            worstDTC_K, bestDTC_K,
            cts_U=cts_U,
            logLik_UK=logLik_UK,
            alphaEbeta_K=alphaEbeta_K)
        pylab.subplot(1,2,2, sharey=ax);
        pylab.plot(wgrid, fgrid, 'c.-')

        ymax = bestL
        ymin = np.minimum(worstL, fgrid.min())
        yspan = np.maximum(ymax - ymin, 10)
        pylab.ylim(
            ymin=ymin - 0.25 * yspan,
            ymax=ymax + 0.1 * yspan)
        pylab.xlabel('interpolation factor');
        #pylab.ylabel('train ELBO');
        pylab.title('convex interpolation \n between worst and best');
        pylab.draw();
        pylab.tight_layout();
        pylab.show(block=False)

        keypress = input("Press any key for next plot >>>")
        if keypress.count("embed"):
            from IPython import embed;
            embed()
        elif keypress.count("exit"):
            break
        if (d + 1) % 25 == 0:
            print("%3d/%d docs done" % (d+1, Data.nDoc))
