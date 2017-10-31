from builtins import *
from scipy.special import digamma, gammaln
import numpy as np
import warnings


def calcLocalParams_SingleDoc(
        wc_d, Lik_d, alphaEbeta, alphaEbetaRem=None,
        DocTopicCount_d=None, sumResp_d=None,
        nCoordAscentItersLP=10, convThrLP=0.001,
        restartLP=0,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        **kwargs):
    ''' Infer local parameters for a single document.

    Args
    --------
    wc_d : scalar or 1D array, size N
        word counts for document d
    Lik_d : 2D array, size N x K
        Likelihood values for each token n and topic k.
    alphaEbeta : 1D array, size K
        Scalar prior parameter for each active topic, under the prior.
    alphaEbetaRem : None or scalar
        Scalar prior parameter for all inactive topics, aggregated.
        Used only for ELBO calculation, not any update equations.

    Kwargs
    --------
    nCoordAscentItersLP : int
        Number of local step iterations to do for this document.
    convThrLP : float
        Threshold for convergence to halt iterations early.
    restartLP : int
        If 0, do not perform sparse restarts.
        If 1, perform sparse restarts.

    Returns
    --------
    DocTopicCount_d : 1D array, size K
    DocTopicProb_d : 1D array, size K
        Updated probability vector for active topics in this doc.
        Known up to a multiplicative constant.
    sumResp_d : 1D array, size N_d
        sumResp_d[n] is normalization constant for token n.
        That is, resp[n, :] / sumResp_d[n] will sum to one, when
        resp[n,k] is computed from DocTopicCount_d and Lik_d.
    Info : dict
        Contains info about convergence, sparse restarts, etc.
    '''
    if sumResp_d is None:
        sumResp_d = np.zeros(Lik_d.shape[0])

    # Initialize prior from global topic probs
    if DocTopicCount_d is None:
        DocTopicCount_d = np.zeros_like(alphaEbeta)

    if initDocTopicCountLP.count('setDocProbsToEGlobalProbs'):
        # Here, we initialize pi_d to alphaEbeta
        DocTopicProb_d = alphaEbeta.copy()
        # Update sumResp for all tokens in document
        np.dot(Lik_d, DocTopicProb_d, out=sumResp_d)
        # Update DocTopicCounts
        np.dot(wc_d / sumResp_d, Lik_d, out=DocTopicCount_d)
        DocTopicCount_d *= DocTopicProb_d
    else:
        # Set E[pi_d] to exp E log[ alphaEbeta ]
        DocTopicProb_d = np.zeros_like(alphaEbeta)

    prevDocTopicCount_d = DocTopicCount_d.copy()
    for iter in range(nCoordAscentItersLP):
        # Update Prob of Active Topics
        # First, in logspace, so Prob_d[k] = E[ log pi_dk ] + const
        np.add(DocTopicCount_d, alphaEbeta, out=DocTopicProb_d)
        digamma(DocTopicProb_d, out=DocTopicProb_d)
        # TODO: subtract max for safe exp? doesnt seem necessary...

        # Convert: Prob_d[k] = exp E[ log pi_dk ] / const
        np.exp(DocTopicProb_d, out=DocTopicProb_d)

        # Update sumResp for all tokens in document
        np.dot(Lik_d, DocTopicProb_d, out=sumResp_d)

        # Update DocTopicCounts
        np.dot(wc_d / sumResp_d, Lik_d, out=DocTopicCount_d)
        DocTopicCount_d *= DocTopicProb_d

        # Check for convergence
        if iter % 5 == 0:
            maxDiff = np.max(np.abs(DocTopicCount_d - prevDocTopicCount_d))
            if maxDiff < convThrLP:
                break
        prevDocTopicCount_d[:] = DocTopicCount_d

    Info = dict(maxDiff=maxDiff, iter=iter)


    # Allow sparse restarts ONLY on first pass through dataset
    if restartLP > 1:
        if 'lapFrac' in kwargs and kwargs['lapFrac'] <= 1.0:
            restartLP = 1
        else:
            restartLP = 0

    if restartLP:
        DocTopicCount_d, DocTopicProb_d, sumResp_d, RInfo = \
            removeJunkTopics_SingleDoc(
                wc_d, Lik_d, alphaEbeta, alphaEbetaRem,
                DocTopicCount_d, DocTopicProb_d, sumResp_d, **kwargs)
        Info.update(RInfo)

    return DocTopicCount_d, DocTopicProb_d, sumResp_d, Info


def removeJunkTopics_SingleDoc(
        wc_d, Lik_d, alphaEbeta, alphaEbetaRem,
        DocTopicCount_d, DocTopicProb_d, sumResp_d,
        restartNumTrialsLP=5,
        restartNumItersLP=2,
        restartCriteriaLP='smallest',
        restartMinSizeThrLP=0.001,
        **kwargs):
    ''' Propose candidate local parameters, accept if ELBO improves.

    Returns
    --------
    DocTopicCount_d : 1D array, size K
    DocTopicProb_d : 1D array, size K
    sumResp_d : 1D array, size N
    Info : dict
    '''
    Info = dict(nTrial=0, nAccept=0)

    # usedTopics : 1D array of int ids of topics with mass above MinSizeThr
    usedTopicMask = DocTopicCount_d > restartMinSizeThrLP
    usedTopics = np.flatnonzero(usedTopicMask)
    nUsed = np.sum(usedTopicMask)
    if nUsed < 2:
        return DocTopicCount_d, DocTopicProb_d, sumResp_d, Info

    # Measure current model quality via ELBO
    curELBO = calcELBO_SingleDoc(
        DocTopicCount_d, DocTopicProb_d, sumResp_d,
        wc_d, alphaEbeta, alphaEbetaRem)
    Info['startELBO'] = curELBO

    # Determine eligible topics to delete
    # smallTopics : 1D array of int topic ids to try deleting
    smallIDs = np.argsort(DocTopicCount_d[usedTopics])[:restartNumTrialsLP]
    smallTopics = usedTopics[smallIDs]
    smallTopics = smallTopics[:nUsed - 1]

    pDocTopicCount_d = np.zeros_like(DocTopicCount_d)
    pDocTopicProb_d = np.zeros_like(DocTopicProb_d)
    psumResp_d = np.zeros_like(sumResp_d)

    for kID in smallTopics:
        # Propose deleting current "small" topic
        pDocTopicCount_d[:] = DocTopicCount_d
        pDocTopicCount_d[kID] = 0

        # Refine initial proposal via standard coord ascent updates
        for iter in range(restartNumItersLP):
            np.add(pDocTopicCount_d, alphaEbeta, out=pDocTopicProb_d)
            digamma(pDocTopicProb_d, out=pDocTopicProb_d)
            np.exp(pDocTopicProb_d, out=pDocTopicProb_d)

            np.dot(Lik_d, pDocTopicProb_d, out=psumResp_d)

            # Update DocTopicCounts
            np.dot(wc_d / psumResp_d, Lik_d, out=pDocTopicCount_d)
            pDocTopicCount_d *= pDocTopicProb_d

        if np.any(np.isnan(pDocTopicCount_d)):
            warnings.warn('Sparse restart failed because NaN occurred.' + \
                ' Will continue with original, unaffected model. No worries.')
            break

        # Evaluate proposal quality via ELBO
        propELBO = calcELBO_SingleDoc(
            pDocTopicCount_d, pDocTopicProb_d, psumResp_d,
            wc_d, alphaEbeta, alphaEbetaRem)

        Info['nTrial'] += 1
        if not np.isfinite(propELBO):
            warnings.warn('Sparse restart failed because NaN occurred.' + \
                ' Will continue with original, unaffected model. No worries.')
            break

        # Update if accepted!
        if propELBO > curELBO:
            Info['nAccept'] += 1
            curELBO = propELBO
            DocTopicCount_d[:] = pDocTopicCount_d
            DocTopicProb_d[:] = pDocTopicProb_d
            sumResp_d[:] = psumResp_d
            nUsed -= 1

        if nUsed < 2:
            break

    assert np.all(np.isfinite(DocTopicCount_d))
    assert np.isfinite(curELBO)
    # Package up and return
    Info['finalELBO'] = curELBO
    return DocTopicCount_d, DocTopicProb_d, sumResp_d, Info


def calcELBO_SingleDoc(DocTopicCount_d, DocTopicProb_d, sumResp_d,
                       wc_d, alphaEbeta, alphaEbetaRem):
    ''' Calculate single document contribution to the ELBO objective.

    This isolates all ELBO terms that depend on local parameters of this doc.

    Returns
    -------
    L : scalar float
        value of ELBO objective, up to additive constant.
        This constant is independent of any local parameter attached to doc d.
    '''
    theta_d = DocTopicCount_d + alphaEbeta

    if alphaEbetaRem is None:
        # LDA model, with K active topics
        sumTheta = theta_d.sum()
        L_alloc = np.sum(gammaln(theta_d)) - gammaln(sumTheta)
        # SLACK terms are always equal to zero!
        #digammaSum = digamma(sumTheta)
        #ElogPi_d = digamma(theta_d) - digammaSum
    else:
        # HDP, with K active topics and one aggregate "leftover" topic
        sumTheta = theta_d.sum() + alphaEbetaRem
        L_alloc = np.sum(gammaln(theta_d)) - gammaln(sumTheta) + \
            gammaln(alphaEbetaRem)
        # SLACK terms are always equal to zero!
        #digammaSum = digamma(sumTheta)
        #ElogPi_d = digamma(theta_d) - digammaSum
        #ElogPiRem = digamma(alphaEbetaRem) - digammaSum
    if isinstance(wc_d, float):
        L_rest = np.sum(np.log(sumResp_d))
    else:
        L_rest = np.inner(wc_d, np.log(sumResp_d))
    L_rest -= np.inner(DocTopicCount_d, np.log(DocTopicProb_d + 1e-100))
    return L_alloc + L_rest




def calcELBO_SingleDocFromSparseResp(
        spResp_d, ElogLik_d, wc_d, alphaEbeta):
    ''' Calculate single document contribution to the ELBO objective.

    This isolates all ELBO terms that depend on local parameters of this doc.

    Returns
    -------
    L : scalar float
        value of ELBO objective, up to additive constant.
        This constant is independent of any local parameter attached to doc d.
    '''
    DocTopicCount_d = wc_d * spResp_d # Sparse multiply
    theta_d = DocTopicCount_d + alphaEbeta
    R_d = spResp_d.toarray()
    wR_d = wc_d[:, np.newaxis] * R_d
    L_alloc = np.sum(gammaln(theta_d))
    L_data = np.sum(wR_d * ElogLik_d)
    RlogR = np.sum(wR_d * np.log(R_d + 1e-100))
    return L_alloc + L_data - RlogR


def calcLocalParams_SingleDoc_WithELBOTrace(
        wc_d, Lik_d, alphaEbeta, alphaEbetaRem=None,
        DocTopicCount_d=None, sumResp_d=None,
        nCoordAscentItersLP=10, convThrLP=0.001,
        restartLP=0,
        **kwargs):
    ''' Infer local parameters for a single document, with ELBO trace.

    Performs same calculations as calcLocalParams_SingleDoc,
    but (expensively) tracks the ELBO at every local step iteration.
    Thus, we refactored this into a separate function, so we do not
    pay a performance penalty for an if statement in the inner loop.

    Args
    --------
    Same as calcLocalParams_SingleDoc


    Returns
    --------
    DocTopicCount_d : updated doc-topic counts
    Prior_d : prob of topic in document, up to mult. constant
    sumR_d : normalization constant for each token
    Info : dict, with field
        * 'ELBOtrace' : 1D array, size nIters
            which gives the ELBO over the iterations on this document
            up to additive const indep of local params.
    '''
    if sumResp_d is None:
        sumResp_d = np.zeros(Lik_d.shape[0])

    # Initialize prior from global topic probs
    DocTopicProb_d = alphaEbeta.copy()

    if DocTopicCount_d is None:
        # Update sumResp for all tokens in document
        np.dot(Lik_d, DocTopicProb_d, out=sumResp_d)

        # Update DocTopicCounts
        DocTopicCount_d = np.zeros_like(DocTopicProb_d)
        np.dot(wc_d / sumResp_d, Lik_d, out=DocTopicCount_d)
        DocTopicCount_d *= DocTopicProb_d

    ELBOtrace = list()
    prevDocTopicCount_d = DocTopicCount_d.copy()
    for iter in range(nCoordAscentItersLP):
        # Update Prob of Active Topics
        # First, in logspace, so Prob_d[k] = E[ log pi_dk ] + const
        np.add(DocTopicCount_d, alphaEbeta, out=DocTopicProb_d)
        digamma(DocTopicProb_d, out=DocTopicProb_d)
        # TODO: subtract max for safe exp? doesnt seem necessary...

        # Convert: Prob_d[k] = exp E[ log pi_dk ] / const
        np.exp(DocTopicProb_d, out=DocTopicProb_d)

        # Update sumResp for all tokens in document
        np.dot(Lik_d, DocTopicProb_d, out=sumResp_d)

        # Update DocTopicCounts
        np.dot(wc_d / sumResp_d, Lik_d, out=DocTopicCount_d)
        DocTopicCount_d *= DocTopicProb_d

        # Calculate ELBO objective at current assignments
        curELBO = calcELBO_SingleDoc(
            DocTopicCount_d, DocTopicProb_d, sumResp_d,
            wc_d, alphaEbeta, alphaEbetaRem)
        ELBOtrace.append(curELBO)

        # Check for convergence
        if iter % 5 == 0:
            maxDiff = np.max(np.abs(DocTopicCount_d - prevDocTopicCount_d))
            if maxDiff < convThrLP:
                break
        prevDocTopicCount_d[:] = DocTopicCount_d

    Info = dict(maxDiff=maxDiff, iter=iter)
    Info['ELBOtrace'] = np.asarray(ELBOtrace)
    if restartLP:
        DocTopicCount_d, DocTopicProb_d, sumResp_d, RInfo = \
            removeJunkTopics_SingleDoc(
                wc_d, Lik_d, alphaEbeta, alphaEbetaRem,
                DocTopicCount_d, DocTopicProb_d, sumResp_d, **kwargs)
        Info.update(RInfo)
    return DocTopicCount_d, DocTopicProb_d, sumResp_d, Info
