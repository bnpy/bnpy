from scipy.special import digamma, gammaln
import numpy as np
import math
import warnings

import itertools


def calcLocalParams_SingleDoc(
        resp_d, theta_d, response_d, word_count_d,
        eta, Lik_d,
        delta=0.1, alpha=1.0,
        nCoordAscentItersLP=30, convThrLP=0.001, **kwargs):
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
    N_d = float(sum(word_count_d))
    nTokens_d, K = resp_d.shape

    converged = False
    if response_d is None:
        t2 = 0.0
    else:
        t2 = eta * (response_d / (N_d * delta))

    for iter in xrange(nCoordAscentItersLP):
        if converged:
            break
        t1 = digamma(theta_d) - digamma(sum(theta_d))
        for i in range(nTokens_d):
            R = np.sum(word_count_d[:, None] * resp_d, axis=0) \
                - (word_count_d[i] * resp_d[i, :])

            t3 = np.inner(eta.T, R)

            t3 = (t3 * eta) / float(delta * np.square(N_d))
            t4 = (word_count_d[i] * eta * eta) \
                / (2.0 * delta * np.square(N_d))
            #if iter==0:
            '''if response_d == -6.79934750e+00:
                print t1
                #print N_d, (response_d / (N_d ))
                #print  eta * (response_d / (N_d ))
                print t2
                print t3
                print t4
                print Lik_d[i, :]
                #print resp_d[i,:]
                #print word_count_d[i]
                print '-----------'
            '''
            T = t1 + t2 - t3 - t4
            resp_d[i, :] = Lik_d[i, :] * np.exp(T - T.max())

            # Normalize
            rsum = resp_d[i, :].sum()
            resp_d[i, :] = resp_d[i, :] / rsum

            # avoid underflow
            resp_d[i, :] = np.maximum(resp_d[i, :], 1e-100)

        prev_theta_d = theta_d.copy()
        theta_d = alpha + np.sum(word_count_d[:, None] * resp_d, axis=0)

        # Check for convergence
        # if iter % 5 == 0:
        maxDiff = np.max(np.abs(theta_d - prev_theta_d))
        if maxDiff < convThrLP:
            converged = True
            # break

    DocTopicCount_d = np.dot(word_count_d, resp_d)

    resp_d_update = resp_d
    Info = dict(maxDiff=maxDiff, iter=iter)

    return resp_d_update, theta_d, DocTopicCount_d, Info


def removeJunkTopics_SingleDoc(
        wc_d, Lik_d, alphaEbeta, alphaEbetaRem,
        DocTopicCount_d, DocTopicProb_d, sumResp_d,
        restartNumTrialsLP=5,
        restartNumItersLP=2,
        restartCriteriaLP='smallest',
        restartMinSizeThrLP=0.001,
        **kwargs):
    pass


def L_supervised_single_doc(resp_d, response_d, wc_d, N_d, delta, eta):
    """Calculate slda term of the ELBO objective.

    E[p(y)]

    Returns
    -------
    L_supervised : scalar float
    """

    nTokens_d, K = resp_d.shape
    weighted_resp_d = wc_d[:, None] * resp_d  # DocTopicCounts

    EZ_d = np.sum(weighted_resp_d, axis=0) / float(N_d)
    '''
        EZ_d_slow = np.zeros(K)
        for t in range(nTokens_d):
                for k in range(K):
                        EZ_d_slow[k] += wc_d[t] * resp_d[t,k]

        EZ_d_slow = EZ_d_slow / float(N_d)

        print np.allclose(EZ_d, EZ_d_slow)
        '''

    EZTZ_d = calc_EZTZ_one_doc(wc_d, resp_d)

    '''
        EZTZ_d_slow = np.zeros((K,K))
        for t in xrange(nTokens_d):
                for s in xrange(nTokens_d):
                        if s != t:
                                EZTZ_d_slow += np.outer(weighted_resp_d[t],weighted_resp_d[s])

        for t in xrange(nTokens_d):
                EZTZ_d_slow += (wc_d[t]) * np.diag(weighted_resp_d[t,:])

        EZTZ_d_slow = EZTZ_d_slow / float(np.square(N_d))
        '''

    # print np.allclose(EZTZ_d_slow,EZTZ_d)

    '''
        A = np.ones((nTokens_d,nTokens_d)) - np.eye(nTokens_d)
        #A = np.ones((N_d,N_d)) - np.eye(N_d)

        EZTZ_d = np.inner(weighted_resp_d.transpose(),A)
        EZTZ_d = np.inner(EZTZ_d,weighted_resp_d.transpose())
        EZTZ_d = np.add(EZTZ_d,np.diag(np.sum((wc_d **2 ) * resp_d,axis=0)))

        '''

    sterm = np.dot(eta, EZTZ_d)
    sterm = np.dot(sterm, eta)
    #sterm = np.dot(EZTZ_d,eta)
    #sterm = np.dot(eta,sterm)

    L_supervised_d = (-0.5) * np.log(2.0 * math.pi * delta)
    L_supervised_d -= np.square(response_d) / (2.0 * delta)
    L_supervised_d += (response_d / delta) * np.inner(eta, EZ_d)
    L_supervised_d -= sterm / (2.0 * delta)

    return L_supervised_d


def calcELBO_SingleDoc(resp_d, wc_d, Lik_d, theta_d, delta, response_d, N_d, eta):
    #K = resp_d.shape[1]

    L = 0
    nTokens_d, K = resp_d.shape

    # print theta_d.shape
    #digammaSumTheta = digamma(theta_d.sum(axis=1))
    #DigammaTheta = digamma(theta_d) - digammaSumTheta[:, np.newaxis]

    for t in range(nTokens_d):
        wc_dt = wc_d[t]

        for k in range(K):
            DigammaTheta = digamma(theta_d[k]) - digamma(sum(theta_d))
            L += wc_dt * resp_d[t, k] * Lik_d[t, k]  # E[log p(w) ]
            L += wc_dt * resp_d[t, k] * DigammaTheta  # E[log p(z) ]
            L -= wc_dt * resp_d[t, k] * np.log(resp_d[t, k])  # E[log q(z) ]

    # E[log p(y) ]
    L += L_supervised_single_doc(resp_d, response_d, wc_d, N_d, delta, eta)

    return L


def calcLocalParams_SingleDoc_WithELBOTrace(
        resp_d, theta_d, response_d, word_count_d,
        eta, Lik_d,
        delta=0.1, alpha=1.0,
        nCoordAscentItersLP=10, convThrLP=0.001, **kwargs):
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

    ELBOtrace = list()
    N_d = sum(word_count_d)
    nTokens_d, K = resp_d.shape

    converged = False

    t2 = eta * (response_d / (N_d * delta))

    for iter in xrange(nCoordAscentItersLP):
        if converged:
            break

        t1 = digamma(theta_d) - digamma(sum(theta_d))
        '''
                t1_slow = np.zeros(K)

                for k in range(K):
                        t1_slow[k] = digamma(theta_d[k]) - digamma(sum(theta_d))
                print np.allclose(t1,t1_slow)
                '''

        # for i in range(len(resp_d)):
        for i in range(nTokens_d):

            R = np.sum(
                word_count_d[:, None] * resp_d, axis=0) - (word_count_d[i] * resp_d[i, :])

            # print np.sum(weighted_resp_d,axis=0).shape
            # print  np.sum(word_count_d[:,None] * resp_d,axis=0).shape
            t3 = np.inner(eta.T, R)

            t3 = (t3 * eta) / float(delta * np.square(N_d))
            t4 = (word_count_d[i] * eta * eta) / (2.0 * delta * np.square(N_d))

            T = t1 + t2 - t3 - t4

            resp_d[i, :] = Lik_d[i, :] * np.exp(T - T.max())

            # Normalize
            rsum = resp_d[i, :].sum()
            resp_d[i, :] = resp_d[i, :] / rsum

            # avoid underflow
            resp_d[i, :] = np.maximum(resp_d[i, :], 1e-100)

        prev_theta_d = theta_d.copy()
        theta_d = alpha + np.sum(word_count_d[:, None] * resp_d, axis=0)

        curELBO = calcELBO_SingleDoc(
            resp_d, word_count_d, Lik_d, theta_d, delta, response_d, N_d, eta)
        # print curELBO
        ELBOtrace.append(curELBO)

        # Check for convergence
        # if iter % 5 == 0:
        maxDiff = np.max(np.abs(theta_d - prev_theta_d))
        if maxDiff < convThrLP:
            converged = True
            # break

    DocTopicCount_d = np.dot(word_count_d, resp_d)

    resp_d_update = resp_d
    Info = dict(maxDiff=maxDiff, iter=iter)
    # print ELBOtrace
    Info['ELBOtrace'] = np.asarray(ELBOtrace)
    L = Info['ELBOtrace']
    # Check for monotinicity

    # if not all(x<=y for x, y in zip(L, L[1:])):
    # print 'elbo decreased!'
    # print L
    return resp_d_update, theta_d, DocTopicCount_d, Info


def calc_EZTZ_one_doc(wc_d, resp_d):

    nTokens_d, K = resp_d.shape
    weighted_resp_d = wc_d[:, None] * resp_d  # Doc
    N_d = np.sum(wc_d)

    A = np.ones((nTokens, nTokens)) - np.eye(nTokens)

    EZTZ = np.inner(weighted_resp_d.transpose(), A)
    EZTZ = np.inner(EZTZ, weighted_resp_d.transpose())

    EZTZ = EZTZ + np.diag(np.sum(w[:, None] * weighted_resp_d, axis=0))

    '''
        EZTZ = np.zeros((K,K))

        for t,s in itertools.combinations(xrange(nTokens_d), 2):
                tmp_KK = np.outer(weighted_resp_d[t], weighted_resp_d[s])
                EZTZ += tmp_KK
                EZTZ += tmp_KK.T

        for t in xrange(nTokens_d):
                EZTZ += (wc_d[t] * wc_d[t]) * np.diag(resp_d[t])
        '''
    EZTZ = (1.0/np.square(N_d)) * EZTZ

    return EZTZ
