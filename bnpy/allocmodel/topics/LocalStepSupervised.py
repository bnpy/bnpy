from scipy.special import digamma, gammaln
import numpy as np
import warnings, os

from bnpy.util import checkWPost, eta_update, calc_Zbar_ZZT, lam

USE_CYTHON = True
try:
    import pyximport; pyximport.install()
    from SupervisedHelper import calcRespInner_cython, calcRespInner_cython_blas, normalizeRows_cython
except:
    warnings.warn('Unable to import cython module for sLDA/sHDP model')
    USE_CYTHON = False


def calcRespInner(resp, Zbar, wc_d, E_outer, l_div_Nd_2):
    nTok = resp.shape[0]
    E_diag = np.diag(E_outer)
    for i in xrange(nTok):
        #Subtract current token from Zbar
        Zbar -= wc_d[i] * resp[i, :]

        #Compute the update to the resp for token i
        update = 2 * np.dot(Zbar.reshape(1,-1), E_outer) + E_diag
        resp[i, :] *= np.exp( -l_div_Nd_2 * update.flatten() )

        #Normalize
        resp[i, :] = resp[i,:] / np.sum(resp[i,:])

        #Update Zbar with the new resp
        Zbar += wc_d[i] * resp[i, :]

    return resp, Zbar

def calcResp(E_pi, Lik_d, w_m, w_var, E_outer, y, wc_d, Nd, lik_weight=1):
    Nd_2 = Nd ** 2

    #Term constant for each token
    cTerm = lik_weight * np.dot(y - 0.5, w_m) / Nd
    cTerm = E_pi * np.exp(cTerm - np.max(cTerm))

    #Responsibilities before iterative adjustments
    resp = Lik_d * cTerm

    if USE_CYTHON:
        normalizeRows_cython(resp)
    else:
        resp = resp / np.sum(resp, axis=1)[:, np.newaxis]

    #Initial covariates for regression
    #(equivalent to docTopicCounts)
    Zbar, ZZT = calc_Zbar_ZZT(resp, wc_d, Nd)

    #Sum the expected outer products for each label
    E_outer_sum = np.zeros_like(E_outer[0])
    for i in xrange(w_m.shape[0]):
        #Update logistic approximation (and weight by lik_weight)
        #(TODO: Consider doing this inside the inner loop)
        l = lam(eta_update(w_m[i], w_var[i], Zbar, ZZT)) * lik_weight
        E_outer_sum += l * E_outer[i]

    #Make Zbar unnormalized for updates
    Zbar *= Nd 

    #Run coordinate ascent loop
    if USE_CYTHON:
        calcRespInner_cython(resp, Zbar, wc_d, E_outer_sum, 1.0 / Nd_2)
    else:
        resp, Zbar = calcRespInner(resp, Zbar, wc_d, E_outer_sum, 1.0 / Nd_2)

    return np.maximum(resp, 1e-300), Zbar


def calcLocalParamsSupervised_SingleDoc(
        wc_d, Lik_d, alphaEbeta, alphaEbetaRem=None,
        DocTopicCount_d=None, sumResp_d=None,
        nCoordAscentItersLP=10, convThrLP=0.001,
        restartLP=0,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        w_m=None, w_var=None, y=0.0, lik_weight=1.0,
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

    #Setup the regression parameters (Outside the inner loop)
    K = Lik_d.shape[1]
    Nd = np.sum(wc_d)

    #Get the posterior parameters for each regression model
    all_w_m, all_w_var, all_E_outer = [], [], []
    for i in xrange(y.size):
        if w_m.ndim < 2:
            w_m_i, w_var_i = w_m, w_var
        else:
            w_m_i, w_var_i = w_m[i], w_var[i]

        #Ensure that the mean and cov are the right size
        w_m_i, w_var_i = checkWPost(w_m_i, w_var_i, K, force2D=True)
        
        #Expectation: E[ww^T]
        E_outer_i = np.outer(w_m_i, w_m_i) + w_var_i

        all_w_m.append(w_m_i)
        all_w_var.append(w_var_i)
        all_E_outer.append(E_outer_i)

    w_m = np.stack(all_w_m)
    w_var = np.stack(all_w_var)
    E_outer = np.stack(all_E_outer)

    prevDocTopicCount_d = DocTopicCount_d.copy()
    for iter in xrange(nCoordAscentItersLP):
        # Update Prob of Active Topics
        # First, in logspace, so Prob_d[k] = E[ log pi_dk ] + const
        np.add(DocTopicCount_d, alphaEbeta, out=DocTopicProb_d)
        digamma(DocTopicProb_d, out=DocTopicProb_d)
        # TODO: subtract max for safe exp? doesnt seem necessary...

        # Convert: Prob_d[k] = exp E[ log pi_dk ] / const
        np.exp(DocTopicProb_d, out=DocTopicProb_d)

        #Update the responsibilities and totals
        Resp_d, DocTopicCount_d = calcResp(DocTopicProb_d, Lik_d, w_m, w_var, E_outer, y, wc_d, Nd, lik_weight)

        # Check for convergence
        if iter % 5 == 0:
            maxDiff = np.max(np.abs(DocTopicCount_d - prevDocTopicCount_d))
            if maxDiff < convThrLP:
                break
        prevDocTopicCount_d[:] = DocTopicCount_d

    Info = dict(maxDiff=maxDiff, iter=iter)

    return DocTopicCount_d, DocTopicProb_d, Resp_d, Info
