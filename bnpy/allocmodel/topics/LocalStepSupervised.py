from scipy.special import digamma, gammaln
import numpy as np
import warnings

USE_CYTHON = True
try:
    import pyximport; pyximport.install()
    from SupervisedHelper import calcRespInner_cython
except:
    warnings.warn('Unable to import cython module for sLDA/sHDP model')
    USE_CYTHON = False

#TODO: These functions appear in multiple places and should maybe be factored out
def eta_update(m, S, X):
    if m.size == S.size:
        eta2 = np.dot(X ** 2, S) + (np.dot(X, m) ** 2)
    else:
        eta2 = (X * np.dot(X.reshape((1, -1)), S)).sum() + (np.dot(X, m) ** 2)
    return np.sqrt(eta2)

def lam(eta):
    return np.tanh(eta / 2.0) / (4.0 * eta)

def checkWPost(w_m, w_var, K):
    w_m, w_var = np.asarray(w_m), np.asarray(w_var)

    w_m_t = np.zeros(K)
    w_m_t[:w_m.size] = w_m.flatten()[:K]
    w_m = w_m_t

    if len(w_var.shape) <= 1:
        w_var_t = np.ones(K)
        w_var_t[:w_var.size] = w_var.flatten()[:K]
        w_var = w_var_t
    else:
        w_var_t = np.eye(K)
        w_var_t[:w_var.shape[0], :w_var.shape[1]] = w_var[:K, :K]
        w_var = w_var_t

    return w_m, w_var

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

def calcResp(E_pi, Lik_d, w_m, w_var, y, wc_d):
    K = Lik_d.shape[1]
    
    #Ensure that the mean and cov are the right size
    w_m, w_var = checkWPost(w_m, w_var, K)

    Nd = np.sum(wc_d)
    Nd_2 = Nd ** 2

    #Term constant for each token
    cTerm = E_pi * np.exp(((y - 0.5) / Nd) * w_m)

    #Responsibilities before iterative adjustments
    resp = Lik_d * cTerm
    resp = resp / np.sum(resp, axis=1)[:, np.newaxis]

    #Initial covariates for regression
    #(equivalent to docTopicCounts)
    Zbar = np.dot(wc_d.reshape((1,-1)), resp).flatten()

    #Expectation: E[ww^T]
    E_outer = np.outer(w_m, w_m) + w_var

    #Update logistic approximation
    #(TODO: Consider doing this inside the inner loop)
    l = lam(eta_update(w_m, w_var, Zbar))

    #Run coordinate ascent loop
    if USE_CYTHON:
        resp, Zbar = calcRespInner_cython(resp, Zbar, wc_d, E_outer, l / Nd_2)
    else:
        resp, Zbar = calcRespInner(resp, Zbar, wc_d, E_outer, l / Nd_2)

    return np.maximum(resp, 1e-300), Zbar


def calcLocalParamsSupervised_SingleDoc(
        wc_d, Lik_d, alphaEbeta, alphaEbetaRem=None,
        DocTopicCount_d=None, sumResp_d=None,
        nCoordAscentItersLP=10, convThrLP=0.001,
        restartLP=0,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        w_m=None, w_var=None, y=0.0,
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
    for iter in xrange(nCoordAscentItersLP):
        # Update Prob of Active Topics
        # First, in logspace, so Prob_d[k] = E[ log pi_dk ] + const
        np.add(DocTopicCount_d, alphaEbeta, out=DocTopicProb_d)
        digamma(DocTopicProb_d, out=DocTopicProb_d)
        # TODO: subtract max for safe exp? doesnt seem necessary...

        # Convert: Prob_d[k] = exp E[ log pi_dk ] / const
        np.exp(DocTopicProb_d, out=DocTopicProb_d)

        #Update the responsibilities and totals
        Resp_d, DocTopicCount_d = calcResp(DocTopicProb_d, Lik_d, w_m, w_var, y, wc_d)

        # Check for convergence
        if iter % 5 == 0:
            maxDiff = np.max(np.abs(DocTopicCount_d - prevDocTopicCount_d))
            if maxDiff < convThrLP:
                break
        prevDocTopicCount_d[:] = DocTopicCount_d

    Info = dict(maxDiff=maxDiff, iter=iter)

    return DocTopicCount_d, DocTopicProb_d, Resp_d, Info
