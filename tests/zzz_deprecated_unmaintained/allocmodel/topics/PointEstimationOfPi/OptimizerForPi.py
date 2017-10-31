import numpy as np
import scipy.optimize
import bnpy
import warnings
from bnpy.util import as1D, as2D

from matplotlib import pylab

def lossFuncAndGrad(pi_d=None,
        cts_d=None, topics_d=None, alpha=0.0, scale=1.0,
        **kwargs):
    ''' Compute objective and gradient together.

    Returns
    -------
    f : scalar real
        value of negative log joint probability
    grad : 1D array, size K
        grad[k] : derivative of f w.r.t. pi_d[k]
    '''
    avgWordFreq_d = np.dot(pi_d, topics_d)
    f_lik = np.inner(cts_d, np.log(avgWordFreq_d))
    grad_lik = np.dot(topics_d, cts_d / avgWordFreq_d)

    if alpha > 0:
        np.maximum(pi_d, 1e-100, out=pi_d)
        f_prior = alpha * np.sum(np.log(pi_d))
        grad_prior = alpha / pi_d
    else:
        f_prior = 0
        grad_prior = 0
    return (-1.0 * scale * (f_lik + f_prior),
            -1.0 * scale * (grad_lik + grad_prior))

def lossFunc(pi_d=None, 
        cts_d=None, topics_d=None, alpha=0.0, scale=1.0,
        **kwargs):
    ''' Compute objective function for document-topic probabilities.

    Args
    ----
    pi_d : 1D array, size K
        pi_d[k] : probability of k-th topic in doc d
    cts_d : 1D array, size U_d
        cts_d[i] : count of i-th unique-word in doc d
    topics_d : 2D array, K x U_d
        topics_d[k,i]: probability of i-th unique-word in d under topic k
    alpha : scalar float > 0

    Returns
    -------
    f : scalar real
        value of negative log joint probability
        suitable for minimization algorithms
    '''
    f_lik = np.inner(cts_d, np.log(np.dot(pi_d, topics_d)))
    f_prior = alpha * np.sum(np.log(pi_d))
    return -1.0 * scale * (f_lik + f_prior)

def gradOfLoss(pi_d=None, cts_d=None, topics_d=None, alpha=0.0, scale=1.0,
        **kwargs):
    ''' Compute gradient of objective function

    Returns
    -------
    grad : 1D array, size K
        grad[k] gives the derivative w.r.t. pi_d[k] of F(...)
    '''
    # avgWordFreq_d : 1D array, size U_d
    #   avgWordFreq_d[i] = probability of word i using pi_d mixture of topics
    avgWordFreq_d = np.dot(pi_d, topics_d)
    grad_lik = np.sum(cts_d / avgWordFreq_d, topics_d)
    if alpha > 0:
        grad_prior = alpha / pi_d
    else:
        grad_prior = 0.0
    return -1 * scale * (grad_lik + grad_prior)

def pi2eta(pi_d):
    ''' Transform vector on simplex to unconstrained real vector

    Returns
    -------
    eta : 1D array, size K-1

    Examples
    --------
    >>> print float(pi2eta(eta2pi(0.42)))
    0.42

    >>> print float(pi2eta(eta2pi(-1.337)))
    -1.337

    >>> print pi2eta(eta2pi([-1, 0, 1]))
    [-1.  0.  1.]
    '''
    pi_d = as1D(np.asarray(pi_d))
    eta_d = pi_d[:-1] / pi_d[-1]
    np.log(eta_d, out=eta_d)
    return eta_d

def eta2pi(eta_d):
    eta_d = as1D(np.asarray(eta_d))
    pi_d = np.ones(eta_d.size+1)
    pi_d[:-1] = np.exp(eta_d)
    pi_d[:-1] += 1e-100
    pi_d /= (1.0 + np.sum(pi_d[:-1]))
    return pi_d

def eta2piJacobian(eta_d=None, pi_d=None):
    ''' Compute Jacobian matrix of transformation of eta_d to pi_d

    Returns
    -------
    J : 2D array, size K-1 x K
        J[a, b] = deriv of pi_{b}(eta_d) w.r.t. eta_d[a]
    '''
    if pi_d is None:
        pi_d = eta2pi(eta_d)
    J = -1.0 * np.outer(pi_d[:-1], pi_d)
    J[:, :-1] += np.diag(pi_d[:-1])
    return J


def estimatePi2(
        ids_d=None, cts_d=None, topics=None, alpha=0.0,
        gtol=1e-7, maxiter=1000,
        method='l-bfgs-b', **kwargs):
    kwargs.update(locals())
    if ids_d is not None:
        kwargs['topics_d'] = topics[:, ids_d]
    else:
        kwargs['topics_d'] = topics
    with warnings.catch_warnings():
        warnings.filterwarnings('error', '', RuntimeWarning)
        try:
            pi, f, Info = _estimatePi2(**kwargs)
        except RuntimeWarning as e:
            raise ValueError(e)
            # Handle errors related to numerical overflow
            '''
            # by restarting from slightly different initialization
            if 'piInit' not in kwargs:
                K = args[2].shape[0] # topics.shape[0]
                piInit = 1.0/K * np.ones(K)
            else:
                piInit = kwargs['piInit']
            K = piInit.size
            piInit = 0.9 * piInit + 0.1 * (1.0/ K * np.ones(K))
            kwargs['piInit'] = piInit
            if 'maxiter' in kwargs:
                kwargs['maxiter'] /= 2
            else:
                kwargs['maxiter'] = MAXITER
            pi, f, Info = estimatePi2(*args, **kwargs)
            '''
    '''
    if Info.message.count('iterations'):
        if kwargs['maxiter'] < MAXITER:
            kwargs['maxiter'] = MAXITER
            kwargs['piInit'] = pi
            pi, f, Info = estimatePi2(*args, **kwargs)
        
    if Info.message.count('precision loss'):
        # Try from slightly larger gtol
        if 2 * kwargs['gtol'] < MAXGTOL:
            kwargs['gtol'] *= 2
            kwargs['piInit'] = pi
            pi, f, Info = estimatePi2(*args, **kwargs)
    '''
    if not Info.success:
        if 'approx_grad' in kwargs and not kwargs['approx_grad']:
            raise ValueError(Info.message)
    return pi, f, Info


def _estimatePi2(
        ids_d=None, cts_d=None, topics=None, alpha=0.0, scale=1.0,
        method='bfgs',
        piInit=None,
        approx_grad=False,
        numRestarts=0,
        gtol=1e-9, # Values > 1e-6 can be bad for simplex
        maxiter=10000,
        options=None,
        **kwargs):
    '''
    Returns
    -------
    pi : 1D array, size K
    f : scalar
    Info : dict
    '''
    if options is None:
        options = dict(gtol=gtol, maxiter=maxiter)
    
    K = topics.shape[0]
    topics_d = topics[:, ids_d]
    if piInit is None:
        piInit = 1.0/K * np.ones(K)
    if approx_grad:
        def naturalLossFunc(eta_d):
            pi_d = eta2pi(eta_d)
            f = lossFunc(
                pi_d, cts_d=cts_d, topics_d=topics_d, alpha=alpha, scale=scale)
            return f
        Result = scipy.optimize.minimize(
            naturalLossFunc,
            x0=pi2eta(piInit),
            jac=False,
            options=options,
            method=method,
            )    
    else:
        def naturalLossFuncAndGrad(eta_d):
            pi_d = eta2pi(eta_d)
            f, gradPi = lossFuncAndGrad(
                pi_d,
                cts_d=cts_d,
                topics_d=topics_d, alpha=alpha, scale=scale)
            gradEta = np.dot(eta2piJacobian(pi_d=pi_d), gradPi)
            return f, gradEta
        Result = scipy.optimize.minimize(
            naturalLossFuncAndGrad,
            x0=pi2eta(piInit),
            jac=True,
            options=options,
            method=method,
            )
    f = Result.fun
    piEst = eta2pi(Result.x)
    return piEst, f, Result

def estimatePi(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', '', RuntimeWarning)
        try:
            pi, f, Info = _estimatePi(*args, **kwargs)
        except RuntimeWarning as e:
            # Handle errors related to numerical overflow
            # by restarting from slightly different initialization
            piInit = kwargs['piInit']
            if piInit is not None:
                K = piInit.size
                piInit = 0.9 * piInit + 0.1 * (1.0/ K * np.ones(K))
            kwargs['piInit'] = piInit
            kwargs['scale'] /= 2
            pi, f, Info = estimatePi(*args, **kwargs)
            f /= kwargs['scale']
    return pi, f, Info


def _estimatePi(ids_d, cts_d, topics, alpha, scale=1.0,
        factr=2.0, # Set L1 convergence tol to 2x machine precision
        pgtol=10 * np.finfo(float).eps,
        piInit=None,
        numRestarts=0,
        approx_grad=False):
    '''

    Returns
    -------
    pi : 1D array, size K
    f : scalar
    Info : dict
    '''
    K = topics.shape[0]
    topics_d = topics[:, ids_d]
    if piInit is None:
        piInit = 1.0/K * np.ones(K)
    if approx_grad:
        def naturalLossFunc(eta_d):
            pi_d = eta2pi(eta_d)
            f = lossFunc(pi_d, cts_d, topics_d, alpha, scale=scale)
            return f
        etaHat, f, Info = scipy.optimize.fmin_l_bfgs_b(
            naturalLossFunc, x0=pi2eta(piInit), approx_grad=True)
        piHat = eta2pi(etaHat)
        Info['piInit'] = piInit
        return piHat, f, Info

    else:
        def naturalLossFuncAndGrad(eta_d):
            pi_d = eta2pi(eta_d)
            f, gradPi = lossFuncAndGrad(
                pi_d, cts_d, topics_d, alpha, scale=scale)
            gradEta = np.dot(eta2piJacobian(pi_d=pi_d), gradPi)
            return f, gradEta
        etaEst, f, Info = scipy.optimize.fmin_l_bfgs_b(
            naturalLossFuncAndGrad,
            x0=pi2eta(piInit),
            factr=factr,
            pgtol=pgtol,
            )
        Info['piInit'] = piInit
        Info['numRestartsTried'] = 0
        Info['fList'] = [f]
        for r in range(numRestarts):
            bumpFrac = 0.1 * 2**(-r)
            piUnif = 1.0 / K * np.ones(K)
            piPrev = eta2pi(etaEst)
            eta2, f2, restartInfo = scipy.optimize.fmin_l_bfgs_b(
                naturalLossFuncAndGrad,
                x0=pi2eta((1-bumpFrac) * piPrev + bumpFrac * piUnif),
                factr=factr,
                pgtol=pgtol,
                )
            if f2 > f:
                continue
            etaEst = eta2
            piEst = eta2pi(etaEst)
            f = f2
            Info['fList'].append(f)
            Info['numRestartsTried'] += 1
            if np.sum(np.abs(piEst - piPrev)) < .0001:
                break
        piEst = eta2pi(etaEst)
        return piEst, f, Info

def pi2str(arr):
    pistr = np.array_str(arr,
        max_line_width=80, precision=4, suppress_small=1)
    return pistr.replace('[','').replace(']','')

def calcLossFuncForInterpolatedPi(piA, piB, lossFunc, nGrid=100):        
    wgrid = np.linspace(0, 1.0, nGrid)
    fgrid = np.zeros(nGrid)
    for ii in range(nGrid):
        pi_ii = wgrid[ii] * piA + (1.0 - wgrid[ii]) * piB
        f = lossFunc(pi_ii)
        if isinstance(f, tuple):
            f = f[0]
        fgrid[ii] = f        
    return fgrid, wgrid

if __name__ == '__main__':
    PRNG = np.random.RandomState(0)

    import CleanBarsK10
    Data = CleanBarsK10.get_data(nDocTotal=100, nWordsPerDoc=500)

    #import nips
    #Data = nips.get_data()

    if hasattr(Data, 'TrueParams'):
        topics = Data.TrueParams['topics']
        nudgePi = PRNG.dirichlet(0.1 * np.ones(topics.shape[1]),
            size=topics.shape[0])
        nudgedTopics = 0.98 * topics + 0.02 * nudgePi
        repeatTopics = np.vstack([topics, nudgedTopics])
        topics = repeatTopics
    else:
        K = np.minimum(50, Data.nDoc)
        chosenDocIDs = PRNG.choice(Data.nDoc, K, replace=False)
        topics = Data.getSparseDocTypeCountMatrix()[chosenDocIDs].toarray()
        topics += 0.1
        topics /= topics.sum(axis=1)[:,np.newaxis]

    assert np.allclose(topics.sum(axis=1), 1.0)

    atol = 1e-3
    K = topics.shape[0]
    alpha = 0.0 #1.0 / K
    
    for d in range(Data.nDoc):
        start_d = Data.doc_range[d]
        stop_d = Data.doc_range[d+1]
        ids_d = Data.word_id[start_d:stop_d]
        cts_d = Data.word_count[start_d:stop_d]

        if hasattr(Data, 'TrueParams'):
            trueN_d = np.dot(cts_d, Data.TrueParams['resp'][start_d:stop_d])
            truePi_d = (trueN_d + alpha)
            truePi_d /= truePi_d.sum()
            print('')
            print("     True Pi[%d]:\n %s" % (d, pi2str(truePi_d)))

        numPi_d, numf, numInfo = estimatePi2(ids_d, cts_d, topics, alpha,
            scale=1.0, #/np.sum(cts_d),
            approx_grad=True)
        print("Numerical Pi[%d]:\n %s" % (d, pi2str(numPi_d)))

        estPi_d, f, Info = estimatePi2(ids_d, cts_d, topics, alpha,
            scale=1.0, #/np.sum(cts_d))
            approx_grad=False,
            )
        print("Estimated Pi[%d]:\n %s" % (d, pi2str(estPi_d)))

        
        PRNG = np.random.RandomState(d)
        nMatch = 0
        nRep = 10
        for rep in range(nRep):
            initPi_d = as1D(PRNG.dirichlet(K*np.ones(K), size=1))
            estPiFromRand, f2, I2 = estimatePi2(ids_d, cts_d, topics, alpha,
                scale=1.0, #/np.sum(cts_d),
                piInit=initPi_d,
                approx_grad=False)
            if np.allclose(estPi_d, estPiFromRand, rtol=0, atol=atol):
                nMatch += 1
            else:
                print("initrandom Pi[%d]:\n %s" % (
                    d, pi2str(estPiFromRand)))
                print(f)
                print(f2)
        print("%d/%d random inits within %s" % (nMatch, nRep, atol))
        
