from builtins import *
import numpy as np
import scipy.optimize
import warnings
from .ShapeUtil import as1D, as2D


def estimatePiForDoc_frankwolfe(
        ids_U=None,
        cts_U=None,
        topics_KV=None,
        topics_KU=None,
        initpi_K=None,
        alpha=1.0,
        maxiter=500,
        seed=0,
        verbose=False,
        returnFuncValAndInfo=True,
        **kwargs):
    ''' Estimate topic-prob vector for doc using Frank-Wolke algorithm.

    Solves optimization problem
        piVec = min f(piVec)
    where
        f = -1 * \sum_{v=1}^V cts_d[v] \log \sum_{k=1}^K piVec[k] topics[k,v]
    which is equivalent to (up to additive constants indep. of piVec)
        piVec = min KL( cts_d[:] || \sum_k piVec[k] * topics[k,:] )

    Returns
    -------
    pi_K : 1D array, size K
    fval : optional, value of optimization objective function
    Info : optional, dict of information about optimization execution

    Resources
    ---------
    Dual online inference for latent Dirichlet allocation
    Than and Doan
    ACML 2014
    http://is.hust.edu.vn/~khoattq/papers/Than36.pdf
    '''
    PRNG = np.random.RandomState(seed)

    if topics_KU is None:
        # Create contiguous matrix for active vocab ids in this doc
        # Using .copy() here will produce 10x speed gains for each call to np.dot
        assert ids_U is not None
        assert topics_KV is not None
        topics_KU = topics_KV[:, np.asarray(ids_U, dtype=np.int32)].copy()
    assert topics_KU.ndim == 2
    K = topics_KU.shape[0]

    # Initialize doc-topic prob randomly, if needed
    if initpi_K is None:
        if PRNG is None:
            initpi_K = 1.0 / K * np.ones(K)
        else:
            initpi_K = make_random_pi_K(K=K, PRNG=PRNG)
    pi_K = initpi_K
    if verbose:
        print('  0 ' + ' '.join(['%.4f' % (p) for p in pi_K]))

    x_U = np.dot(pi_K, topics_KU)
    # Loop
    T_2 = [1, 0]
    for t in range(1, maxiter):
        # Pick a term uniformly at random
        T_2[PRNG.randint(2)] += 1
        # Select a vertex with the largest value of
        # derivative of the objective function
        df_K = T_2[0] * np.dot(topics_KU, cts_U / x_U) + \
             T_2[1] * (alpha - 1) / pi_K
        kmax = np.argmax(df_K)
        lrate = 1.0 / (t + 1)
        # Update probabilities
        pi_K *= 1 - lrate
        pi_K[kmax] += lrate
        # Update x
        x_U += lrate * (topics_KU[kmax,:] - x_U)
        # Print status
        if verbose and (t < 10 or t % 5 == 0):
            print('%3d ' % (t) + ' '.join(['%.4f' % (p) for p in pi_K]))

    if returnFuncValAndInfo:
        fval = -1 * np.inner(cts_U, np.log(np.dot(pi_K, topics_KU)))
        return (pi_K, fval, dict(
            niter=t,
            initpi_K=initpi_K))
    else:
        return pi_K


def estimatePiForDoc_graddescent(
        ids_U=None,
        cts_U=None,
        topics_KV=None,
        topics_KU=None,
        initpi_K=None,
        alpha=1.0,
        gtol=1e-7,
        maxiter=1000,
        method='l-bfgs-b',
        **kwargs):
    ''' Estimate topic-prob vector for doc using natural-parameter grad descent.

    Solves the optimization problem
        piVec = min f(piVec)
    where
        f =

    Returns
    -------
    piVec : 1D array, size K
    fval : scalar real
    Info : dict
    '''
    if topics_KU is None:
        assert topics_KV is not None
        assert ids_U is not None
        topics_KU = topics_KV[:, ids_U].copy()
    assert topics_KU.ndim == 2
    assert topics_KU.shape[1] == cts_U.shape[0]
    # Package up all local vars into kwargs dict
    kwargs.update(locals())
    with warnings.catch_warnings():
        warnings.filterwarnings('error', '', RuntimeWarning)
        try:
            pi_K, f, Info = _estimatePiForDoc(**kwargs)
        except RuntimeWarning as e:
            # Handle errors related to numerical overflow
            raise ValueError(e)

    if not Info.success:
        if 'approx_grad' in kwargs and not kwargs['approx_grad']:
            raise ValueError(Info.message)
    return pi_K, f, Info


def _estimatePiForDoc(
        cts_U=None,
        topics_KU=None,
        initpi_K=None,
        alpha=1.0,
        scale=1.0,
        method='l-bfgs-b',
        approx_grad=False,
        gtol=1e-9, # Values > 1e-6 can be bad for simplex
        maxiter=10000,
        options=None,
        PRNG=None,
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

    K = topics_KU.shape[0]
    if initpi_K is None:
        if PRNG is None:
            initpi_K = 1.0 / K * np.ones(K)
        else:
            initpi_K = make_random_pi_K(K=K, PRNG=PRNG)
    if approx_grad:
        def naturalLossFunc(eta_Km1):
            pi_K = eta2pi(eta_Km1)
            f = lossFunc(
                pi_K=pi_K,
                cts_U=cts_U,
                topics_KU=topics_KU, alpha=alpha, scale=scale)
            return f
        Result = scipy.optimize.minimize(
            naturalLossFunc,
            x0=pi2eta(initpi_K),
            jac=False,
            options=options,
            method=method,
            )
    else:
        def naturalLossFuncAndGrad(eta_Km1):
            pi_K = eta2pi(eta_Km1)
            f, grad_K = lossFuncAndGrad(
                pi_K=pi_K,
                cts_U=cts_U,
                topics_KU=topics_KU, alpha=alpha, scale=scale)
            grad_Km1 = np.dot(eta2piJacobian(pi_K=pi_K), grad_K)
            return f, grad_Km1
        Result = scipy.optimize.minimize(
            naturalLossFuncAndGrad,
            x0=pi2eta(initpi_K),
            jac=True,
            options=options,
            method=method,
            )
    f = Result.fun
    pi_K = eta2pi(Result.x)
    return pi_K, f, Result

def lossFuncAndGrad(
        pi_K=None,
        cts_U=None,
        topics_KU=None,
        alpha=1.0,
        scale=1.0,
        **kwargs):
    ''' Compute objective and gradient together.

    This uses the *natural* parameterization, where the random variable is
    a vector of real numbers whose corresponding mean parameter is pi_K.

    Minimization objective function:
        min f(pi_K)
    where
        f = - log MultPDF(cts_U | np.dot(pi_K, topics_KU)) \
            - log NaturalDirPDF( pi2eta(pi_K) | alpha)

    Returns
    -------
    f : scalar real
        value of negative log joint probability
    grad_K : 1D array, size K
        grad_K[k] : derivative of f w.r.t. pi_K[k]
    '''
    assert alpha > 0.0
    avgWordFreq_U = np.dot(pi_K, topics_KU)
    f_lik = np.inner(cts_U, np.log(avgWordFreq_U))
    grad_lik_K = np.dot(topics_KU, cts_U / avgWordFreq_U)

    f_prior = alpha * np.sum(np.log(pi_K + 1e-100))
    grad_prior_K = alpha / pi_K
    return (-1.0 * scale * (f_lik + f_prior),
            -1.0 * scale * (grad_lik_K + grad_prior_K))

def lossFunc(
        pi_K=None,
        cts_U=None,
        topics_KU=None,
        alpha=1.0,
        scale=1.0,
        **kwargs):
    ''' Compute objective function for document-topic probabilities.

    Args
    ----
    pi_K : 1D array, size K
        pi_K[k] : probability of k-th topic in doc d
    cts_U : 1D array, size U
        cts_U[i] : count of i-th unique-word in doc d
    topics_KU : 2D array, K x U
        topics_KU[k,i]: probability of i-th unique-word in doc d under topic k
    alpha : scalar float >= 0

    Returns
    -------
    f : scalar real
        value of negative log joint probability
        suitable for minimization algorithms
    '''
    f_lik = np.inner(cts_U, np.log(np.dot(pi_K, topics_KU)))
    f_prior = alpha * np.sum(np.log(pi_K + 1e-100))
    return -1.0 * scale * (f_lik + f_prior)

def gradOfLoss(
        pi_K=None,
        cts_U=None,
        topics_KU=None,
        alpha=0.0,
        scale=1.0,
        **kwargs):
    ''' Compute gradient of objective function

    Returns
    -------
    grad : 1D array, size K
        grad[k] gives the derivative w.r.t. pi_K[k] of f
    '''
    # avgWordFreq_U : 1D array, size U_d
    #   avgWordFreq_U[i] = probability of word i using mixture of topics
    avgWordFreq_U = np.dot(pi_K, topics_KU)
    grad_lik_K = np.dot(topics_KU, cts_U / avgWordFreq_U)
    grad_prior_K = alpha / pi_K
    return -1 * scale * (grad_lik_K + grad_prior_K)

def pi2eta(pi_K):
    ''' Transform vector on simplex to unconstrained real vector

    Returns
    -------
    eta_Km1 : 1D array, size K-1

    Examples
    --------
    >>> print float(pi2eta(eta2pi(0.42)))
    0.42

    >>> print float(pi2eta(eta2pi(-1.337)))
    -1.337

    >>> print pi2eta(eta2pi([-1, 0, 1]))
    [-1.  0.  1.]
    '''
    pi_K = as1D(np.asarray(pi_K))
    eta_Km1 = pi_K[:-1] / pi_K[-1]
    np.log(eta_Km1, out=eta_Km1)
    return eta_Km1

def eta2pi(eta_Km1):
    eta_Km1 = as1D(np.asarray(eta_Km1))
    pi_K = np.ones(eta_Km1.size+1)
    pi_K[:-1] = np.exp(eta_Km1)
    pi_K[:-1] += 1e-100
    pi_K /= (1.0 + np.sum(pi_K[:-1]))
    return pi_K

def eta2piJacobian(eta_Km1=None, pi_K=None):
    ''' Compute Jacobian matrix of transformation of eta to pi

    Returns
    -------
    J_Km1K : 2D array, size K-1 x K
        J[a, b] = deriv of pi_K{b}(eta_Km1) w.r.t. eta_Km1[a]
    '''
    if pi_K is None:
        pi_K = eta2pi(eta_Km1)
    J_Km1K = -1.0 * np.outer(pi_K[:-1], pi_K)
    J_Km1K[:, :-1] += np.diag(pi_K[:-1])
    return J_Km1K

def pi2str(arr):
    pistr = np.array_str(arr,
        max_line_width=80, precision=4, suppress_small=1)
    return pistr.replace('[','').replace(']','')

def calcLossFuncForInterpolatedPi(piA_K, piB_K, lossFunc, nGrid=100):
    wgrid = np.linspace(0, 1.0, nGrid)
    fgrid = np.zeros(nGrid)
    for ii in range(nGrid):
        pi_K = wgrid[ii] * piA_K + (1.0 - wgrid[ii]) * piB_K
        f = lossFunc(pi_K)
        if isinstance(f, tuple):
            f = f[0]
        fgrid[ii] = f
    return fgrid, wgrid

def make_random_pi_K(K=2, seed=0, PRNG=None):
    if PRNG is None:
        PRNG = np.random.RandomState(seed)
    initpi_K = PRNG.rand(K) + 1.
    initpi_K /= sum(initpi_K)
    return initpi_K

if __name__ == '__main__':
    import bnpy
    PRNG = np.random.RandomState(0)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataName', default='CleanBarsK10')
    parser.add_argument('--optim_method', default='frankwolfe',
        choices=['frankwolfe', 'graddescent'])
    parser.add_argument('--nDocTotal', default=10)
    args = parser.parse_args()

    # Load data
    datamod = __import__(args.dataName, fromlist=[])
    Data = datamod.get_data(nDocTotal=args.nDocTotal)
    # Select function
    estimatePiForDoc = locals()['estimatePiForDoc_' + args.optim_method]

    if hasattr(Data, 'TrueParams'):
        topics_KV = Data.TrueParams['topics']
        K, V = topics_KV.shape
        nudgeVals_KV = PRNG.dirichlet(
            0.1 * np.ones(V),
            size=K)
        nudgedTopics_KV = 0.98 * topics_KV + 0.02 * nudgeVals_KV
        topics_KV = np.vstack([topics_KV, nudgedTopics_KV])
    else:
        K = np.minimum(50, Data.nDoc)
        chosenDocIDs = PRNG.choice(Data.nDoc, K, replace=False)
        topics_KV = Data.getSparseDocTypeCountMatrix()[chosenDocIDs].toarray()
        topics_KV += 0.1
        topics_KV /= topics_KV.sum(axis=1)[:,np.newaxis]

    assert np.allclose(topics_KV.sum(axis=1), 1.0)

    atol = 1e-3
    K, V = topics_KV.shape
    alpha = 1.0 / K

    for d in range(Data.nDoc):
        start = Data.doc_range[d]
        stop = Data.doc_range[d+1]
        ids_U = Data.word_id[start:stop]
        cts_U = Data.word_count[start:stop]
        scale = 1.0

        if hasattr(Data, 'TrueParams'):
            trueDTC_K = np.dot(cts_U, Data.TrueParams['resp'][start:stop])
            truePi_K = (trueDTC_K + alpha)
            truePi_K /= truePi_K.sum()
            print('')
            print("     True Pi[%d]:\n %s" % (d, pi2str(truePi_K)))

        if not args.optim_method.count("frankwolfe"):
            numPi_K, numf, numInfo = estimatePiForDoc(
                ids_U=ids_U,
                cts_U=cts_U,
                topics_KV=topics_KV,
                alpha=alpha,
                scale=scale,
                approx_grad=True)
            print("Numerical Pi[%d]:\n %s" % (d, pi2str(numPi_K)))

        estPi_K, estf, estInfo = estimatePiForDoc(
            ids_U=ids_U,
            cts_U=cts_U,
            topics_KV=topics_KV,
            alpha=alpha,
            scale=scale,
            )
        print("Estimated Pi[%d]:\n %s" % (d, pi2str(estPi_K)))


        # Generate random initializations,
        # and look at convergence properties
        PRNG = np.random.RandomState(d)
        nMatch = 0
        nRep = 10
        for rep in range(nRep):
            initpi_K = make_random_pi_K(PRNG=PRNG, K=K)
            pi_K, f, Info = estimatePiForDoc(
                ids_U=ids_U,
                cts_U=cts_U,
                topics_KV=topics_KV,
                alpha=alpha,
                scale=scale,
                initpi_K=initpi_K)
            if np.allclose(estPi_K, pi_K, rtol=0, atol=atol):
                nMatch += 1
            else:
                print("initrandom Pi[%d]:\n %s" % (
                    d, pi2str(pi_K)))
                print(estf)
                print(f)
        print("%d/%d random inits within %s" % (nMatch, nRep, atol))
