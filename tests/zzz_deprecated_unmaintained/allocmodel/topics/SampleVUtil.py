import numpy as np
import warnings


def sampleVd(u, nDoc=100, alpha=0.5, PRNG=np.random.RandomState(0)):
    ''' Draw document-level stick lengths Vd given global stick lengths u

        Returns
        ---------
        Vd : 2D array, D x K
             Vd[d,k] = prob of choosing k at doc d, among {k, k+1, ... K, K+1}
    '''
    K = u.size
    cumprod1mu = np.ones(K)
    cumprod1mu[1:] *= np.cumprod(1 - u[:-1])

    Vd = np.zeros((nDoc, K))
    for k in range(K):
        Vd[:, k] = PRNG.beta(alpha * cumprod1mu[k] * u[k],
                             alpha * cumprod1mu[k] * (1. - u[k]),
                             size=nDoc)
        # Warning: beta rand generator can fail when both params
        # are very small (~1e-8). This will yield NaN values.
        # To fix, we use fact that Beta(eps, eps) will always yield a 0 or 1.
        badIDs = np.flatnonzero(np.isnan(Vd[:, k]))
        if len(badIDs) > 0:
            p = np.asarray([1. - u[k], u[k]])
            Vd[badIDs, k] = PRNG.choice([1e-12, 1 - 1e-12],
                                        len(badIDs), p=p, replace=True)
    assert not np.any(np.isnan(Vd))
    assert np.all(np.isfinite(Vd))
    return Vd


def summarizeVdToPi(Vd):
    ''' Calculate summary vector of given doc-topic stick lengths Vd

        Returns
        --------
        sumLogPi : 1D array, size K+1
                    sumELogPi[k] = \sum_d log pi_{dk}
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='divide by zero')
        logVd = np.log(Vd)
        log1mVd = np.log(1 - Vd)
        mask = Vd < 1e-15
        log1mVd[mask] = np.log1p(-1 * Vd[mask])

    assert not np.any(np.isnan(logVd))
    logVd = replaceInfVals(logVd)
    log1mVd = replaceInfVals(log1mVd)
    sumlogVd = np.sum(logVd, axis=0)
    sumlog1mVd = np.sum(log1mVd, axis=0)
    sumlogPi = np.hstack([sumlogVd, 0])
    sumlogPi[1:] += np.cumsum(sumlog1mVd)
    return sumlogPi


def summarizeVd(Vd):
    ''' Calculate summary vector of given doc-topic stick lengths Vd

        Returns
        --------
        sumLogV : 1D array, size K
                    sumELogV[k] = \sum_d log v_{dk}
        sumLog1mV : 1D array, size K
                     sumELog1mV[k] = \sum_d log 1-v_{dk}
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='divide by zero')
        logVd = np.log(Vd)
        log1mVd = np.log(1 - Vd)

    assert not np.any(np.isnan(logVd))
    logVd = replaceInfVals(logVd)
    log1mVd = replaceInfVals(log1mVd)
    return np.sum(logVd, axis=0), np.sum(log1mVd, axis=0)


def replaceInfVals(logX, replaceVal=-100):
    infmask = np.isinf(logX)
    logX[infmask] = replaceVal
    return logX


def summarizeVdToDocTopicCount(Vd):
    ''' Create DocTopicCount matrix from given stick-breaking parameters Vd

        Returns
        --------
        DocTopicCount : 2D array, size D x K
    '''
    assert not np.any(np.isnan(Vd))
    PRNG = np.random.RandomState(0)
    DocTopicCount = np.zeros(Vd.shape)
    for d in range(Vd.shape[0]):
        N_d = 100 + 50 * PRNG.rand()
        Pi_d = Vd[d, :].copy()
        Pi_d[1:] *= np.cumprod(1.0 - Vd[d, :-1])
        np.maximum(Pi_d, 1e-10, out=Pi_d)
        Pi_d /= np.sum(Pi_d)
        DocTopicCount[d, :] = N_d * Pi_d
    return DocTopicCount
