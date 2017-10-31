'''
NumericUtil.py

Library of efficient vectorized implementations of
 operations common to unsupervised machine learning

* inplaceExpAndNormalizeRows
* calcRlogR
* calcRlogRdotv
* calcRlogR_allpairs
* calcRlogR_specificpairs

'''
from __future__ import print_function
from builtins import *
import os
import configparser
import numpy as np
import scipy.sparse
import timeit

from .EntropyUtil import calcRlogR, calcRlogRdotv

def LoadConfig():
    global Config, cfgfilepath
    root = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
    cfgfilepath = os.path.join(root, 'config', 'numeric.conf')
    Config = readConfigFileIntoDict(cfgfilepath, 'LibraryPrefs')


def UpdateConfig(**kwargs):
    global Config
    for key in list(kwargs.keys()):
        if key in Config:
            Config[key] = kwargs[key]


def readConfigFileIntoDict(confFile, targetSecName=None):
    ''' Read contents of a config file into a dictionary

    Returns
    --------
    dict : dictionary of key-values for each configuration options
    '''
    config = configparser.SafeConfigParser()
    config.optionxform = str
    config.read(confFile)
    for secName in config.sections():
        if secName.count("Help") > 0:
            continue
        if targetSecName is not None:
            if secName != targetSecName:
                continue
        BigSecDict = dict(config.items(secName))
    return BigSecDict


def inplaceExp(R):
    ''' Calculate exp of each entry of input matrix, done in-place.
    '''
    if Config['inplaceExpAndNormalizeRows'] == "numexpr" and hasNumexpr:
        return inplaceExp_numexpr(R)
    else:
        return inplaceExp_numpy(R)


def inplaceExp_numpy(R):
    ''' Calculate exp of each entry of input matrix, done in-place.
    '''
    np.exp(R, out=R)


def inplaceExp_numexpr(R):
    ''' Calculate exp of each entry of input matrix, done in-place.
    '''
    ne.evaluate("exp(R)", out=R)


def inplaceLog(R):
    ''' Calculate log of each entry of input matrix, done in-place

    Post Condition
    --------------
    Provided array R will have each entry equal to log of its input value.

    Example
    -------
    >>> R = np.eye(2) + np.ones(2)
    >>> print R
    [[ 2.  1.]
     [ 1.  2.]]
    >>> inplaceLog(R) # Look Mom, no return value!
    >>> print R
    [[ 0.69314718  0.        ]
     [ 0.          0.69314718]]
    '''
    if Config['inplaceExpAndNormalizeRows'] == "numexpr" and hasNumexpr:
        return inplaceLog_numexpr(R)
    else:
        return inplaceLog_numpy(R)


def inplaceLog_numpy(R):
    ''' Calculate log of each entry of input matrix, done in-place.
    '''
    np.log(R, out=R)


def inplaceLog_numexpr(R):
    ''' Calculate log of each entry of input matrix, done in-place.
    '''
    ne.evaluate("log(R)", out=R)


def inplaceExpAndNormalizeRows(R, minVal=1e-40):
    ''' Compute exp(R), normalize rows to sum to one, and set min val.

    Post Condition
    --------
    Each row of R sums to one.
    Minimum value of R is equal to minVal.
    '''
    if Config['inplaceExpAndNormalizeRows'] == "numexpr" and hasNumexpr:
        inplaceExpAndNormalizeRows_numexpr(R)
    else:
        inplaceExpAndNormalizeRows_numpy(R)
    if minVal is not None:
        np.maximum(R, minVal, out=R)


def inplaceExpAndNormalizeRows_numpy(R):
    ''' Compute exp(R), normalize rows to sum to one, and set min val.

    Post Condition
    --------
    Each row of R sums to one.
    Minimum value of R is equal to minVal.
    '''
    R -= np.max(R, axis=1)[:, np.newaxis]
    np.exp(R, out=R)
    R /= R.sum(axis=1)[:, np.newaxis]


def inplaceExpAndNormalizeRows_numexpr(R):
    ''' Compute exp(R), normalize rows to sum to one, and set min val.

    Post Condition
    --------
    Each row of R sums to one.
    Minimum value of R is equal to minVal.
    '''
    R -= np.max(R, axis=1)[:, np.newaxis]
    ne.evaluate("exp(R)", out=R)
    R /= R.sum(axis=1)[:, np.newaxis]


def sumRtimesS(R, S):
    ''' Calculate sum along first axis of the product R times S

    Uses faster numexpr library if available, but safely falls back
    to plain numpy otherwise.

    Args
    --------
    R : 3D array, shape N x D1 x D2
    S : 3D array, shape N x D1 x D2

    Returns
    --------
    s : 2D array, size D1xD2
    '''
    if Config['calcRlogR'] == "numexpr" and hasNumexpr:
        return sumRtimesS_numexpr(R, S)
    else:
        return sumRtimesS_numpy(R, S)


def sumRtimesS_numpy(R, S):
    return np.sum(R * S, axis=0)


def sumRtimesS_numexpr(R, S):
    if R.shape[0] > 1:
        return ne.evaluate("sum(R*S, axis=0)")
    else:
        return sumRtimesS_numpy(R, S)


def calcRlogR_allpairs(R):
    ''' Calculate column sums of element-wise R*log(R) for all pairs.

    Uses faster numexpr library if available, but safely falls back
      to plain numpy otherwise.

    Returns
    --------
    Z : 2D array, size K x K
        Z[a,b] = sum(Rab*log(Rab)), Rab = R[:,a] + R[:,b]
        Only upper-diagonal entries of Z are non-zero,
        since we restrict potential pairs a,b to satisfy a < b
    '''
    if Config['calcRlogR'] == "numexpr" and hasNumexpr:
        return calcRlogR_allpairs_numexpr(R)
    else:
        return calcRlogR_allpairs_numpy(R)


def calcRlogR_allpairs_numpy(R):
    K = R.shape[1]
    Z = np.zeros((K, K))
    for jj in range(K - 1):
        curR = R[:, jj][:, np.newaxis] + R[:, jj + 1:]
        curR *= np.log(curR)
        Z[jj, jj + 1:] = np.sum(curR, axis=0)
    return Z


def calcRlogR_allpairs_numexpr(R):
    K = R.shape[1]
    Z = np.zeros((K, K))
    for jj in range(K - 1):
        curR = R[:, jj][:, np.newaxis] + R[:, jj + 1:]
        curZ = ne.evaluate("sum(curR * log(curR), axis=0)")
        Z[jj, jj + 1:] = curZ
    return Z


def calcRlogR_specificpairs(R, mPairs):
    ''' Calculate \sum_n R[n] log R[n]

    Uses faster numexpr library if available, but safely falls back
      to plain numpy otherwise.

    Args
    --------
    R : NxK matrix
    mPairs : list of possible merge pairs, where each pair is a tuple
               [(a,b), (c,d), (e,f)]

    Returns
    --------
    Hvec : 1D array, size M
        H[m] = sum(Rab*log(Rab))
        where Rab = R[:,a] + R[:,b], and (a,b) = mPairs[m]
    '''
    if Config['calcRlogR'] == "numexpr" and hasNumexpr:
        return calcRlogR_specificpairs_numexpr(R, mPairs)
    else:
        return calcRlogR_specificpairs_numpy(R, mPairs)


def calcRlogR_specificpairs_numpy(R, mPairs):
    K = R.shape[1]
    Hvec = np.zeros(len(mPairs))
    if K == 1:
        return Hvec
    for m, (kA, kB) in enumerate(mPairs):
        curR = R[:, kA] + R[:, kB]
        curR *= np.log(curR)
        Hvec[m] = np.sum(curR, axis=0)
    return Hvec

def calcRlogR_specificpairs_numexpr(R, mPairs):
    K = R.shape[1]
    Hvec = np.zeros(len(mPairs))
    if K == 1:
        return Hvec
    for m, (kA, kB) in enumerate(mPairs):
        curR = R[:, kA] + R[:, kB]
        Hvec[m] = ne.evaluate("sum(curR * log(curR), axis=0)")
    return Hvec


def calcRlogRdotv_allpairs(R, v):
    ''' Calculate dot product dot(v, Rm * log(Rm)) for all merge pairs.

    Uses faster numexpr library if available, but safely falls back
    to plain numpy otherwise.

    Args
    --------
    R : NxK matrix
    v : N-vector


    Returns
    --------
    Z : 2D array, size K x K
        Z[a,b] = sum(v * Rab*log(Rab)), Rab = R[:,a] + R[:,b]
        We restrict potential pairs a,b to satisfy a < b
    '''
    if Config['calcRlogRdotv'] == "numexpr" and hasNumexpr:
        return calcRlogRdotv_allpairs_numexpr(R, v)
    else:
        return calcRlogRdotv_allpairs_numpy(R, v)


def calcRlogRdotv_allpairs_numpy(R, v):
    K = R.shape[1]
    Z = np.zeros((K, K))
    for jj in range(K):
        curR = R[:, jj][:, np.newaxis] + R[:, jj + 1:]
        curR *= np.log(curR)
        Z[jj, jj + 1:] = np.dot(v, curR)
    return Z


def calcRlogRdotv_allpairs_numexpr(R, v):
    K = R.shape[1]
    Z = np.zeros((K, K))
    for jj in range(K - 1):
        curR = R[:, jj][:, np.newaxis] + R[:, jj + 1:]
        ne.evaluate("curR * log(curR)", out=curR)
        curZ = np.dot(v, curR)
        Z[jj, jj + 1:] = curZ
    return Z


def calcRlogRdotv_specificpairs(R, v, mPairs):
    ''' Calculate dot product dot(v, Rm * log(Rm)) for specific pairs.

        Uses faster numexpr library if available, but safely falls back
        to plain numpy otherwise.

        Args
        --------
        R : 2D array, size NxK
        v : 1D array, size N
        mPairs : list of possible merge pairs, where each pair is a tuple
            [(a,b), (c,d), (e,f)]

        Returns
        --------
        M : 1D array, size M
            Z[m] = sum(Rab*log(Rab)), Rab = R[:,a] + R[:,b]
    '''
    if Config['calcRlogRdotv'] == "numexpr" and hasNumexpr:
        return calcRlogRdotv_specificpairs_numexpr(R, v, mPairs)
    else:
        return calcRlogRdotv_specificpairs_numpy(R, v, mPairs)


def calcRlogRdotv_specificpairs_numpy(R, v, mPairs):
    K = R.shape[1]
    m_Hresp = np.zeros(len(mPairs))
    if K == 1:
        return m_Hresp
    for m, (kA, kB) in enumerate(mPairs):
        curWV = R[:, kA] + R[:, kB]
        curWV *= np.log(curWV)
        m_Hresp[m] = np.dot(v, curWV)
    return m_Hresp


def calcRlogRdotv_specificpairs_numexpr(R, v, mPairs):
    K = R.shape[1]
    m_Hresp = np.zeros(len(mPairs))
    if K == 1:
        return m_Hresp
    for m, (kA, kB) in enumerate(mPairs):
        curR = R[:, kA] + R[:, kB]
        ne.evaluate("curR * log(curR)", out=curR)
        m_Hresp[m] = np.dot(v, curR)
    return m_Hresp


def autoconfigure():
    ''' Perform timing experiments on current hardware to assess which
         of various implementations is the fastest for each key routine
    '''
    config = configparser.SafeConfigParser()
    config.optionxform = str
    config.read(cfgfilepath)
    methodNames = ['inplaceExpAndNormalizeRows', 'calcRlogR', 'calcRlogRdotv']
    for mName in methodNames:
        if mName == 'inplaceExpAndNormalizeRows':
            expectedGainFactor = _runTiming_inplaceExpAndNormalizeRows()
        elif mName == 'calcRlogR':
            expectedGainFactor = runTiming_calcRlogR()
        elif mName == 'calcRlogRdotv':
            expectedGainFactor = runTiming_calcRlogRdotv()
        print(mName, end=' ')
        if expectedGainFactor > 1.05:
            config.set('LibraryPrefs', mName, 'numexpr')
            print("numexpr preferred: %.2f X faster" % (expectedGainFactor))
        else:
            config.set('LibraryPrefs', mName, 'numpy')
            print("numpy preferred: %.2f X faster" % (expectedGainFactor))
        with open(cfgfilepath, 'w') as f:
            config.write(f)
    LoadConfig()


def _runTiming_inplaceExpAndNormalizeRows(N=2e5, D=100, repeat=3):
    if not hasNumexpr:
        return 0
    setup = "import numpy as np; import numexpr as ne;"
    setup += "from bnpy.util import NumericUtil as N;"
    setup += "R = np.random.rand(%d, %d)" % (N, D)
    elapsedTimes_np = timeit.repeat("N.inplaceExpAndNormalizeRows_numpy(R)",
                                    setup=setup, number=1, repeat=repeat)
    elapsedTimes_ne = timeit.repeat("N.inplaceExpAndNormalizeRows_numexpr(R)",
                                    setup=setup, number=1, repeat=repeat)
    meanTime_np = np.mean(elapsedTimes_np)
    meanTime_ne = np.mean(elapsedTimes_ne)
    expectedGainFactor = meanTime_np / meanTime_ne
    return expectedGainFactor


def _runTiming_calcRlogR(N=2e5, D=100, repeat=3):
    if not hasNumexpr:
        return 0
    setup = "import numpy as np; import numexpr as ne;"
    setup += "import bnpy.util.NumericUtil as N;"
    setup += "R = np.random.rand(%d, %d)" % (N, D)
    elapsedTimes_np = timeit.repeat("N.calcRlogR_numpy(R)",
                                    setup=setup, number=1, repeat=repeat)
    elapsedTimes_ne = timeit.repeat("N.calcRlogR_numexpr(R)",
                                    setup=setup, number=1, repeat=repeat)
    meanTime_np = np.mean(elapsedTimes_np)
    meanTime_ne = np.mean(elapsedTimes_ne)
    expectedGainFactor = meanTime_np / meanTime_ne
    return expectedGainFactor


def _runTiming_calcRlogRdotv(N=2e5, D=100, repeat=3):
    if not hasNumexpr:
        return 0
    setup = "import numpy as np; import numexpr as ne;"
    setup += "import bnpy.util.NumericUtil as N;"
    setup += "R = np.random.rand(%d, %d);" % (N, D)
    setup += "v = np.random.rand(%d)" % (N)
    elapsedTimes_np = timeit.repeat("N.calcRlogRdotv_numpy(R, v)",
                                    setup=setup, number=1, repeat=repeat)
    elapsedTimes_ne = timeit.repeat("N.calcRlogRdotv_numexpr(R, v)",
                                    setup=setup, number=1, repeat=repeat)
    meanTime_np = np.mean(elapsedTimes_np)
    meanTime_ne = np.mean(elapsedTimes_ne)
    expectedGainFactor = meanTime_np / meanTime_ne
    return expectedGainFactor

hasNumexpr = True
try:
    import numexpr as ne
except ImportError:
    hasNumexpr = False
if hasNumexpr and 'OMP_NUM_THREADS' in os.environ:
    try:
        nThreads = int(os.environ['OMP_NUM_THREADS'])
        ne.set_num_threads(nThreads)
    except TypeError as ValueError:
        print('Unrecognized OMP_NUM_THREADS', os.environ['OMP_NUM_THREADS'])
        pass

LoadConfig()
