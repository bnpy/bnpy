'''
ToyFox3: 1-st order auto-regressive toy dataset

3 sequences, 4 states

From the BP-AR-HMM paper
Fox, Sudderth, Willsky, Jordan
NIPS 2009
'''
import numpy as np
from bnpy.data import GroupXData
from bnpy.viz import GaussViz

DEFAULT_SEED = 123
DEFAULT_LEN = 2000


def get_data(seed=DEFAULT_SEED, T=DEFAULT_LEN, **kwargs):
    ''' Generate toy data sequences, returned as a bnpy data-object

      Args
      -------
      seed : integer seed for random number generator,
              used for actually *generating* the data
      T : int number of observations in each sequence

      Returns
      -------
      Data : bnpy GroupXData object, with nObsTotal observations
    '''
    X, Xprev, Z, doc_range = get_X(seed, T)

    nUsedStates = len(np.unique(Z))
    if nUsedStates < K:
        print('WARNING: NOT ALL TRUE STATES USED IN GENERATED DATA')

    Data = GroupXData(X=X, Xprev=Xprev, doc_range=doc_range, TrueZ=Z)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def get_short_name():
    return 'ToyFox3'


def get_data_info():
    return 'Toy sequential data with unique state in third sequence.'

D = 1
K = 4


def makeStickyTransMatrix(K, pSame=0.75, T=50):
    # Find self-trans prob such that
    # 75% chance of staying in state after 50 timesteps
    stickyP = np.exp(np.log(pSame) / T)
    transPi = (1 - stickyP) / (K - 1) * np.ones((K, K))

    for k in range(K):
        transPi[k, k] = stickyP

    assert np.allclose(1.0, transPi.sum(axis=1))
    return transPi

# Means for each component
A_master = np.asarray([-0.8, -0.4, 0.8, -0.3])
Sigma_master = None


def get_X(seed, T=DEFAULT_LEN):
    ''' TODO
    '''
    prng = np.random.RandomState(seed)

    # 3 sequences, all the same length (=T)
    Ts = int(T) * np.ones(3, dtype=np.int32)

    # Covariance for each component
    # Covariance for all states drawn from InvWishart(0.5, deg-free=3) prior
    # This is the same as an inverse gamma with a = 3/2, b= 0.25 (=.5/2)
    # which is the same as a gamma on precision, with a = 3/2, b = 4
    invSigmas = prng.gamma(3.0 / 2, 4, size=K)
    Sigmas = 1.0 / invSigmas
    global Sigma_master
    Sigma_master = Sigmas

    # Each iteration generates one time-series/sequence
    # with starting state deterministically rotating among all states
    Z1, X1, Xprev1 = generateSequence_ZandX(
        T, [0, 1, 2], A_master, Sigmas, prng)
    Z2, X2, Xprev2 = generateSequence_ZandX(
        T, [0, 1, 2], A_master, Sigmas, prng)
    Z3, X3, Xprev3 = generateSequence_ZandX(
        T / 4, [2, 3], A_master, Sigmas, prng)

    Z = np.hstack([Z1, Z2, Z3])
    X = np.hstack([X1, X2, X3])
    X = X[:, np.newaxis]

    Xprev = np.hstack([Xprev1, Xprev2, Xprev3])
    Xprev = Xprev[:, np.newaxis]

    Ts = [Z1.size, Z2.size, Z3.size]
    seq_indptr = np.hstack([0, np.cumsum(Ts)])
    return X, Xprev, Z, seq_indptr


def generateSequence_ZandX(T, activeStateIDs, A_all, Sigma_all, prng):
    ''' Generate single sequence

        Returns
        ----------
        Z : 1D array, size T
            state assignments at each timestep
        X : 1D array, size T
         data for each timestep
        Xprev : 1D array, size T
         previous data for each timestep
    '''
    X = np.zeros(T + 1)
    Z = np.zeros(T + 1, dtype=np.int32)
    K = len(activeStateIDs)

    # Emission parameters
    A = A_all[activeStateIDs]
    Sigma = Sigma_all[activeStateIDs]

    # Transition matrix
    transPi = makeStickyTransMatrix(K)

    # Generate sequence
    Z[0] = 0
    X[0] = 0
    for t in range(1, T):
        Z[t] = prng.choice(range(K), p=transPi[Z[t - 1]])

        X[t] = prng.normal(A[Z[t]] * X[t - 1],
                           np.sqrt(Sigma_all[Z[t]]))

    # Relabel the Z states to have values in activeStateIDs
    # Z_all[t] \in activeStateIDs
    # Z[t] \in [0, 1,... K-1]
    Z_all = np.zeros_like(Z)
    for j, k in enumerate(activeStateIDs):
        mask = Z == j
        Z_all[mask] = k
    return Z_all[1:], X[1:], X[:-1]


if __name__ == '__main__':
    from matplotlib import pylab

    Data = get_data(DEFAULT_SEED)

    B = np.max(np.abs(Data.X))
    Tmax = np.max(Data.doc_range[1:] - Data.doc_range[:-1])
    for n in range(Data.nDoc):
        start = Data.doc_range[n]
        stop = Data.doc_range[n + 1]

        X = Data.X[start:stop]
        Z = Data.TrueParams['Z'][start:stop]

        IDs0 = np.flatnonzero(Z == 0)
        IDs1 = np.flatnonzero(Z == 1)
        IDs2 = np.flatnonzero(Z == 2)
        IDs3 = np.flatnonzero(Z == 3)

        pylab.subplot(3, 1, n + 1)
        pylab.plot(IDs0, X[IDs0], 'r.')
        pylab.plot(IDs1, X[IDs1], 'b.')
        pylab.plot(IDs2, X[IDs2], 'g.')
        pylab.plot(IDs3, X[IDs3], 'y.')

        pylab.ylim([-B, B])
        pylab.xlim([0, Tmax + 1])

    # Print out a ML estimate of each state's parameters
    import bnpy
    obsPriorParams = dict(ECovMat='eye', nu=1, sF=1, sV=1)
    oModel = bnpy.obsmodel.AutoRegGaussObsModel('EM', Data=Data,
                                                min_covar=1e-8,
                                                **obsPriorParams)
    TrueZ = Data.TrueParams['Z'].copy()
    resp = np.zeros((TrueZ.size, np.max(TrueZ) + 1))
    for t in range(TrueZ.size):
        resp[t, TrueZ[t]] = 1.0
    LP = dict(resp=resp)
    SS = bnpy.suffstats.SuffStatBag(K=K, D=D)
    SS = oModel.calcSummaryStats(Data, SS, LP)
    oModel.update_global_params(SS)

    print('A truth')
    print(A_master)
    print('A estimated')
    print(oModel.EstParams.A.flatten())
    print('')
    print('Sigma truth')
    print(Sigma_master)
    print('Sigma estimated')
    print(oModel.EstParams.Sigma.flatten())

    pylab.show(block=True)
