'''
ToyMMSBK6.py
'''
import numpy as np
import scipy.io
from bnpy.data import GraphXData
from bnpy.viz import RelationalViz

# User-facing
###########################################################


def get_short_name():
    return 'ToyMMSBK6Gauss'


# Data generation
###########################################################
K = 6
delta = .1
epsilon = 1e-4
sigmas = np.ones(K) * 1.0
mus = [25, 35, 45, 55, 65, 75]


def get_data(
        seed=123, nNodes=100, alpha=0.05,
        epsilon=1e-4, delta=.1,
        **kwargs):
    ''' Create toy dataset as bnpy GraphXData object.

                Args
                -------
                seed : int
                                seed for random number generator
                nNodes : int
                                number of nodes in the generated network
                alpha : float
                                Controls the Dirichlet prior on pi, pi ~ Dir(alpha)
                epsilon : float
                                Probability that an edge representing an out of community
                                interaction will have a value outside [-delta, delta]
                delta : float
                                See above

                Returns
                -------
                Data : bnpy GraphXData object
        '''

    prng = np.random.RandomState(seed)
    np.random.seed(seed)

    # Create membership probabilities at each node
    N = nNodes
    if not hasattr(alpha, '__len__'):
        alpha = alpha * np.ones(K)
        pi = prng.dirichlet(alpha, size=nNodes)

        # Make source / receiver assignments and pack into TrueZ
        s = np.zeros((N, N), dtype=int)
        r = np.zeros((N, N), dtype=int)
        for i in range(N):
            s[i, :] = prng.choice(range(K), p=pi[i, :], size=nNodes)
            r[:, i] = prng.choice(range(K), p=pi[i, :], size=nNodes)
        TrueZ = np.zeros((N, N, 2), dtype=int)
        TrueZ[:, :, 0] = s
        TrueZ[:, :, 1] = r
        TrueParams = {'TrueZ': TrueZ, 'pi': pi, 'mu': mus, 'sigma': sigmas}

        # Generate graph
        X = np.zeros((N, N))
        cnt = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if s[i, j] == r[i, j]:
                    X[i, j] = np.random.normal(mus[s[i, j]], sigmas[s[i, j]])
                    cnt += 1

        M = np.max(np.abs(X))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if s[i, j] != r[i, j]:
                    inInterval = prng.binomial(n=1, p=1 - epsilon)
                    if inInterval:
                        X[i, j] = np.random.uniform(low=-delta, high=delta)
                    else:
                        negativeHalf = prng.binomial(n=1, p=.5)
                        if negativeHalf:
                            X[i, j] = np.random.uniform(low=-M, high=-delta)
                        else:
                            X[i, j] = np.random.uniform(low=delta, high=M)

        Data = GraphXData(AdjMat=X, X=None, edges=None,
                          nNodesTotal=nNodes, nNodes=nNodes,
                          TrueParams=TrueParams, isSparse=False)
        return Data

if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    Data = get_data()
    N = Data.nNodesTotal

    Epi = Data.TrueParams['pi']
    K = np.shape(Epi)[1]
    colors = np.sum(Epi * np.arange(K)[np.newaxis, :], axis=1)

    RelationalViz.plotTransMtxTruth(Data, doPerm=True)
    plt.show()
