'''
ToyMMSBK6.py
'''

import numpy as np
from bnpy.data import GraphXData

K = 6


def get_data(
        seed=123, nNodes=100, alpha=0.05,
        w_diag=.95,
        w_offdiag_eps=.01,
        **kwargs):
    ''' Create toy dataset as bnpy GraphXData object.

    Uses a simple mixed membership generative model.
    Assumes high within-block edge probability, small epsilon otherwise.

    Args
    -------
    seed : int
        seed for random number generator
    nNodes : int
        number of nodes in the generated network

    Returns
    -------
    Data : bnpy GraphXData object
    '''
    nNodes = int(nNodes)
    prng = np.random.RandomState(seed)

    # Create membership probabilities at each node
    if not hasattr(alpha, '__len__'):
        alpha = alpha * np.ones(K)
    pi = prng.dirichlet(alpha, size=nNodes)

    # Create block relation matrix W, shape K x K
    w = w_offdiag_eps * np.ones((K, K))
    w[np.diag_indices(6)] = w_diag

    # Generate community assignments, s, r, and pack into TrueZ
    s = np.zeros((nNodes, nNodes), dtype=int)
    r = np.zeros((nNodes, nNodes), dtype=int)
    for i in range(nNodes):
        s[i, :] = prng.choice(range(K), p=pi[i, :], size=nNodes)
        r[:, i] = prng.choice(range(K), p=pi[i, :], size=nNodes)
    TrueZ = np.zeros((nNodes, nNodes, 2), dtype=int)
    TrueZ[:, :, 0] = s
    TrueZ[:, :, 1] = r

    TrueParams = dict(Z=TrueZ, w=w, pi=pi)

    # Generate adjacency matrix
    AdjMat = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            if i == j:
                continue
            AdjMat[i,j] = prng.binomial(n=1, p=w[s[i, j], r[i, j]])

    Data = GraphXData(AdjMat=AdjMat,
                      nNodesTotal=nNodes, nNodes=nNodes,
                      TrueParams=TrueParams, isSparse=True)
    Data.name = get_short_name()
    return Data


def get_short_name():
    return 'ToyMMSBK6'


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Data = get_data(nNodes=100, alpha=0.5)
    w = Data.TrueParams['w']

    # # Draw graph with nodes colored by their mixed community membership
    # from bnpy.viz import RelationalViz
    # RelationalViz.plotTrueLabels(
    #  'ToyMMSBK6', Data,
    #  gtypes=['Actual'],
    #  mixColors=True, thresh=.65, colorEdges=False, title='ToyMMSBK6 Graph')

    # Plot adj matrix
    f, ax = plt.subplots(1)
    Xdisp = np.squeeze(Data.toAdjacencyMatrix())
    sortids = np.argsort(Data.TrueParams['pi'].argmax(axis=1))
    Xdisp = Xdisp[sortids, :]
    Xdisp = Xdisp[:, sortids]
    ax.imshow(
        Xdisp, cmap='Greys', interpolation='nearest',
        vmin=0, vmax=1)
    ax.set_title('Adjacency matrix')
    ax.set_yticks(np.arange(0, Data.nNodes+1, 10))
    ax.set_xticks(np.arange(0, Data.nNodes+1, 10))


    # Plot subset of pi
    Epi = Data.TrueParams['pi']
    fix, ax = plt.subplots(1)
    ax.imshow(
        Epi[0:30, :],
        cmap='Greys', interpolation='nearest',
        vmin=0, vmax=1.0)
    ax.set_ylabel('nodes')
    ax.set_yticks(np.arange(0, 31, 5))
    ax.set_xlabel('states')
    ax.set_xticks(np.arange(K))
    ax.set_title('Membership vectors pi (first 30 rows)')

    # Plot w
    fig, ax = plt.subplots(1)
    im = ax.imshow(
        w, cmap='Greys', interpolation='nearest',
        vmin=0, vmax=1.0)
    ax.set_xlabel('states')
    ax.set_xticks(np.arange(K))
    ax.set_ylabel('states')
    ax.set_yticks(np.arange(K))
    ax.set_title('Edge probability matrix w')

    plt.show()
