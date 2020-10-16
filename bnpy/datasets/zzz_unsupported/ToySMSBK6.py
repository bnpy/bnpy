import numpy as np
from bnpy.data import GraphXData

K = 6


def get_data(
        seed=123, nNodes=100,
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
    prng = np.random.RandomState(seed)

    # Create membership probabilities at each node
    pi = 1.0 / K * np.ones(K)

    # Create block relation matrix W, shape K x K
    w = w_offdiag_eps * np.ones((K, K))
    w[np.diag_indices(K)] = w_diag

    # Generate node assignments
    Z = prng.choice(range(K), p=pi, size=nNodes)
    TrueParams = dict(Z=Z, w=w, pi=pi)

    # Generate edges
    AdjMat = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            if i != j:
                AdjMat[i, j] = prng.binomial(n=1, p=w[Z[i], Z[j]])

    Data = GraphXData(AdjMat=AdjMat,
                      nNodesTotal=nNodes,
                      TrueParams=TrueParams)
    Data.name = get_short_name()
    return Data


def get_short_name():
    return 'ToySMSBK6'


def get_data_info():
    return 'Toy SMSB dataset. K=%d' % (K)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bnpy.viz import RelationalViz

    Data = get_data(nNodes=100)
    w = Data.TrueParams['w']

    # # Plot illustrated graph
    # f, ax = plt.subplots(1)
    # RelationalViz.drawGraph(
    #    Data, fig=f, curAx=ax, colors=Data.TrueParams['Z'],)

    # Plot w
    fig, ax = plt.subplots(1)
    im = ax.imshow(
        w, cmap='Greys', interpolation='nearest',
        vmin=0, vmax=1.0)
    ax.set_xlabel('states')
    ax.set_xlabel('states')
    ax.set_title('Edge probability matrix w')

    # Plot adj matrix
    f, ax = plt.subplots(1)
    Xdisp = np.squeeze(Data.toAdjacencyMatrix())
    sortids = np.argsort(Data.TrueParams['Z'])
    Xdisp = Xdisp[sortids, :]
    Xdisp = Xdisp[:, sortids]
    ax.imshow(
        Xdisp, cmap='Greys', interpolation='nearest',
        vmin=0, vmax=1)
    ax.set_title('Adjacency matrix')

    plt.show()
