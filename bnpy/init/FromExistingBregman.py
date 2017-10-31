from builtins import *
import numpy as np
from bnpy.util import split_str_into_fixed_width_lines

def runKMeans_BregmanDiv_existing(
        X, Kfresh, obsModel,
        W=None,
        Niter=100,
        seed=0,
        smoothFracInit=1.0,
        smoothFrac=0,
        logFunc=None,
        eps=1e-10,
        assert_monotonic=True,
        setOneToPriorMean=0,
        distexp=1.0,
        init='plusplus',
        noiseSD=0.0,
        **kwargs):
    ''' Run clustering algorithm to add Kfresh new clusters to existing model.

    Given an existing model with Korig clusters,
    We first initialize K brand-new clusters via Bregman kmeans++.
    Next, we run Niter iterations of coordinate ascent, which iteratively
    updates the assignments of data to clusters, and then updates cluster means.

    Importantly, *only* the new clusters have their mean parameters updated.
    Existing clusters are *fixed* to values given by provided obsModel.

    Returns
    -------
    Z : 1D array, size N
        Contains assignments to Korig + K possible clusters
        if Niter == 0, unassigned data items have value Z[n] = -1

    Mu : 2D array, size (Korig + Kfresh) x D
        Includes original Korig clusters and Kfresh new clusters

    Lscores : 1D array, size Niter
    '''
    Korig = obsModel.K
    obsModel.Prior.B += noiseSD**2 * np.eye(obsModel.D)
    chosenZ, Mu, _, _ = initKMeans_BregmanDiv_existing(
        X, Kfresh, obsModel,
        W=W,
        seed=seed,
        smoothFrac=smoothFracInit,
        distexp=distexp,
        noiseSD=noiseSD)
    # Make sure we update K to reflect the returned value.
    # initKMeans_BregmanDiv will return fewer than K clusters
    # in some edge cases, like when data matrix X has duplicate rows
    # and specified K is larger than the number of unique rows.
    Kfresh = len(Mu) - Korig
    assert Kfresh > 0
    assert Niter >= 0
    if Niter == 0:
        Z = -1 * np.ones(X.shape[0])
        if chosenZ[0] == -1:
            Z[chosenZ[1:]] = Korig + np.arange(chosenZ.size - 1)
        else:
            Z[chosenZ] = Korig + np.arange(chosenZ.size)

    # Run coordinate ascent,
    # Alternatively these updates:
    # Local step: set assignment of every data point to its nearest cluster
    # (partial) global step: update mean parameters for Kfresh new clusters
    Lscores = list()
    prevN_K = np.zeros(Korig + Kfresh)
    for riter in range(Niter):
        Div = obsModel.calcSmoothedBregDiv(
            X=X, Mu=Mu, W=W,
            includeOnlyFastTerms=True,
            smoothFrac=smoothFrac, eps=eps)
        Z = np.argmin(Div, axis=1)
        Ldata = Div.min(axis=1).sum()
        Lprior = obsModel.calcBregDivFromPrior(
            Mu=Mu, smoothFrac=smoothFrac).sum()
        Lscore = Ldata + Lprior
        Lscores.append(Lscore)
        if assert_monotonic:
            # Verify objective is monotonically increasing
            try:
                # Test allows small positive increases that are
                # numerically indistinguishable from zero. Don't care about these.
                assert np.all(np.diff(Lscores) <= 1e-5)
            except AssertionError:
                msg = 'In the kmeans update loop of FromScratchBregman.py'
                msg += 'Lscores not monotonically decreasing...'
                if logFunc:
                    logFunc(msg)
                else:
                    print(msg)
                assert np.all(np.diff(Lscores) <= 1e-5)

        curN_K = np.zeros(Korig + Kfresh)
        for k in range(Korig + Kfresh):
            if W is None:
                W_k = None
                curN_K[k] = np.sum(Z==k)
            else:
                W_k = W[Z==k]
                curN_K[k] = np.sum(W_k)

            # Update mean parameters
            if k >= Korig:
                if curN_K[k] > 0:
                    Mu[k] = obsModel.calcSmoothedMu(X[Z==k], W_k)
                else:
                    Mu[k] = obsModel.calcSmoothedMu(X=None)
        if logFunc:
            logFunc("iter %d: Lscore %.3e" % (riter, Lscore))
            def countvec2str(curN_K):
                if W is None:
                     str_sum_w = ' '.join(['%7.0f' % (x) for x in curN_K])
                else:
                     assert np.allclose(curN_K.sum(), W.sum())
                     str_sum_w = ' '.join(['%7.2f' % (x) for x in curN_K])
                return split_str_into_fixed_width_lines(
                    str_sum_w, tostr=True)


            str_sum_w_exist = countvec2str(curN_K[:Korig])
            logFunc(str_sum_w_exist)

            str_sum_w_fresh = countvec2str(curN_K[Korig:])
            logFunc(str_sum_w_fresh)

        if np.max(np.abs(curN_K - prevN_K)) == 0:
            break
        prevN_K[:] = curN_K

    if Niter > 0:
        # In case a new cluster (index Korig, Korig+1, ... Korig+Kfresh)
        # has mass pushed to zero, we delete it.
        # All original clusters (index 0, 1, ... Korig) remain untouched.
        for k in reversed(range(Korig, Korig + Kfresh)):
            if curN_K[k] == 0:
                del(Mu[k])
                Z[Z > k] -= 1
        Kfreshnonzero = np.unique(Z[Z >= Korig]).size
        assert len(Mu) == Korig + Kfreshnonzero
    else:
        # Did not do a full assignment step, so many items are unassigned
        # This is indicated with Z value of -1.
        Kfreshnonzero = np.unique(Z[Z >= Korig]).size
        assert Kfreshnonzero == Kfresh
        assert len(Mu) == Korig + Kfresh
    obsModel.Prior.B -= noiseSD**2 * np.eye(obsModel.D)
    return Z, Mu, np.asarray(Lscores)

def initKMeans_BregmanDiv_existing(
        X, K, obsModel,
        W=None,
        seed=0,
        smoothFrac=1.0,
        distexp=1.0,
        noiseSD=0.0):
    ''' Initialize cluster means Mu with existing clusters and K new clusters.

    Returns
    -------
    chosenZ : 1D array, size K
        int ids of atoms selected
    Mu : list of size Kexist + K
        each entry is a tuple of ND arrays
    minDiv : 1D array, size N
    '''
    PRNG = np.random.RandomState(int(seed))
    N = X.shape[0]
    if W is None:
        W = np.ones(N)

    # Create array to hold chosen data atom ids
    chosenZ = np.zeros(K, dtype=np.int32)

    # Initialize list Mu to hold all mean vectors
    # First obsModel.K entries go to existing clusters found in the obsModel.
    # Final K entries are placeholders for the new clusters we'll make below.
    Mu = [obsModel.getSmoothedMuForComp(k) + noiseSD**2*np.eye(obsModel.D) for k in range(obsModel.K)]
    Mu.extend([None for k in range(K)])

    # Compute minDiv between all data and existing clusters
    minDiv, DivDataVec = obsModel.calcSmoothedBregDiv(
        X=X, Mu=Mu[:obsModel.K], W=W,
        returnDivDataVec=True,
        return1D=True,
        smoothFrac=smoothFrac)

    # Sample each cluster id using distance heuristic
    for k in range(0, K):
        sum_minDiv = np.sum(minDiv)
        if sum_minDiv == 0.0:
            # Duplicate rows corner case
            # Some rows of X may be exact copies,
            # leading to all minDiv being zero if chosen covers all copies
            chosenZ = chosenZ[:k]
            for emptyk in reversed(list(range(k, K))):
                # Remove remaining entries in the Mu list,
                # so its total size is now k, not K
                Mu.pop(emptyk)
            # Escape loop to return statement below
            break

        elif sum_minDiv < 0 or not np.isfinite(sum_minDiv):
            raise ValueError("sum_minDiv not valid: %f" % (sum_minDiv))

        if distexp >= 9:
            chosenZ[k] = np.argmax(minDiv)
        else:
            if distexp > 1:
                minDiv = minDiv**distexp
                sum_minDiv = np.sum(minDiv)
            pvec = minDiv / sum_minDiv
            chosenZ[k] = PRNG.choice(N, p=pvec)

        # Compute mean vector for chosen data atom
        # Then add to the list
        Mu_k = obsModel.calcSmoothedMu(X[chosenZ[k]], W=W[chosenZ[k]])
        Mu[obsModel.K + k] = Mu_k

        # Performe distance calculation for latest chosen mean vector
        curDiv = obsModel.calcSmoothedBregDiv(
            X=X, Mu=Mu_k, W=W,
            DivDataVec=DivDataVec,
            return1D=True,
            smoothFrac=smoothFrac)
        # Enforce chosen data atom has distance 0
        # so we cannot pick it again
        curDiv[chosenZ[k]] = 0
        # Update distance between each atom and its nearest cluster
        minDiv = np.minimum(minDiv, curDiv)

    # Some final verification
    assert len(Mu) == chosenZ.size + obsModel.K
    return chosenZ, Mu, minDiv, np.sum(DivDataVec)
