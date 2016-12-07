'''
GaussViz.py

Visualizing learned Gaussian mixture models.
'''
import numpy as np
from PlotUtil import pylab

from bnpy.util import as1D, as2D

Colors = [(1, 0, 0),
          (1, 0, 1),
          (0, 1, 0),
          (0, 1, 1),
          (0, 0, 1),
          (1, 0.6, 0),
          (1, 0, 0.5),
          (0.5, 0.8, 0.8)]


def plotGauss1DFromHModel(hmodel,
                          compListToPlot=None,
                          compsToHighlight=None,
                          activeCompIDs=None,
                          MaxKToDisplay=50,
                          proba_thr=0.0001,
                          ax_handle=None,
                          Colors=Colors,
                          dataset=None,
                          **kwargs):
    ''' Make line plot of pdf for each component (1D observations).
    '''
    if ax_handle is not None:
        pylab.sca(ax_handle)

    if compsToHighlight is not None:
        compsToHighlight = as1D(np.asarray(compsToHighlight))
    else:
        compsToHighlight = list()
    if compListToPlot is None:
        compListToPlot = np.arange(0, hmodel.obsModel.K)
    if activeCompIDs is None:
        activeCompIDs = np.arange(0, hmodel.obsModel.K)

    # Load appearance probabilities as single vector
    if hmodel.allocModel.K == hmodel.obsModel.K:
        w = hmodel.allocModel.get_active_comp_probs()
    else:
        w = np.ones(hmodel.obsModel.K)

    if dataset is not None:
        if hasattr(dataset, 'X'):
            pylab.hist(dataset.X[:, 0], 50, normed=1)
            #Xtile = np.tile(Data.X[:, 0], (2, 1))
            #ys = 0.1 * np.arange(2)
            #pylab.plot(Xtile, ys, 'k-')

    nSkip = 0
    nGood = 0
    for ii, compID in enumerate(compListToPlot):
        if compID not in activeCompIDs:
            continue

        kk = np.flatnonzero(activeCompIDs == compID)
        assert kk.size == 1
        kk = kk[0]

        if w[kk] < proba_thr and compID not in compsToHighlight:
            nSkip += 1
            continue

        mu = hmodel.obsModel.get_mean_for_comp(kk)
        sigma2 = hmodel.obsModel.get_covar_mat_for_comp(kk)

        if len(compsToHighlight) == 0 or compID in compsToHighlight:
            color = Colors[ii % len(Colors)]
            plotGauss1D(mu, sigma2, color=color)
        elif kk not in compsToHighlight:
            plotGauss1D(mu, sigma2, color='k')

        nGood += 1
        if nGood >= MaxKToDisplay:
            print 'DISPLAY LIMIT EXCEEDED. Showing %d/%d components' \
                % (nGood, len(activeCompIDs))
            break
    if nSkip > 0:
        print 'SKIPPED %d comps with size below %.2f' % (nSkip, proba_thr)


def plotGauss1D(mu, sigma2, color='b', ax_handle=None, **kwargs):
    if ax_handle is not None:
        pylab.sca(ax_handle)

    mu = np.squeeze(mu)
    sigma = np.sqrt(np.squeeze(sigma2))

    assert mu.size == 1 and mu.ndim == 0
    assert sigma.size == 1 and sigma.ndim == 0

    xs = mu + sigma * np.arange(-4, 4, 0.01)
    ps = 1. / np.sqrt(2 * np.pi) * 1. / sigma * \
        np.exp(-0.5 * (xs - mu)**2 / sigma**2)
    pylab.plot(xs, ps, '.', markerfacecolor=color, markeredgecolor=color)


def plotGauss2DFromHModel(
        hmodel,
        compListToPlot=None,
        compsToHighlight=None,
        activeCompIDs=None,
        MaxKToDisplay=50,
        proba_thr=0.0001,
        ax_handle=None,
        dataset=None,
        Colors=Colors,
        **kwargs):
    ''' Plot 2D contours for components in hmodel in current pylab figure

    Args
    ----
    hmodel : bnpy HModel object
    compListToPlot : array-like of integer IDs of components within hmodel
    compsToHighlight : int or array-like
        integer IDs to highlight
        if None, all components get unique colors
        if not None, only highlighted components get colors.
    proba_thr : float
        Minimum weight assigned to component in order to be plotted.
        All components with weight below proba_thr are ignored.
    '''
    if ax_handle is not None:
        pylab.sca(ax_handle)

    if compsToHighlight is not None:
        compsToHighlight = np.asarray(compsToHighlight)
        if compsToHighlight.ndim == 0:
            compsToHighlight = np.asarray([compsToHighlight])
    else:
        compsToHighlight = list()
    if compListToPlot is None:
        compListToPlot = np.arange(0, hmodel.obsModel.K)
    if activeCompIDs is None:
        activeCompIDs = np.arange(0, hmodel.obsModel.K)

    # Load appearance probabilities as single vector
    if hmodel.allocModel.K == hmodel.obsModel.K:
        w = hmodel.allocModel.get_active_comp_probs()
    else:
        w = np.ones(hmodel.obsModel.K)


    if dataset is not None and hasattr(dataset, 'X'):
        pylab.plot(
            dataset.X[:, 0], dataset.X[:, 1], '.', 
            color=(.3,.3,.3),
            alpha=0.5)

    nSkip = 0
    nGood = 0
    for ii, compID in enumerate(compListToPlot):
        if compID not in activeCompIDs:
            continue

        kk = np.flatnonzero(activeCompIDs == compID)
        assert kk.size == 1
        kk = kk[0]

        if w[kk] < proba_thr and compID not in compsToHighlight:
            nSkip += 1
            continue

        mu = hmodel.obsModel.get_mean_for_comp(kk)
        Sigma = hmodel.obsModel.get_covar_mat_for_comp(kk)

        if len(compsToHighlight) == 0 or compID in compsToHighlight:
            color = Colors[ii % len(Colors)]
            plotGauss2DContour(mu, Sigma, color=color)
        elif kk not in compsToHighlight:
            plotGauss2DContour(mu, Sigma, color='k')

        nGood += 1
        if nGood >= MaxKToDisplay:
            print 'DISPLAY LIMIT EXCEEDED. Showing %d/%d components' \
                % (nGood, len(activeCompIDs))
            break
    if nSkip > 0:
        print 'SKIPPED %d comps with size below %.2f' % (nSkip, proba_thr)

    # pylab.gca().set_aspect('equal', 'datalim')
    # pylab.axis('image')


def plotGauss2DContour(
        mu, Sigma,
        color='b',
        radiusLengths=[1.0, 3.0],
        markersize=3.0,
        ax_handle=None,
        ):
    ''' Plot elliptical contours for provided mean mu, covariance Sigma.

    Uses only the first 2 dimensions.

    Post Condition
    --------------
    Plot created on current axes
    '''
    if ax_handle is not None:
        pylab.sca(ax_handle)

    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    mu = mu[:2]
    Sigma = Sigma[:2, :2]
    D, V = np.linalg.eig(Sigma)
    sqrtSigma = np.dot(V, np.sqrt(np.diag(D)))

    # Prep for plotting elliptical contours
    # by creating grid of (x,y) points along perfect circle
    ts = np.arange(-np.pi, np.pi, 0.03)
    x = np.sin(ts)
    y = np.cos(ts)
    Zcirc = np.vstack([x, y])

    # Warp circle into ellipse defined by Sigma's eigenvectors
    Zellipse = np.dot(sqrtSigma, Zcirc)

    # plot contour lines across several radius lengths
    # TODO: instead, choose radius by percentage of prob mass contained within
    for r in radiusLengths:
        Z = r * Zellipse + mu[:, np.newaxis]
        pylab.plot(
            Z[0], Z[1], '.',
            markersize=markersize,
            markerfacecolor=color,
            markeredgecolor=color)


def plotCovMatFromHModel(hmodel,
                         compListToPlot=None,
                         compsToHighlight=None,
                         proba_thr=0.001,
                         ax_handle=None,
                         **kwargs):
    ''' Plot square image of covariance matrix for each component.

    Parameters
    -------
    hmodel : bnpy HModel object
    compListToPlot : array-like of integer IDs of components within hmodel
    compsToHighlight : int or array-like
        integer IDs to highlight
        if None, all components get unique colors
        if not None, only highlighted components get colors.
    proba_thr : float
        Minimum weight assigned to component in order to be plotted.
        All components with weight below proba_thr are ignored.
    '''

    nRow = 2
    nCol = int(np.ceil(hmodel.obsModel.K / 2.0))
    if ax_handle is None:
        ax_handle = pylab.subplots(
            nrows=nRow, ncols=nCol, figsize=(nCol * 2, nRow * 2))
    else:
        pylab.subplots(nrows=nRow, ncols=nCol, num=ax_handle.number)

    if compsToHighlight is not None:
        compsToHighlight = np.asarray(compsToHighlight)
        if compsToHighlight.ndim == 0:
            compsToHighlight = np.asarray([compsToHighlight])
    else:
        compsToHighlight = list()
    if compListToPlot is None:
        compListToPlot = np.arange(0, hmodel.obsModel.K)

    if hmodel.allocModel.K == hmodel.obsModel.K:
        w = hmodel.allocModel.get_active_comp_probs()
    else:
        w = np.ones(hmodel.obsModel.K)

    colorID = 0
    for plotID, kk in enumerate(compListToPlot):
        if w[kk] < proba_thr and kk not in compsToHighlight:
            Sigma = getEmptyCompSigmaImage(hmodel.obsModel.D)
            clim = [0, 1]
        else:
            Sigma = hmodel.obsModel.get_covar_mat_for_comp(kk)
            clim = [-.25, 1]
        pylab.subplot(nRow, nCol, plotID + 1)
        pylab.imshow(Sigma, interpolation='nearest', cmap='hot', clim=clim)
        pylab.xticks([])
        pylab.yticks([])
        pylab.xlabel('%.2f' % (w[kk]))
        if kk in compsToHighlight:
            pylab.xlabel('***')

    for emptyID in xrange(plotID + 1, nRow * nCol):
        aH = pylab.subplot(nRow, nCol, emptyID + 1)
        aH.axis('off')


def getEmptyCompSigmaImage(D):
    EmptySig = np.eye(D)
    for dd in range(D):
        EmptySig[dd, D - 1 - dd] = 1.0
    return EmptySig
