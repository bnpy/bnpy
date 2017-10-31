'''
BarsViz.py

Visualization tools for toy bars data for topic models.
'''
from builtins import *
import numpy as np

from .PlotUtil import pylab

imshowArgs = dict(interpolation='nearest',
                  cmap='bone_r', # zero is white, large values are black
                  aspect=1.0,
                  vmin=0.0,
                  vmax=1.0)

def show_square_images(
        topics_KV=None,
        xlabels=[],
        max_n_images=50,
        ncols=5,
        ax_list=None,
        im_width=1,
        im_height=1,
        fontsize=10,
        **kwargs):
    ''' Show provided vectors as square images

    Post Condition
    --------------
    Provided axes have plots updated.
    '''
    global imshowArgs
    local_imshowArgs = dict(**imshowArgs)
    for key in local_imshowArgs:
        if key in kwargs:
            local_imshowArgs[key] = kwargs[key]

    K, V = topics_KV.shape
    sqrtV = int(np.sqrt(V))
    assert np.allclose(sqrtV, np.sqrt(V))

    n_images_to_plot = np.minimum(K, max_n_images)
    ncols = np.minimum(ncols, n_images_to_plot)
    if ax_list is None:
        # Make a new figure
        nrows = int(np.ceil(n_images_to_plot / float(ncols)))
        fig_h, ax_list = pylab.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(ncols * im_width, nrows * im_height))
    if isinstance(ax_list, np.ndarray):
        ax_list = ax_list.flatten().tolist()
    assert isinstance(ax_list, list)
    n_images_viewable = len(ax_list)

    # Plot each row as square image
    for k, ax_h in enumerate(ax_list[:n_images_to_plot]):
        cur_im_sVsV = np.reshape(topics_KV[k, :], (sqrtV, sqrtV))
        ax_h.imshow(cur_im_sVsV, **local_imshowArgs)
        ax_h.set_xticks([])
        ax_h.set_yticks([])

        if xlabels is not None:
            if len(xlabels) > 0:
                ax_h.set_xlabel(xlabels[k], fontsize=fontsize)

    # Disable empty plots
    for k, ax_h in enumerate(ax_list[n_images_to_plot:]):
        ax_h.axis('off')

    # Fix margins between subplots
    #pylab.subplots_adjust(
    #    wspace=0.1,
    #    hspace=0.1 * nrows,
    #    left=0.001, right=0.999,
    #   bottom=0.1, top=0.999)
    return ax_list


def plotExampleBarsDocs(Data, docIDsToPlot=None, figID=None,
                        vmax=None, nDocToPlot=16, doShowNow=False,
                        seed=0, randstate=np.random.RandomState(0),
                        xlabels=None,
                        W=1, H=1,
                        **kwargs):
    kwargs['vmin'] = 0
    kwargs['interpolation'] = 'nearest'
    if vmax is not None:
        kwargs['vmax'] = vmax
    if seed is not None:
        randstate = np.random.RandomState(seed)
    V = Data.vocab_size
    sqrtV = int(np.sqrt(V))
    assert np.allclose(sqrtV * sqrtV, V)
    if docIDsToPlot is not None:
        nDocToPlot = len(docIDsToPlot)
    else:
        size = np.minimum(Data.nDoc, nDocToPlot)
        docIDsToPlot = randstate.choice(Data.nDoc, size=size, replace=False)
    ncols = 5
    nrows = int(np.ceil(nDocToPlot / float(ncols)))
    if vmax is None:
        DocWordArr = Data.getDocTypeCountMatrix()
        vmax = int(np.max(np.percentile(DocWordArr, 98, axis=0)))

    if figID is None:
        figH, ha = pylab.subplots(nrows=nrows, ncols=ncols,
                                  figsize=(ncols * W, nrows * H))

    for plotPos, docID in enumerate(docIDsToPlot):
        start = Data.doc_range[docID]
        stop = Data.doc_range[docID + 1]
        wIDs = Data.word_id[start:stop]
        wCts = Data.word_count[start:stop]
        docWordHist = np.zeros(V)
        docWordHist[wIDs] = wCts
        squareIm = np.reshape(docWordHist, (sqrtV, sqrtV))
        pylab.subplot(nrows, ncols, plotPos + 1)
        pylab.imshow(squareIm, **kwargs)
        pylab.axis('image')
        pylab.xticks([])
        pylab.yticks([])
        if xlabels is not None:
            pylab.xlabel(xlabels[plotPos])

    # Disable empty plots!
    for kdel in range(plotPos + 2, nrows * ncols + 1):
        aH = pylab.subplot(nrows, ncols, kdel)
        aH.axis('off')

    # Fix margins between subplots
    pylab.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99,
                          top=0.99, bottom=0.01)
    if doShowNow:
        pylab.show()


def plotBarsFromHModel(hmodel, Data=None, doShowNow=False, figH=None,
                       doSquare=1,
                       xlabels=[],
                       compsToHighlight=None, compListToPlot=None,
                       activeCompIDs=None, Kmax=50,
                       width=6, height=3, vmax=None,
                       block=0,  # unused
                       jobname='',  # unused
                       **kwargs):
    if vmax is not None:
        kwargs['vmax'] = vmax
    if hasattr(hmodel.obsModel, 'Post'):
        lam = hmodel.obsModel.Post.lam
        topics = lam / lam.sum(axis=1)[:, np.newaxis]
    else:
        topics = hmodel.obsModel.EstParams.phi.copy()

    # Determine intensity scale for topic-word image
    global imshowArgs
    if vmax is not None:
        imshowArgs['vmax'] = vmax
    else:
        imshowArgs['vmax'] = 1.5 * np.percentile(topics, 95)

    if doSquare:
        figH = showTopicsAsSquareImages(topics,
                                        activeCompIDs=activeCompIDs,
                                        compsToHighlight=compsToHighlight,
                                        compListToPlot=compListToPlot,
                                        Kmax=Kmax, figH=figH,
                                        xlabels=xlabels,
                                        **kwargs)
    else:
        if figH is None:
            figH = pylab.figure(figsize=(width, height))
        else:
            pylab.axes(figH)
        showAllTopicsInSingleImage(topics, compsToHighlight, **kwargs)
    if doShowNow:
        pylab.show()
    return figH


def plotBarsForTopicMATFile(matfilename, sortBy=None, keepWorst=0,
                            levels=None, worstThr=0.01,
                            Kmax=20, **kwargs):
    kwargs['vmin'] = 0
    kwargs['interpolation'] = 'nearest'
    if vmax is not None:
        kwargs['vmax'] = vmax
    import bnpy.ioutil
    if isinstance(matfilename, np.ndarray):
        topics = matfilename
        probs = np.ones(topics.shape[0])
    else:
        topics, probs, alph = bnpy.ioutil.ModelReader.loadTopicModel(
            matfilename, returnTPA=1)
    print('total K=', topics.shape[0])
    print('beta>0.0001 K=', np.sum(probs > .0001))
    if levels is not None:
        assert topics.max() > 1.0
        topics = np.floor(topics)
        for b in range(len(levels) - 1):
            mask = np.logical_and(topics > levels[b],
                                  topics <= levels[b + 1])
            topics[mask] = b
    else:
        topics /= topics.sum(axis=1)[:, np.newaxis]
    if sortBy == 'probs':
        sortIDs = np.argsort(-1 * probs)
        if keepWorst > 0:
            probs = probs[sortIDs]
            worstLoc = 0
            while (probs[worstLoc - 1] < worstThr):
                worstLoc -= 1
            L = len(sortIDs)
            sortIDs = sortIDs[:L + worstLoc]
            probs = probs[:L + worstLoc]
            print(probs[-1], '<<<< first above cutoff')
            keepIDs = np.hstack(
                [sortIDs[:(Kmax - keepWorst)], sortIDs[-keepWorst:]])
            print(probs[:(Kmax - keepWorst)])
            print(probs[-keepWorst:])
            print(len(sortIDs), '<<< count above cutoff')
            topics = topics[keepIDs]
        else:
            topics = topics[sortIDs[:Kmax]]
    showTopicsAsSquareImages(topics, Kmax=Kmax, **kwargs)


def showTopicsAsSquareImages(topics,
                             activeCompIDs=None,
                             compsToHighlight=None,
                             compListToPlot=None,
                             xlabels=[],
                             Kmax=50,
                             ncols=5,
                             W=1, H=1, figH=None,
                             **kwargs):
    global imshowArgs
    local_imshowArgs = dict(**imshowArgs)
    for key in local_imshowArgs:
        if key in kwargs:
            local_imshowArgs[key] = kwargs[key]

    if len(xlabels) > 0:
        H = 1.5 * H
    K, V = topics.shape
    sqrtV = int(np.sqrt(V))
    assert np.allclose(sqrtV, np.sqrt(V))

    if compListToPlot is None:
        compListToPlot = np.arange(0, K)
    if activeCompIDs is None:
        activeCompIDs = np.arange(0, K)
    compsToHighlight = np.asarray(compsToHighlight)
    if compsToHighlight.ndim == 0:
        compsToHighlight = np.asarray([compsToHighlight])

    # Create Figure
    Kplot = np.minimum(len(compListToPlot), Kmax)
    #ncols = 5  # int(np.ceil(Kplot / float(nrows)))
    nrows = int(np.ceil(Kplot / float(ncols)))
    if figH is None:
        # Make a new figure
        figH, ha = pylab.subplots(nrows=nrows, ncols=ncols,
                                  figsize=(ncols * W, nrows * H))
    else:
        # Use existing figure
        # TODO: Find a way to make this call actually change the figsize
        figH, ha = pylab.subplots(nrows=nrows, ncols=ncols,
                                  figsize=(ncols * W, nrows * H),
                                  num=figH.number)

    for plotID, compID in enumerate(compListToPlot):
        if plotID >= Kmax:
            print('DISPLAY LIMIT EXCEEDED. Showing %d/%d components' \
                % (plotID, len(activeCompIDs)))
            break

        if compID not in activeCompIDs:
            aH = pylab.subplot(nrows, ncols, plotID + 1)
            aH.axis('off')
            continue

        kk = np.flatnonzero(compID == activeCompIDs)[0]
        topicIm = np.reshape(topics[kk, :], (sqrtV, sqrtV))
        ax = pylab.subplot(nrows, ncols, plotID + 1)
        pylab.imshow(topicIm, **local_imshowArgs)
        pylab.xticks([])
        pylab.yticks([])

        # Draw colored border around highlighted topics
        if compID in compsToHighlight:
            [i.set_color('green') for i in ax.spines.values()]
            [i.set_linewidth(3) for i in ax.spines.values()]

        if xlabels is not None:
            if len(xlabels) > 0:
                pylab.xlabel(xlabels[plotID], fontsize=11)

    # Disable empty plots!
    for kdel in range(plotID + 2, nrows * ncols + 1):
        aH = pylab.subplot(nrows, ncols, kdel)
        aH.axis('off')

    # Fix margins between subplots
    pylab.subplots_adjust(
        wspace=0.1,
        hspace=0.1 * nrows,
        left=0.001, right=0.999,
        bottom=0.1, top=0.999)
    return figH


def showAllTopicsInSingleImage(topics, compsToHighlight, **imshowArgs):
    K, V = topics.shape
    aspectR = V / float(K)
    pylab.imshow(topics, aspect=aspectR, **imshowArgs)
    if compsToHighlight is not None:
        ks = np.asarray(compsToHighlight)
        if ks.ndim == 0:
            ks = np.asarray([ks])
        pylab.yticks(ks, ['**** %d' % (k) for k in ks])
