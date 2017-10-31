'''
PrintTopics.py

Prints the top topics

Usage
-------
python PrintTopics.py dataName allocModelName obsModelName algName [options]

Saves topics as top_words.txt within the directory that the script draws from.

Options
--------
--topW : int
    Desired number of top words to show.
    (Must be less than the size of vocabulary).
--taskids : int or array_like
    ids of the tasks (individual runs) of the given job to plot.
    Ex: "1" or "3" or "1,2,3" or "1-6"
'''
from builtins import *
from .PlotUtil import pylab
import numpy as np
import argparse
import os
import sys

import bnpy
from bnpy.ioutil.ModelReader import \
    loadWordCountMatrixForLap, load_model_at_lap

STYLE = """
<style>
pre.num {line-height:13px; font-size:10px; display:inline; color:gray;}
pre.word {line-height:13px; font-size:13px; display:inline; color:black;}
h2 {line-height:16px; font-size:16px; color:gray;
    text-align:left; padding:0px; margin:0px;}
td {padding-top:5px; padding-bottom:5px;}
table { page-break-inside:auto }
tr { page-break-inside:avoid; page-break-after:auto }
</style>
"""


def showTopWordsForTask(taskpath, vocabfile, lap=None, doHTML=1,
                        doCounts=1, sortTopics=False, **kwargs):
    ''' Print top words for each topic from results saved on disk.

    Returns
    -------
    html : string, ready-to-print to display of the top words
    '''
    with open(vocabfile, 'r') as f:
        vocabList = [x.strip() for x in f.readlines()]

    if doCounts and (lap is None or lap > 0):
        WordCounts = loadWordCountMatrixForLap(taskpath, lap)
        countVec = WordCounts.sum(axis=1)
        if sortTopics:
            sortIDs = np.argsort(-1 * countVec)  # -1 to get descending order
            countVec = countVec[sortIDs]
            WordCounts = WordCounts[sortIDs]
        if doHTML:
            return htmlTopWordsFromWordCounts(
                WordCounts, vocabList, countVec=countVec, **kwargs)
        else:
            return printTopWordsFromWordCounts(WordCounts, vocabList)

    else:
        hmodel, lap = load_model_at_lap(taskpath, lap)
        if doHTML:
            return htmlTopWordsFromHModel(hmodel, vocabList, **kwargs)
        else:
            return printTopWordsFromHModel(hmodel, vocabList)


def htmlTopWordsFromWordCounts(
        WordCounts, vocabList, order=None, Ktop=10,
        ncols=5, maxKToDisplay=50, countVec=None,
        fmtstr='%8d',
        activeCompIDs=None, **kwargs):
    K, W = WordCounts.shape
    if order is None:
        order = np.arange(K)
    if activeCompIDs is None:
        activeCompIDs = np.arange(K)
    #if countVec is None:
    #    countVec = np.sum(WordCounts, axis=1)

    htmllines = list()
    htmllines.append(STYLE)
    htmllines.append('<table>')
    for posID, compID in enumerate(order[:maxKToDisplay]):
        if posID % ncols == 0:
            htmllines.append('  <tr>')

        k = np.flatnonzero(activeCompIDs == compID)
        if len(k) == 1:
            k = k[0]
            if countVec is None:
                titleline = '<h2>%4d/%d</h2>' % (
                    k + 1, K)
            else:
                titleline = '<h2>%4d/%d %10d</h2>' % (
                    k + 1, K, countVec[k])
            htmllines.append('    <td>' + titleline)
            htmllines.append('    ')

            htmlPattern = \
                '<pre class="num">' + fmtstr + ' ' + \
                '</pre><pre class="word">%s </pre>'
            topIDs = np.argsort(-1 * WordCounts[k])[:Ktop]
            for topID in topIDs:
                dataline = htmlPattern % (
                    WordCounts[k, topID],
                    vocabList[topID][:16])
                htmllines.append(dataline + "<br />")
            htmllines.append('    </td>')
        else:
            htmllines.append('    <td></td>')

        if posID % ncols == ncols - 1:
            htmllines.append(' </tr>')
    htmllines.append('</table>')
    return '\n'.join(htmllines)


def htmlTopWordsFromHModel(hmodel, vocabList, order=None, Ktop=10,
                           ncols=5, maxKToDisplay=50, activeCompIDs=None,
                           **kwargs):
    try:
        topics = hmodel.obsModel.EstParams.phi
    except AttributeError:
        hmodel.obsModel.setEstParamsFromPost(hmodel.obsModel.Post)
        topics = hmodel.obsModel.EstParams.phi
    K, W = topics.shape
    if order is None:
        order = np.arange(K)
    if activeCompIDs is None:
        activeCompIDs = np.arange(K)

    htmllines = list()
    htmllines.append(STYLE)
    htmllines.append('<table>')

    for posID, compID in enumerate(order[:maxKToDisplay]):
        if posID % ncols == 0:
            htmllines.append('  <tr>')

        k = np.flatnonzero(activeCompIDs == compID)
        if len(k) == 1:
            k = k[0]
            htmllines.append('    <td><pre>')
            topIDs = np.argsort(-1 * topics[k])[:Ktop]
            for topID in topIDs:
                dataline = ' %.3f %s ' % (
                    topics[k, topID], vocabList[topID][:16])
                htmllines.append(dataline)
            htmllines.append('    </pre></td>')

        else:
            htmllines.append('   <td></td>')

        if posID % ncols == ncols - 1:
            htmllines.append(' </tr>')
    htmllines.append('</table>')
    return '\n'.join(htmllines)


def printTopWordsFromHModel(hmodel, vocabList, **kwargs):
    try:
        topics = hmodel.obsModel.EstParams.phi
    except AttributeError:
        hmodel.obsModel.setEstParamsFromPost(hmodel.obsModel.Post)
        topics = hmodel.obsModel.EstParams.phi
    printTopWordsFromTopics(topics, vocabList, **kwargs)


def printTopWordsFromWordCounts(
        WordCounts, vocabList, order=None,
        prefix='topic', Ktop=10):
    K, W = WordCounts.shape
    if order is None:
        order = np.arange(K)
    N = np.sum(WordCounts, axis=1)
    for posID, k in enumerate(order):
        print('----- %s %d. count %5d.' % (prefix, k, N[k]))

        topIDs = np.argsort(-1 * WordCounts[k])
        for wID in topIDs[:Ktop]:
            if WordCounts[k, wID] > 0:
                print('%3d %s' % (WordCounts[k, wID], vocabList[wID]))


def printTopWordsFromTopics(
        topics, vocabList, order=None,
        prefix='topic', ktarget=None, Ktop=10):
    K, W = topics.shape
    if ktarget is not None:
        topIDs = np.argsort(-1 * topics[ktarget])
        for wID in topIDs[:Ktop]:
            print('%.3f %s' % (topics[ktarget, wID], vocabList[wID]))
        return
    # Base case: print all topics
    for k in range(K):
        print('----- %s %d' % (prefix, k))
        topIDs = np.argsort(-1 * topics[k])
        for wID in topIDs[:Ktop]:
            print('%.3f %s' % (topics[k, wID], vocabList[wID]))

def plotCompsFromHModel(hmodel, **kwargs):
    ''' Create subplots of top 10 words from each topic, from a trained model.
    '''
    if hmodel.getObsModelName().count('Mult'):
        WordCounts = hmodel.obsModel.Post.lam.copy()
        WordCounts -= hmodel.obsModel.Prior.lam[np.newaxis,:]
        plotCompsFromWordCounts(WordCounts, **kwargs)
    elif hmodel.getObsModelName().count('Bern'):
        lam1 = hmodel.obsModel.Post.lam1
        lam0 = hmodel.obsModel.Post.lam0
        probs = lam1 / (lam1 + lam0)
        plotCompsFromWordCounts(probs, **kwargs)


def plotCompsFromWordCounts(
        WordCounts=None,
        topics_KV=None,
        vocabList=None,
        compListToPlot=None,
        compsToHighlight=None,
        xlabels=None,
        wordSizeLimit=10,
        Ktop=10, Kmax=32,
        H=2.5, W=2.0,
        figH=None, ncols=10,
        ax_list=None,
        fontsize=10,
        **kwargs):
    ''' Create subplots of top 10 words from each topic, from word count array.

    Post Condition
    --------------
    Current matplotlib figure has subplot for each topic.
    '''
    if vocabList is None:
        raise ValueError('Missing vocabList. Cannot display topics.')
    if WordCounts is not None:
        WordCounts = np.asarray(WordCounts, dtype=np.float64)
        if WordCounts.ndim == 1:
            WordCounts = WordCounts[np.newaxis,:]
        K, vocab_size = WordCounts.shape
        N = np.sum(WordCounts, axis=1)
    else:
        topics_KV = np.asarray(topics_KV, dtype=np.float64)
        K, vocab_size = topics_KV.shape

    if compListToPlot is None:
        compListToPlot = np.arange(0, K)
    Kplot = np.minimum(len(compListToPlot), Kmax)
    if len(compListToPlot) > Kmax:
        print('DISPLAY LIMIT EXCEEDED. Showing %d/%d components' \
            % (Kplot, len(compListToPlot)))
    compListToPlot = compListToPlot[:Kplot]
    # Parse comps to highlight
    compsToHighlight = np.asarray(compsToHighlight)
    if compsToHighlight.ndim == 0:
        compsToHighlight = np.asarray([compsToHighlight])
    nrows = int(np.ceil(Kplot / float(ncols)))
    # Create Figure
    if ax_list is None:
        fig_h, ax_list = pylab.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(ncols * W, nrows * H))
    if isinstance(ax_list, np.ndarray):
        ax_list = ax_list.flatten().tolist()
    assert isinstance(ax_list, list)
    n_images_viewable = len(ax_list)
    n_images_to_plot = len(compListToPlot)

    for plotID, compID in enumerate(compListToPlot):
        cur_ax_h = ax_list[plotID] #pylab.subplot(nrows, ncols, plotID + 1)

        topicMultilineStr = ''
        if WordCounts is None:
            topIDs = np.argsort(-1 * topics_KV[compID])
        else:
            topIDs = np.argsort(-1 * WordCounts[compID])
        for wID in topIDs[:Ktop]:
            if WordCounts is not None and WordCounts[compID, wID] > 0:
                wctStr = count2str(WordCounts[compID, wID])
                topicMultilineStr += '%s %s\n' % (
                    wctStr, vocabList[wID][:wordSizeLimit])
            else:
                topicMultilineStr += "%.4f %s\n" % (
                    topics_KV[compID, wID],
                    vocabList[wID][:wordSizeLimit])
        cur_ax_h.text(
            0, 0, topicMultilineStr, fontsize=fontsize, family='monospace')
        cur_ax_h.set_xlim([0, 1]);
        cur_ax_h.set_ylim([0, 1]);
        cur_ax_h.set_xticks([])
        cur_ax_h.set_yticks([])

        # Draw colored border around highlighted topics
        if compID in compsToHighlight:
            [i.set_color('green') for i in ax.spines.values()]
            [i.set_linewidth(3) for i in ax.spines.values()]
        if xlabels is not None:
            if len(xlabels) > 0:
                cur_ax_h.set_xlabel(xlabels[plotID], fontsize=11)

    # Disable empty plots
    for k, ax_h in enumerate(ax_list[n_images_to_plot:]):
        ax_h.axis('off')

    return figH, ax_list

def count2str(val, width=4, minVal=0.01, **kwargs):
    ''' Pretty print large positive count values in confined 4 char format.

    Examples
    --------
    >>> count2str(.02, width=4)
    '0.02'
    >>> count2str(.0001, width=4)
    '<.01'
    >>> count2str(.02, width=5, minVal=0.001)
    '0.020'
    >>> count2str(.99, width=4)
    '0.99'
    >>> count2str(1, width=4)
    '   1'
    >>> count2str(1.9, width=4)
    ' 1.9'
    >>> count2str(1.85, width=5)
    ' 1.85'
    >>> count2str(10, width=4)
    '  10'
    >>> count2str(9997.875, width=4) # bug??
    '9997'
    >>> count2str(9999, width=4)
    '9999'
    >>> count2str(10003, width=4)
    ' 10k'
    >>> count2str(123000, width=4)
    '123k'
    >>> count2str(7654321, width=4)
    '  7M'
    >>> count2str(987654321, width=4)
    '987M'
    >>> count2str(1111111111, width=4)
    ' >1B'
    '''
    val = np.asarray(val)
    assert width >= 4
    if val.dtype == np.float:
        nDecPlace = int(np.abs(np.log10(minVal)))
        if val < minVal:
            fmt = '%' + str(width) + 's'
            return fmt % ('<' + str(minVal)[1:])
        elif val < 1:
            fmt = '%' + str(width) + '.' + str(nDecPlace) + 'f'
            return fmt % (val)
        elif val < 10**(width):
            valstr = str(val)[:width]
            fmtstr = '%' + str(width) + 's'
            return fmtstr % (valstr)
    val = np.round(val)
    if val < 10**(width):
        fmt = '%' + str(width) + 'd'
        return fmt % (val)
    elif val < 10**6:
        nThou = val // 1000
        fmt = '%' + str(width-1) + 'd'
        return fmt % (nThou) + 'k'
    elif val < 10**9:
        nMil = val // 1000000
        fmt = '%' + str(width-1) + 'd'
        return fmt % (nMil) + 'M'
    else:
        fmt = '%' + str(width) + 's'
        return fmt % ('>1B')

def num2fwstr(num, width=4):
    ''' Convert scalar number to fixed-width string

    TODO: Handle cases with num >> 10**(width-1)

    Returns
    -------
    s : string with exact width
    '''
    minVal = 10.0**(-(width-2))
    fwfmt = '%' + str(width) + 's'
    if num < minVal:
        s = '<' + str(minVal)[1:]
    elif num < 1.0:
        s = str(num)
    else:
        P = int(np.ceil(np.log10(num)))
        tenP = (10.0**P)
        rnum = round(num / tenP, width-1)
        s = str(rnum * tenP)
    fws = fwfmt % (s)
    return fws[:width]

def countvec2list(countvec):
    return [count2str(x) for x in countvec]

def vec2str(vec, **kwargs):
    if len(vec) == 0:
        return 'empty'
    return ' '.join([count2str(x, **kwargs) for x in vec])

def uidsAndCounts2strlist(SS):
    countvec = SS.getCountVec()
    if SS.hasSelectionTerm('DocUsageCount'):
        usagevec = SS._SelectTerms.DocUsageCount
        return ['%5d : %s \n %s docs' % (
            SS.uids[k],
            count2str(countvec[k]),
            count2str(usagevec[k])) for k in range(SS.K)]
    else:
        return ['%5d : %s' % (
            SS.uids[k], count2str(countvec[k])) for k in range(SS.K)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('taskpath')
    parser.add_argument('vocabfilepath')
    parser.add_argument('--sortTopics', default=1)
    parser.add_argument('--maxKToDisplay', default=10000)
    parser.add_argument('--doCounts', type=int, default=1)
    parser.add_argument('--doHTML', type=int, default=1)
    args = parser.parse_args()
    htmlstr = showTopWordsForTask(args.taskpath, args.vocabfilepath,
                              sortTopics=args.sortTopics,
                              doCounts=args.doCounts,
                              doHTML=args.doHTML,
                              maxKToDisplay=args.maxKToDisplay)
    if htmlstr is not None:
        print(htmlstr)
