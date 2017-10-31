'''
'''
from builtins import *
import numpy as np
import argparse
import os
import glob
import scipy.io

from .PlotUtil import pylab
from bnpy.ioutil import BNPYArgParser
from .JobFilter import filterJobs

import matplotlib
matplotlib.rcParams['text.usetex'] = False

Colors = [(0, 0, 0),  # black
          (0, 0, 1),  # blue
          (1, 0, 0),  # red
          (0, 1, 0.25),  # green (darker)
          (1, 0, 1),  # magenta
          (0, 1, 1),  # cyan
          (1, 0.6, 0),  # orange
          ]

XLabelMap = dict(laps='num pass thru train data',
                 K='num topics K',
                 times='training time (sec)'
                 )
YLabelMap = dict(
    avgLikScore='heldout log lik',
    avgAUCScore='heldout AUC',
    avgRPrecScore='heldout R precision',
    Kactive='num topics / doc',
    )


def plotJobsThatMatchKeywords(jpathPattern='/tmp/', **kwargs):
    ''' Make line plots for jobs matching pattern and provided kwargs.

        Example
        ---------
        plotJobsThatMatchKeywords('MyData', '
    '''
    if not jpathPattern.startswith(os.path.sep):
        jpathPattern = os.path.join(os.environ['BNPYOUTDIR'], jpathPattern)
    jpaths, legNames = filterJobs(jpathPattern, **kwargs)
    plotJobs(jpaths, legNames, **kwargs)


def plotJobs(jpaths, legNames, styles=None, fileSuffix='PredLik.mat',
             xvar='laps', yvar='avgLikScore', loc='upper right',
             minLap=0, showFinalPt=0,
             prefix='predlik',
             taskids=None, savefilename=None, tickfontsize=None,
             xjitter=None, bbox_to_anchor=None, **kwargs):
    ''' Create line plots for provided jobs
    '''
    nLines = len(jpaths)
    nLeg = len(legNames)
    assert nLines <= nLeg

    jitterByJob = np.linspace(-.5, .5, len(jpaths))

    for lineID in range(nLines):
        if styles is None:
            curStyle = dict(colorID=lineID)
        else:
            curStyle = styles[lineID]

        if xjitter is not None:
            xjitter = jitterByJob[lineID]
        plot_all_tasks_for_job(jpaths[lineID], legNames[lineID], minLap=minLap,
                               xvar=xvar, yvar=yvar, fileSuffix=fileSuffix,
                               showFinalPt=showFinalPt,
                               prefix=prefix,
                               taskids=taskids, xjitter=xjitter, **curStyle)

    if loc is not None and len(jpaths) > 1:
        pylab.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)

    if tickfontsize is not None:
        pylab.tick_params(axis='both', which='major', labelsize=tickfontsize)

    if savefilename is not None:
        try:
            pylab.show(block=False)
        except TypeError:
            pass  # when using IPython notebook
        pylab.savefig(savefilename, bbox_inches='tight', pad_inches=0)
    else:
        try:
            pylab.show(block=True)
        except TypeError:
            pass  # when using IPython notebook


def plot_all_tasks_for_job(jobpath, label, taskids=None,
                           lineType='.-',
                           spreadLineType='--',
                           color=None,
                           yvar='avgLikScore',
                           xvar='laps',
                           markersize=10,
                           linewidth=2,
                           minLap=0,
                           showFinalPt=0,
                           fileSuffix='PredLik.mat',
                           xjitter=None,
                           prefix='predlik',
                           colorID=0,
                           **kwargs):
    ''' Create line plot in current figure for each task/run of jobpath
    '''
    if not os.path.exists(jobpath):
        print('PATH NOT FOUND', jobpath)
        return None
    if not yvar.startswith('avg') and yvar.count('Kactive') == 0:
        yvar = 'avg' + yvar
    if not yvar.endswith('Score') and yvar.count('Kactive') == 0:
        yvar = yvar + 'Score'

    if color is None:
        color = Colors[colorID % len(Colors)]
    taskids = BNPYArgParser.parse_task_ids(jobpath, taskids)

    for tt, taskid in enumerate(taskids):
        taskoutpath = os.path.join(jobpath, taskid)
        hpaths = glob.glob(os.path.join(taskoutpath, '*' + fileSuffix))
        txtpaths = glob.glob(os.path.join(taskoutpath, 'predlik-*.txt'))
        ys_hi = None
        ys_lo = None
        if len(txtpaths) > 0:
            if fileSuffix.endswith('.txt'):
                suffix = '-' + fileSuffix
            else:
                suffix = '.txt'
            if xvar.count('lap'):
                xs = np.loadtxt(
                    os.path.join(taskoutpath, prefix + '-lapTrain.txt'))
            elif xvar.count('K'):
                xs = np.loadtxt(os.path.join(taskoutpath, prefix + '-K.txt'))
            elif xvar.count('time'):
                xs = np.loadtxt(os.path.join(
                    taskoutpath, prefix + '-timeTrain.txt'))
            else:
                raise ValueError("Unrecognized xvar: " + xvar)
            if yvar.count('Kactive') and not yvar.count('Percentile'):
                ys = np.loadtxt(os.path.join(taskoutpath,
                        prefix + '-' + yvar + 'Percentile50.txt'))
                ys_lo = np.loadtxt(os.path.join(taskoutpath,
                    prefix + '-' + yvar + 'Percentile10.txt'))
                ys_hi = np.loadtxt(os.path.join(taskoutpath,
                    prefix + '-' + yvar + 'Percentile90.txt'))
            else:
                ys = np.loadtxt(
                    os.path.join(taskoutpath, prefix + '-' + yvar + suffix))

            if minLap > 0 and taskoutpath.count('fix'):
                mask = laps > minLap
                xs = xs[mask]
                ys = ys[mask]
        elif len(hpaths) > 0:
            hpaths.sort()
            basenames = [x.split(os.path.sep)[-1] for x in hpaths]
            xs = np.asarray([float(x[3:11]) for x in basenames])
            ys = np.zeros_like(xs)
            for ii, hpath in enumerate(hpaths):
                MatVars = scipy.io.loadmat(hpath)
                ys[ii] = float(MatVars['avgPredLL'])
        else:
            raise ValueError(
                'Pred Lik data unavailable for job\n' + taskoutpath)

        plotargs = dict(markersize=markersize, linewidth=linewidth, label=None,
                        color=color, markeredgecolor=color,
                        )
        plotargs.update(kwargs)

        if tt == 0:
            plotargs['label'] = label
        if xjitter is not None:
            xs = xs + xjitter
        pylab.plot(xs, ys, lineType, **plotargs)
        if ys_lo is not None:
            del plotargs['label']
            pylab.plot(xs, ys_lo, spreadLineType, **plotargs)
            pylab.plot(xs, ys_hi, spreadLineType, **plotargs)

        if showFinalPt:
            pylab.plot(xs[-1], ys[-1], '.', **plotargs)
    pylab.xlabel(XLabelMap[xvar])
    pylab.ylabel(YLabelMap[yvar])


def parse_args():
    ''' Returns Namespace of parsed arguments retrieved from command line
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('dataName', type=str, default='AsteriskK8')
    parser.add_argument('jpath', type=str, default='demo*')

    helpMsg = "ids of trials/runs to plot from given job." + \
              " Example: '4' or '1,2,3' or '2-6'."
    parser.add_argument(
        '--taskids',
        type=str, default=None,
        help=helpMsg)
    parser.add_argument(
        '--savefilename', type=str, default=None,
        help="location where to save figure (absolute path directory)")
    parser.add_argument('--fileSuffix', type=str, default='PredLik.mat')
    args, unkList = parser.parse_known_args()

    argDict = BNPYArgParser.arglist_to_kwargs(unkList)
    argDict.update(args.__dict__)
    argDict['jpathPattern'] = os.path.join(os.environ['BNPYOUTDIR'],
                                           args.dataName,
                                           args.jpath)
    del argDict['dataName']
    del argDict['jpath']
    return argDict

if __name__ == "__main__":
    argDict = parse_args()
    plotJobsThatMatchKeywords(**argDict)
