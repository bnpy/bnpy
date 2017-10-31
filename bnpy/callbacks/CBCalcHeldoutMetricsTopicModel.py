'''
CBCalcHeldoutMetricsTopicModel.py

Learning alg callback extension for fitting topic models on heldout data.

When applied, will perform heldout inference at every parameter-save checkpoint.

Usage
--------
Add the following keyword arg to any call to bnpy.run
 --customFuncPath CBCalcHeldoutMetricsTopicModel.py

Example
-------
$ python -m bnpy.Run BarsK10V900 FiniteTopicModel Mult VB \
    --K 10 --nLap 50 \
    --saveEvery 10 \
    --customFuncPath CBCalcHeldoutMetricsTopicModel

Notes
--------
Uses the custom-function interface for learning algorithms.
This interface means that the functions onAlgorithmComplete and onBatchComplete
defined here will be called at appropriate time in *every* learning algorithm.
See LearnAlg.py's eval_custom_function for details.
'''

from builtins import *
import os
import numpy as np
import scipy.io

from . import InferHeldoutTopics
from . import HeldoutMetricsLogger
SavedLapSet = set()

def onAlgorithmComplete(**kwargs):
    ''' Runs at completion of the learning algorithm.

    Keyword Args
    --------T
    All workspace variables passed along from learning alg.
    '''
    if kwargs['lapFrac'] not in SavedLapSet:
        runHeldoutCallback(**kwargs)

def onBatchComplete(**kwargs):
    ''' Runs viterbi whenever a parameter-saving checkpoint is reached.

    Keyword Args
    --------
    All workspace variables passed along from learning alg.
    '''
    global SavedLapSet
    if kwargs['isInitial']:
        SavedLapSet = set()
        HeldoutMetricsLogger.configure(
            **kwargs['learnAlg'].BNPYRunKwArgs['OutputPrefs'])

    if not kwargs['learnAlg'].isSaveParamsCheckpoint(kwargs['lapFrac'],
                                                     kwargs['iterid']):
        return
    if kwargs['lapFrac'] in SavedLapSet:
        return
    SavedLapSet.add(kwargs['lapFrac'])
    runHeldoutCallback(**kwargs)


def runHeldoutCallback(**kwargs):
    ''' Run heldout metrics evaluation on test dataset.

    Kwargs will contain all workspace vars passed from the learning alg.

    Keyword Args
    ------------
    hmodel : current HModel object
    Data : current Data object
        representing *entire* dataset (not just one chunk)

    Returns
    -------
    None. MAP state sequences are saved to a MAT file.

    Output
    -------
    MATfile format: Lap0020.000MAPStateSeqs.mat
    '''
    taskpath = kwargs['learnAlg'].savedir
    for splitName in ['validation', 'test']:
        elapsedTime = kwargs['learnAlg'].get_elapsed_time()
        InferHeldoutTopics.evalTopicModelOnTestDataFromTaskpath(
            dataSplitName=splitName,
            taskpath=taskpath,
            elapsedTime=elapsedTime,
            queryLap=kwargs['lapFrac'],
            printFunc=HeldoutMetricsLogger.pprint,
            **kwargs)
