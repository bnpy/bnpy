import logging
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
import time

# Configure Logger
LogDict = dict()
CumulativeTimesDict = defaultdict(lambda: defaultdict(float))
StartTimesDict = defaultdict(lambda: defaultdict(float))
CurrentLapTimesDict = defaultdict(lambda: defaultdict(float))

EVENTNAMES = [
    'all', 'local', 'global', 'io', 'callback']

def startEvent(eventName, subeventName='', level=logging.DEBUG):
    '''

    Post condition
    --------------
    StartTimesDict will store start time of specific event.
    '''
    StartTimesDict[eventName][subeventName] = time.time()

def stopEvent(eventName, subeventName='', level=logging.DEBUG):
    '''
    Post condition
    --------------
    '''
    elapsedTime = time.time() - StartTimesDict[eventName][subeventName]
    CumulativeTimesDict[eventName][subeventName] += elapsedTime
    CurrentLapTimesDict[eventName][subeventName] += elapsedTime
    CumulativeTimesDict['all'][eventName] += elapsedTime
    CurrentLapTimesDict['all'][eventName] += elapsedTime


def writeToLogOnLapCompleted(lapFrac, level=logging.DEBUG):
    lapstr = '%04.3f' % (lapFrac)
    for eventName in sorted(CumulativeTimesDict.keys()):
        curlapstr = ''
        totalstr = ''
        agg_curlap = 0.0
        agg_total = 0.0

        df_dict = OrderedDict()
        df_dict['lap'] = lapFrac
        for subeventName in sorted(CumulativeTimesDict[eventName]):
            df_dict['curlap_%s' % subeventName] = (
                CurrentLapTimesDict[eventName][subeventName])
            df_dict['alllaps_%s' % subeventName] = (
                CumulativeTimesDict[eventName][subeventName])

            agg_curlap += CurrentLapTimesDict[eventName][subeventName]
            agg_total += CumulativeTimesDict[eventName][subeventName]
            # Reset counter
            CurrentLapTimesDict[eventName][subeventName] = 0.0

        df_dict['curlap_total'] = agg_curlap
        df_dict['alllaps_total'] = agg_total
        df = pd.DataFrame([df_dict])
        opts = dict(index=False, float_format='%.3f')
        if np.allclose(lapFrac, 1.0):
            msg = df.to_csv(header=True, **opts)
        else:
            msg = df.to_csv(header=False, **opts)
        LogDict[eventName].log(level, msg)

def configure(taskoutpath, moveNames, doSaveToDisk=0, doWriteStdOut=0):
    global LogDict
    global EVENTNAMES
    EVENTNAMES += moveNames
    # Config logger to save transcript of log messages to plain-text file
    for eventName in EVENTNAMES:
        Log = logging.getLogger('elapsedtime.' + eventName)
        Log.setLevel(logging.DEBUG)
        Log.handlers = []  # remove pre-existing handlers!
        formatter = logging.Formatter('%(message)s')

        if doSaveToDisk:
            assert os.path.exists(taskoutpath)
            fh = logging.FileHandler(
                os.path.join(
                    taskoutpath,
                    "log-elapsedtime-%s.csv" % (eventName)))
            fh.setLevel(0)
            fh.setFormatter(formatter)
            Log.addHandler(fh)

        else:
            Log.addHandler(logging.NullHandler())

        LogDict[eventName] = Log
