from builtins import *
import logging
import os
import sys
from collections import defaultdict
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
        for subeventName in sorted(CumulativeTimesDict[eventName]):
            curlapstr += ' cur_%s %.2f' % (
                subeventName, CurrentLapTimesDict[eventName][subeventName])
            totalstr += ' ttl_%s %.2f' % (
                subeventName, CumulativeTimesDict[eventName][subeventName])
            agg_curlap += CurrentLapTimesDict[eventName][subeventName]
            agg_total += CumulativeTimesDict[eventName][subeventName]
            CurrentLapTimesDict[eventName][subeventName] = 0.0
        agg_curlapstr = ' cur %.2f' % (agg_curlap)
        agg_totalstr = ' ttl %.2f' % (agg_total)
        msg = lapstr + agg_curlapstr + agg_totalstr + curlapstr + totalstr
        LogDict[eventName].log(level, msg)

def configure(taskoutpath, moveNames, doSaveToDisk=0, doWriteStdOut=0):
    global LogDict
    global EVENTNAMES
    EVENTNAMES += moveNames
    # Config logger to save transcript of log messages to plain-text file
    if doSaveToDisk:
        for eventName in EVENTNAMES:
            Log = logging.getLogger('elapsedtime.' + eventName)
            Log.setLevel(logging.DEBUG)
            Log.handlers = []  # remove pre-existing handlers!
            fh = logging.FileHandler(
                os.path.join(
                    taskoutpath,
                    "log-elapsedtime-%s.txt" % (eventName)))
            fh.setLevel(0)
            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)
            Log.addHandler(fh)
            LogDict[eventName] = Log
