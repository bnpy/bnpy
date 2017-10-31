from builtins import *
import logging
import os
import sys
from collections import defaultdict

# Configure Logger
Log = None
Cache = defaultdict(lambda: list())
CacheOrder = list()


def log(msg, level='debug'):
    if Log is None:
        return
    if level == 'debug':
        Log.debug(msg)
    else:
        Log.info(msg)


def logStartMove(lapFrac, batchID):
    msg = '=' * (50)
    msg = msg + ' lap %.2f | batch %d' % (lapFrac, batchID)
    log(msg, 'debug')


def logPosVector(vec, fmt='%8.1f', Nmax=10):
    if Log is None:
        return
    vstr = ' '.join([fmt % (x) for x in vec[:Nmax]])
    log(vstr, 'debug')


def logProbVector(vec, fmt='%8.4f', Nmax=10):
    if Log is None:
        return
    vstr = ' '.join([fmt % (x) for x in vec[:Nmax]])
    log(vstr, 'debug')


def configure(taskoutpath=None, doSaveToDisk=0, doWriteStdOut=0):
    global Log
    Log = logging.getLogger('localstep')

    Log.setLevel(logging.DEBUG)
    Log.handlers = []  # remove pre-existing handlers!
    formatter = logging.Formatter('%(message)s')
    # Config logger to save transcript of log messages to plain-text file
    if doSaveToDisk:
        logfile = os.path.join(taskoutpath, "localstep-transcript.txt")
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        Log.addHandler(fh)
    # Config logger that can write to stdout
    if doWriteStdOut:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG + 1)
        ch.setFormatter(formatter)
        Log.addHandler(ch)
    # Config null logger, avoids error messages about no handler existing
    if not doSaveToDisk and not doWriteStdOut:
        Log.addHandler(logging.NullHandler())
