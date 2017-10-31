from builtins import *
import logging
import os
import sys

# Configure Logger
Log = None

def pprint(msg, level=logging.DEBUG):
    global Log
    if Log is None:
        return
    if isinstance(level, str):
        if level.count('info'):
            level = logging.INFO
        elif level.count('debug'):
            level = logging.DEBUG
    Log.log(level, msg)

def configure(taskoutpath=None, doSaveToDisk=0, doWriteStdOut=0,
              verboseLevel=logging.DEBUG, summaryLevel=logging.INFO, **kwargs):
    global Log
    Log = logging.getLogger('heldout')

    Log.setLevel(logging.DEBUG)
    Log.handlers = []  # remove pre-existing handlers!
    formatter = logging.Formatter('%(message)s')
    # Config logger to save transcript of log messages to plain-text file
    if doSaveToDisk:
        # heldout-transcript-verbose.txt logs all messages
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "heldout-transcript-verbose.txt"))
        fh.setLevel(verboseLevel)
        fh.setFormatter(formatter)
        Log.addHandler(fh)

        # heldout-transcript-summary.txt logs one summary message per chkpt
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "heldout-transcript-summary.txt"))
        fh.setLevel(summaryLevel)
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
