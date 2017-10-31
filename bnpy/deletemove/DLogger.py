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

def configure(taskoutpath, doSaveToDisk=0, doWriteStdOut=0):
    global Log
    Log = logging.getLogger('deletemove')

    Log.setLevel(logging.DEBUG)
    Log.handlers = []  # remove pre-existing handlers!
    formatter = logging.Formatter('%(message)s')
    # Config logger to save transcript of log messages to plain-text file
    if doSaveToDisk:
        # birth-vtranscript.txt logs everything
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "delete-transcript-verbose.txt"))
        fh.setLevel(0)
        fh.setFormatter(formatter)
        Log.addHandler(fh)

        # birth-transcript.txt logs high-level messages
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "delete-transcript-summary.txt"))
        fh.setLevel(logging.DEBUG + 1)
        fh.setFormatter(formatter)
        Log.addHandler(fh)

    # Config logger that can write to stdout
    #if doWriteStdOut:
    if True:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        Log.addHandler(ch)
    # Config null logger, avoids error messages about no handler existing
    if not doSaveToDisk and not doWriteStdOut:
        Log.addHandler(logging.NullHandler())
