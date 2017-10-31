import logging
import os
import sys
import numpy as np

Log = None

def pprint1Darr(
        arr, fmt='%.1f', nIndentSpaces=0, linewidth=80, label=None,
        returnStr=0, returnMaxLen=0):
    ''' Pretty print a 1D array

    Examples
    --------
    >>> pprint1Darr([1, 2, 3], fmt='%.1f')
     1.0 2.0 3.0
    >>> pprint1Darr(range(25), fmt='%.1f', linewidth=40)
      0.0  1.0  2.0  3.0  4.0  5.0  6.0
      7.0  8.0  9.0 10.0 11.0 12.0 13.0
     14.0 15.0 16.0 17.0 18.0 19.0 20.0
     21.0 22.0 23.0 24.0 
    >>> pprint1Darr(range(25), fmt='%.0f', linewidth=40, nIndentSpaces=3)
         0  1  2  3  4  5  6  7  8  9 10 11 12
        13 14 15 16 17 18 19 20 21 22 23 24 
    >>> pprint1Darr(range(20), fmt='%.0f', linewidth=40, label='my_array')
    my_array  0  1  2  3  4  5  6  7  8  9 10 11 12
             13 14 15 16 17 18 19 
    '''
    arr = np.asarray(arr)
    assert arr.ndim == 1
    strList = [fmt % (val) for val in arr]
    maxStrLen = np.max([len(s) for s in strList])
    maxlenfmt = '%' + str(maxStrLen) + "s"
    strList = [maxlenfmt % (s) for s in strList]
    longstr = "&" + '&'.join(strList) + "&" # add tail delimiter
    # Break lines at last possible spot before linewidth exceeds limit
    linesize = longstr[:linewidth].rfind('&')
    longstr = longstr.replace('&', ' ')
    lines = list()

    if isinstance(label, str):
        nIndentSpaces = max(nIndentSpaces, len(label))
    lineid = 0
    while len(longstr) > 0:
        indentstr = ' ' * nIndentSpaces
        if isinstance(label, str) and lineid == 0:
            indentstr = label + indentstr[len(label):]

        contentstr = longstr[:linesize]
        if len(contentstr.strip()) < 1: # skip all whitespace and blank lines
            break
        lines.append(indentstr + contentstr)
        longstr = longstr[linesize:]
        lineid += 1

    prettystr = '\n'.join(lines) + "\n"

    if returnStr:
        if returnMaxLen:
            return prettystr, maxStrLen
        return prettystr

    print(prettystr)    

def printDataSummary(
        Data_t, curSS_t, curSS_nott, relevantCompIDs, Plan, **kwargs):
    ''' Print summary of target dataset
    '''
    sizestr_t, maxLen1 = pprint1Darr(
        curSS_t.getCountVec()[relevantCompIDs],
        fmt='%5.1f',
        nIndentSpaces=2,
        returnStr=1, returnMaxLen=1)
    sizestr_nott, maxLen2 = pprint1Darr(
        curSS_nott.getCountVec()[relevantCompIDs],
        fmt='%5.1f',
        nIndentSpaces=2,
        returnStr=1, returnMaxLen=1)
    maxLen = np.maximum(maxLen1, maxLen2)
    sizefmt = '%' + str(maxLen) + '.1f'
    if maxLen1 < maxLen:
        sizestr_t = pprint1Darr(
            curSS_t.getCountVec()[relevantCompIDs],
            fmt=sizefmt,
            nIndentSpaces=2,
            returnStr=1)
    elif maxLen2 < maxLen:
        sizestr_nott = pprint1Darr(
            curSS_nott.getCountVec()[relevantCompIDs],
            fmt=sizefmt,
            nIndentSpaces=2,
            returnStr=1)
    Plan['sizefmt'] = sizefmt

    msg = "Data Stats\n"
    msg += "------------\n"
    msg += "TARGET total size: %.0f\n" % (
        curSS_t.getCountVec().sum())
    msg += "REST total size: %.0f\n" % (
        curSS_nott.getCountVec().sum())

    msg += "TARGET relevant comps: \n%s" % (
        pprint1Darr(relevantCompIDs, 
                    fmt='%' + str(maxLen) + 'd',
                    nIndentSpaces=2, returnStr=1))

    msg += "TARGET size of relevant comps: \n%s" % (sizestr_t)

    msg += "REST size of relevant comps: \n%s" % (sizestr_nott)

    print(msg)


def printRefineStatus(
        propSS=None, origK=0, riter=0,
        sizefmt='%5.1f', relevantCompIDs=None, **Plan):
    iterstr = ' %3d' % (riter)
    newNstr = pprint1Darr(
        propSS.getCountVec()[origK:], fmt=sizefmt, returnStr=1)
    if relevantCompIDs is None:
        targetNstr = ''
    else:
        targetNstr = pprint1Darr(
            propSS.getCountVec()[relevantCompIDs], fmt=sizefmt,
            returnStr=1)
    if riter == 0:
        headerfmt = "%3s | %" + str(len(newNstr)-1) + "s | %s" 
        header = headerfmt % ('iter', 'new comps', 'rel comps')
        print(header)
    print(iterstr + " | " + newNstr.rstrip() + " | " + targetNstr.rstrip())

def printEarlyExitMsg(msg, e, **kwargs):
    print(msg)
    print(str(e))

def log(msg, level='debug'):
    if Log is None:
        return
    if level == 'info':
        Log.info(msg)
    elif level == 'moreinfo':
        Log.log(15, msg)
    elif level == 'debug':
        Log.debug(msg)
    else:
        Log.log(level, msg)


def logStartPrep(lapFrac):
    msg = '=' * (50)
    msg = msg + ' lap %.2f Target Selection' % (lapFrac)
    log(msg, 'moreinfo')


def logStartMove(lapFrac, moveID, nMoves):
    msg = '=' * (50)
    msg = msg + ' lap %.2f %d/%d' % (lapFrac, moveID, nMoves)
    log(msg, 'moreinfo')


def logPhase(title):
    title = '.' * (50 - len(title)) + ' %s' % (title)
    log(title, 'debug')


def logPosVector(vec, fmt='%8.1f', Nmax=10, label='', level='debug'):
    if Log is None:
        return
    vstr = ' '.join([fmt % (x) for x in vec[:Nmax]])
    if len(label) > 0:
        log(vstr + " | " + label, level)
    else:
        log(vstr, level)


def logProbVector(vec, fmt='%8.4f', Nmax=10, level='debug'):
    if Log is None:
        return
    vstr = ' '.join([fmt % (x) for x in vec[:Nmax]])
    log(vstr, level)

def configure(taskoutpath, doSaveToDisk=0, doWriteStdOut=0):
    ''' Configure logging for 
    '''
    global Log
    Log = logging.getLogger('birthmove')

    Log.setLevel(logging.DEBUG)
    Log.handlers = []  # remove pre-existing handlers!
    formatter = logging.Formatter('%(message)s')
    # Config logger to save transcript of log messages to plain-text file
    if doSaveToDisk:
        # birth-vtranscript.txt logs everything
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "birth-vtranscript.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        Log.addHandler(fh)

        # birth-transcript.txt logs high-level messages
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "birth-transcript.txt"))
        fh.setLevel(logging.DEBUG + 1)
        fh.setFormatter(formatter)
        Log.addHandler(fh)

    # Config logger that can write to stdout
    if doWriteStdOut:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        Log.addHandler(ch)
    # Config null logger, avoids error messages about no handler existing
    if not doSaveToDisk and not doWriteStdOut:
        Log.addHandler(logging.NullHandler())
