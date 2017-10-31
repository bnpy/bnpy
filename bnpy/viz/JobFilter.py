from builtins import *
import numpy as np
import glob
import os

from collections import defaultdict, OrderedDict
from bnpy.ioutil import BNPYArgParser

# kwargs that arent needed for any job pattern matching
SkipKeys = ['taskids', 'savefilename', 'fileSuffix',
            'loc', 'xvar', 'yvar', 'bbox_to_anchor']


def findKeysWithDiffVals(dA, dB):
    ''' Find keys in both dA, dB that have different values.
    '''
    keepKeys = list()
    for key in dA:
        if key in dB:
            if dA[key] != dB[key]:
                keepKeys.append(key)
    return keepKeys


def mapToKey(val):
    if val.count('DP') or val.count('HDP'):
        return 'allocModelName'
    elif _isInitname(val):
        return 'initname'
    return None


def _isInitname(val):
    return val.count('rand') or val.count('true') or \
        val.count('contig') or val.count('sacb') or \
        val.count('spectral') or val.count('kmeans') or \
        val.count('plusplus')


def jpath2jdict(jpath):
    ''' Convert provided path string to dictionary of options.

    Example
    ---------
    >>> jpath2jdict('abc-nBatch=10-initname=randexamples')
    OrderedDict(\
[('field1', 'abc'), ('nBatch', '10'), ('initname', 'randexamples')])
    '''
    basename = jpath.split(os.path.sep)[-1]
    fields = basename.split('-')
    D = OrderedDict()
    for fID, field in enumerate(fields):
        if field.count('=') == 1:
            key, val = field.split('=')
            D[key] = str(val)
        else:
            val = field
            key = mapToKey(val)
            if key is not None:
                D[key] = val
            else:
                D['field' + str(fID + 1)] = val
    return D


def makeListOfJPatternsWithSpecificVals(PPListMap,
                                        key='',
                                        vals=None,
                                        **keyValPairs):
    ''' Make iterable list of jpaths with specific values.

    Example
    -------
    >>> jpaths = []
    >>> jpaths.append('demo-K=25-initname=random')
    >>> jpaths.append('demo-K=25-initname=smart')
    >>> jpaths.append('demo-K=50-initname=random')
    >>> jpaths.append('demo-K=50-initname=smart')
    >>> PPListMap = makePPListMapFromJPattern(jpathList=jpaths)
    >>> makeListOfJPatternsWithSpecificVals(PPListMap, key='initname', K='25')
    ['demo-K=25-initname=random', 'demo-K=25-initname=smart']
    >>> makeListOfJPatternsWithSpecificVals(PPListMap, key='initname')
    ['demo-K=*-initname=random', 'demo-K=*-initname=smart']
    '''
    assert key in PPListMap
    if vals is None:
        vals = PPListMap[key]
    jpattern = makeJPatternWithSpecificVals(PPListMap,
                                            doBest='.best' in vals,
                                            **keyValPairs)
    jpList = list()
    for v in vals:
        wildcardkey = "%s=*" % (key)
        keyandval = "%s=%s" % (key, v)
        jpath = jpattern.replace(wildcardkey, keyandval)
        jpList.append(jpath)
    return jpList


def makeJPatternWithSpecificVals(PPListMap,
                                 prefixfilepath='',
                                 doBest=False,
                                 **keyValPairs):
    '''
    Example
    -------
    >>> jpaths = []
    >>> jpaths.append('demo-K=25-initname=random')
    >>> jpaths.append('demo-K=25-initname=smart')
    >>> jpaths.append('demo-K=50-initname=random')
    >>> jpaths.append('demo-K=50-initname=smart')
    >>> PPListMap = makePPListMapFromJPattern(jpathList=jpaths)
    >>> makeJPatternWithSpecificVals(PPListMap, K='25')
    'demo-K=25-initname=*'
    >>> makeJPatternWithSpecificVals(PPListMap, initname='smart')
    'demo-K=*-initname=smart'
    '''
    if doBest:
        jpattern = '.best'
    else:
        jpattern = ''
    for key in PPListMap:
        if len(jpattern) > 0:
            jpattern += '-'

        # Determine specific value to use next
        if len(PPListMap[key]) == 1:
            val = PPListMap[key][0]
        elif key in keyValPairs:
            val = keyValPairs[key]
        else:
            val = '*'

        # Append the value to jpattern string
        if key.startswith('field'):
            jpattern += val
        else:
            jpattern += '%s=%s' % (key, val)
    if len(prefixfilepath) > 0:
        jpattern = os.path.join(prefixfilepath, jpattern)
    return jpattern


def makePPListMapFromJPattern(jpathPattern=None,
                              jpathList=None,
                              verbose=0):
    ''' Make dict that indicates all possible parameters for jobs.

    PPList stands for Possible Parameter List.

    Example
    -------
    >>> jpaths = []
    >>> jpaths.append('demo-K=25-initname=random')
    >>> jpaths.append('demo-K=25-initname=smart')
    >>> jpaths.append('demo-K=50-initname=random')
    >>> jpaths.append('demo-K=50-initname=smart')
    >>> jpaths.append('demo-K=100-initname=random')
    >>> jpaths.append('demo-K=100-initname=smart')
    >>> PPListMap = makePPListMapFromJPattern(jpathList=jpaths)
    >>> PPListMap['K']
    ['25', '50', '100']
    >>> PPListMap['initname']
    ['random', 'smart']

    Returns
    -------
    PPListMap : dict mapping str param names to lists of possible values
    '''
    if jpathList is None:
        if jpathPattern.count('*') == 0:
            jpathPattern += "*"
        jpathList = glob.glob(jpathPattern)

    if verbose:
        print('Looking for jobs with pattern:')
        print(jpathPattern)
        print('%d candidates found' % (len(jpathList)))
        print('    (before filtering by keywords)')

    if len(jpathList) == 0:
        raise ValueError('No matching jobs found.')

    PPDict = defaultdict(set)
    for jID, jpath in enumerate(jpathList):
        jdict = jpath2jdict(jpath)
        if jID > 0:
            if len(list(jdict.keys())) != len(list(PPDict.keys())):
                raise ValueError('Inconsistent key lists!')
            for key in jdict:
                if key not in PPDict:
                    raise ValueError('Inconsistent key lists!')
        for key in jdict:
            PPDict[key].add(jdict[key])

    PPListMap = OrderedDict()
    for key in jdict:
        try:
            numvals = sorted([float(x) for x in PPDict[key]])
            vals = list()
            for v in numvals:
                for s in PPDict[key]:
                    if float(s) == v:
                        vals.append(s)
        except ValueError:
            vals = [str(x) for x in sorted(PPDict[key])]
        PPListMap[key] = vals
    return PPListMap


def filterJobs(jpathPattern,
               returnAll=0, verbose=0, **reqKwArgs):
    for key in SkipKeys:
        if key in reqKwArgs:
            del reqKwArgs[key]

    jpathdir = os.path.sep.join(jpathPattern.split(os.path.sep)[:-1])
    if not os.path.isdir(jpathdir):
        raise ValueError('Not valid directory:\n %s' % (jpathdir))
    if not jpathPattern.endswith('*'):
        jpathPattern += '*'
    jpathList = glob.glob(jpathPattern)

    if verbose:
        print('Looking for jobs with pattern:')
        print(jpathPattern)
        print('%d candidates found' % (len(jpathList)))
        print('    (before filtering by keywords)')

    if len(jpathList) == 0:
        raise ValueError('No matching jobs found.')

    if verbose:
        print('\nRequirements:')
        for key in reqKwArgs:
            print('%s = %s' % (key, reqKwArgs[key]))

    keepListP = list()  # list of paths to keep
    keepListD = list()  # list of dicts to keep (one for each path)
    reqKwMatches = defaultdict(int)
    for jpath in jpathList:
        jdict = jpath2jdict(jpath)
        doKeep = True
        for reqkey in reqKwArgs:
            if reqkey not in jdict:
                doKeep = False
                continue
            reqval = reqKwArgs[reqkey]
            if jdict[reqkey] != str(reqval):
                doKeep = False
            else:
                reqKwMatches[reqkey] += 1
        if doKeep:
            keepListP.append(jpath)
            keepListD.append(jdict)

    if len(keepListP) == 0:
        for reqkey in reqKwArgs:
            if reqKwMatches[reqkey] == 0:
                raise ValueError('BAD REQUIRED PARAMETER.\n' +
                                 'No matches found for %s=%s: ' %
                                 (reqkey, reqKwArgs[reqkey]))

    if verbose:
        print('\nCandidates matching requirements')
        for p in keepListP:
            print(p.split(os.path.sep)[-1])

    # Figure out intelligent labels for the final jobs
    K = len(keepListD)
    varKeys = set()
    for kA in range(K):
        for kB in range(kA + 1, K):
            varKeys.update(findKeysWithDiffVals(keepListD[kA], keepListD[kB]))
    varKeys = [x for x in varKeys]

    if returnAll:
        return keepListP

    RangeMap = dict()
    for key in varKeys:
        RangeMap[key] = set()
        for jdict in keepListD:
            RangeMap[key].add(jdict[key])
        RangeMap[key] = [x for x in RangeMap[key]]  # to list
        try:
            float(RangeMap[key][0])
            RangeMap[key].sort(key=float)
        except ValueError as e:
            RangeMap[key].sort()

    if len(varKeys) > 1:
        print('ERROR! Need to constrain more variables')
        for key in RangeMap:
            print(key, RangeMap[key])
        raise ValueError('ERROR! Need to constrain more variables')

    elif len(varKeys) == 1:
        plotkey = varKeys[0]
        if plotkey.count('initname'):
            legNames = ['%s' % (x) for x in RangeMap[plotkey]]
        else:
            legNames = ['%s=%s' % (plotkey, x) for x in RangeMap[plotkey]]

        # Build list of final jpaths in order of decided legend
        keepListFinal = list()
        for x in RangeMap[plotkey]:
            for jID, jdict in enumerate(keepListD):
                if jdict[plotkey] == x:
                    keepListFinal.append(keepListP[jID])
    else:
        keepListFinal = keepListP[:1]
        legNames = [None]

    if verbose:
        print('\nLegend entries for selected jobs (auto-selected)')
        for name in legNames:
            print(name)

    return keepListFinal, legNames


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataName', default='AsteriskK8')
    parser.add_argument('jobName', default='bm')
    args, unkList = parser.parse_known_args()
    reqDict = BNPYArgParser.arglist_to_kwargs(unkList,
                                              doConvertFromStr=False)
    jpath = os.path.join(os.environ['BNPYOUTDIR'],
                         args.dataName,
                         args.jobName)

    keepJobs, legNames = filterJobs(jpath, verbose=1, **reqDict)
