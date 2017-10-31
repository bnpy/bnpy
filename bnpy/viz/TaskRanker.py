from builtins import *
import os
import numpy as np
import glob

from . import JobFilter

from bnpy.util import as1D
from bnpy.ioutil.BNPYArgParser import parse_task_ids, arglist_to_kwargs
from .JobFilter import filterJobs


def rankTasksForSingleJobOnDisk(joboutpath, **kwargs):
    ''' Make files for job allowing rank-based referencing of tasks.

    Post Condition
    ----------
    joboutputpath will contain files:
    - .rank1/
    - .rank2/
    - ...
    - .rankN/
        which are each references (symlinks) to tasks in that directory.
    '''
    # First, rank the tasks from best to worst
    sortedScores, sortedTaskIDs = rankTasksForSingleJob(joboutpath, **kwargs)

    # TODO If we've sorted these same tasks before, just quit early

    # Remove all old hidden rank files
    hiddenFileList = glob.glob(os.path.join(joboutpath, '.*'))
    for fpath in hiddenFileList:
        if os.path.islink(fpath):
            os.unlink(fpath)

    # Save record of new info file
    for rankID, taskidstr in enumerate(sortedTaskIDs):
        rankID += 1  # 1 based indexing!

        # Make symlink to file like .rank1 or .rank27
        os.symlink(os.path.join(joboutpath, taskidstr),
                   os.path.join(joboutpath, '.rank' + str(rankID)))

        if rankID == 1:
            os.symlink(os.path.join(joboutpath, taskidstr),
                       os.path.join(joboutpath, '.best'))
        if rankID == len(sortedTaskIDs):
            os.symlink(os.path.join(joboutpath, taskidstr),
                       os.path.join(joboutpath, '.worst'))
        if len(sortedTaskIDs) > 3:
            if rankID == int(np.ceil(len(sortedTaskIDs) / 2.0)):
                os.symlink(os.path.join(joboutpath, taskidstr),
                           os.path.join(joboutpath, '.median'))
    # Return scores and associated task ids, sorted from best to worst
    return sortedScores, sortedTaskIDs

def rankTasksForSingleJob(
        joboutpath,
        multiTaskRankOrder='bigtosmall',
        **kwargs):
    ''' Get list of tasks for job, ranked best-to-worst by final ELBO score

    Returns
    ----------
    sortedtaskIDs : list of task names, each entry is an int
    '''
    # Read in the score for each task
    scores, taskids = scoreTasksForSingleJob(joboutpath, **kwargs)
    # Sort in descending order, largest to smallest!
    if multiTaskRankOrder == 'bigtosmall':
        sortIDs = np.argsort(-1 * scores)
    else:
        sortIDs = np.argsort(scores)
    return scores[sortIDs], [taskids[t] for t in sortIDs]

def scoreTasksForSingleJob(
        joboutpath,
        taskids='all',
        scoreTxtFile='evidence.txt',
        singleTaskScoreFunc=np.max,
        extraTxtFileDict=None,
        **kwargs):
    ''' Get list of tasks for job, ranked best-to-worst by final ELBO score

    Returns
    ----------
    scores : list of scores, each entry is a float
    taskids : list of ids, each entry is an int
    '''
    taskids = parse_task_ids(joboutpath, taskids)
    # Read in the score for each task
    scores = np.zeros(len(taskids))
    for tid, taskidstr in enumerate(taskids):
        assert isinstance(taskidstr, str)
        try:
            scorevec = np.loadtxt(
                os.path.join(joboutpath, taskidstr, scoreTxtFile))
        except IOError as e:
            scores[tid] = np.nan
            continue
        kwargs = dict()
        if extraTxtFileDict is not None:
            for key, xTxtFile in extraTxtFileDict:
                kwargs[key] = np.loadtxt(
                    os.path.join(joboutpath, taskidstr, xTxtFile))
        scores[tid] = singleTaskScoreFunc(scorevec, **kwargs)
    mask = np.flatnonzero(np.isfinite(scores))
    return scores[mask], np.asarray(taskids)[mask].tolist()


def markBestAmongJobPatternOnDisk(jobPattern, key='initname'):
    ''' Create symlink to best run among all jobs matching given pattern.

    Post Condition
    --------------
    Creates new job output path on disk, with pattern
    $BNPYOUTDIR/$dataName/.best-$jobPattern.replace('$key=*', '$key=best')
    '''
    prefixfilepath = os.path.sep.join(jobPattern.split(os.path.sep)[:-1])
    PPListMap = JobFilter.makePPListMapFromJPattern(jobPattern)
    bestArgs = dict()
    bestArgs[key] = '.best'
    bestpath = JobFilter.makeJPatternWithSpecificVals(PPListMap, **bestArgs)
    bestpath = '.best-' + bestpath
    bestpath = os.path.join(prefixfilepath, bestpath)

    # Remove all old hidden rank files
    if os.path.islink(bestpath):
        os.unlink(bestpath)

    jpath = findBestAmongJobPattern(jobPattern, key=key)
    os.symlink(jpath, bestpath)


def findBestAmongJobPattern(jobPattern, key='initname',
                            prefixfilepath='',
                            **kwargs):
    ''' Identify single best run among all jobs/tasks matching given pattern.

    Returns
    -------
    bestjpath : str
        valid file path
    '''
    prefixfilepath = os.path.sep.join(jobPattern.split(os.path.sep)[:-1])
    PPListMap = JobFilter.makePPListMapFromJPattern(jobPattern)
    jpaths = JobFilter.makeListOfJPatternsWithSpecificVals(
        PPListMap,
        key=key,
        prefixfilepath=prefixfilepath)

    Scores = np.zeros(len(jpaths))
    for jID, jpath in enumerate(jpaths):
        jtaskpath = os.path.join(jpath, '.best')
        if not os.path.exists(jtaskpath):
            rankTasksForSingleJobOnDisk(jpath)
        if not os.path.exists(jtaskpath):
            msg = 'Does not exist: %s' % (jtaskpath)
            raise ValueError(msg)

        Scores[jID] = np.loadtxt(os.path.join(jtaskpath, 'evidence.txt'))[-1]
    sortIDs = np.argsort(-1 * Scores)
    return jpaths[sortIDs[0]]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataName')
    parser.add_argument('jobNamePattern')
    args, unkList = parser.parse_known_args()
    reqDict = arglist_to_kwargs(unkList)
    jpathPattern = os.path.join(os.environ['BNPYOUTDIR'],
                                args.dataName,
                                args.jobNamePattern)

    jobPaths = filterJobs(
        jpathPattern, returnAll=1, verbose=0, **reqDict)
    for jpath in jobPaths:
        rankTasksForSingleJobOnDisk(jpath)
