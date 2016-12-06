'''
Classes
-------
LearnAlg
    Defines some generic routines for
        * saving global parameters
        * assessing convergence
        * printing progress updates to stdout
        * recording run-time
'''
import numpy as np
import time
import logging
import os
import sys
import scipy.io
import ElapsedTimeLogger
from bnpy.ioutil import ModelWriter
from bnpy.util import isEvenlyDivisibleFloat

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)


class LearnAlg(object):

    """ Abstract base class for learning algorithms that train HModels.

    Attributes
    ------
    task_output_path : str
        file system path to directory where files are saved
    seed : int
        seed used for random initialization
    PRNG : np.random.RandomState
        random number generator
    algParams : dict
        keyword parameters controlling algorithm behavior
        default values for each algorithm live in config/
        Can be overrided by keyword arguments specified by user
    outputParams : dict
        keyword parameters controlling saving to file / logs
    """

    def __init__(self, task_output_path=None, seed=0,
                 algParams=dict(), outputParams=dict(),
                 BNPYRunKwArgs=dict()):
        ''' Constructs and returns a LearnAlg object
        '''
        if isinstance(task_output_path, str):
            self.task_output_path = os.path.splitext(task_output_path)[0]
        else:
            self.task_output_path = None
        self.seed = int(seed)
        self.PRNG = np.random.RandomState(self.seed)
        self.algParams = algParams
        self.outputParams = outputParams
        self.BNPYRunKwArgs = BNPYRunKwArgs
        self.lap_list = list()
        self.loss_list = list()
        self.SavedIters = set()
        self.PrintIters = set()
        self.totalDataUnitsProcessed = 0
        self.status = 'active. not converged.'

        self.algParamsLP = dict()
        for k, v in algParams.items():
            if k.count('LP') > 0:
                self.algParamsLP[k] = v

    def fit(self, hmodel, Data):
        ''' Execute learning algorithm to train hmodel on Data.

        This method is extended by any subclass of LearnAlg

        Returns
        -------
        Info : dict of diagnostics about this run
        '''
        pass

    def set_random_seed_at_lap(self, lap):
        ''' Set internal random generator based on current lap.

        Reset the seed deterministically for each lap.
        using combination of seed attribute (unique to this run),
        and the provided lap argument. This allows reproducing
        exact values from this run later without starting over.

        Post Condition
        ------
        self.PRNG rest to new random seed.
        '''
        if isEvenlyDivisibleFloat(lap, 1.0):
            self.PRNG = np.random.RandomState(self.seed + int(lap))

    def set_start_time_now(self):
        ''' Record start time (in seconds since 1970)
        '''
        self.start_time = time.time()

    def updateNumDataProcessed(self, N):
        ''' Update internal count of number of data observations processed.
            Each lap thru dataset of size N, this should be updated by N
        '''
        self.totalDataUnitsProcessed += N

    def get_elapsed_time(self):
        ''' Returns float of elapsed time (in seconds) since this object's
            set_start_time_now() method was called
        '''
        return time.time() - self.start_time

    def buildRunInfo(self, Data, **kwargs):
        ''' Create dict of information about the current run.

        Returns
        ------
        Info : dict
            contains information about completed run.
        '''
        # Convert TraceLaps from set to 1D array, sorted in ascending order
        lap_history = np.asarray(self.lap_list)
        loss_history = np.asarray(self.loss_list)
        return dict(status=self.status,
                    task_output_path=self.task_output_path,
                    loss_history=loss_history,
                    lap_history=lap_history,
                    Data=Data,
                    elapsedTimeInSec=time.time() - self.start_time,
                    **kwargs)

    def hasMove(self, moveName):
        if moveName in self.algParams:
            return True
        return False

    def verify_monotonic_decrease(
            self, cur_loss=0.00001, prev_loss=0, lapFrac=None):
        ''' Verify current loss does not increase from previous loss

        Returns
        -------
        boolean : True if monotonic decrease, False otherwise
        '''
        if np.isnan(cur_loss):
            raise ValueError("Evidence should never be NaN")
        if np.isinf(prev_loss):
            return False
        isDecreasing = cur_loss <= prev_loss

        thr = self.algParams['convergeThrELBO']
        isWithinTHR = np.abs(cur_loss - prev_loss) < thr
        mLPkey = 'doMemoizeLocalParams'
        if not isDecreasing and not isWithinTHR:
            serious = True
            if mLPkey in self.algParams and not self.algParams[mLPkey]:
                warnMsg = 'loss increased when doMemoizeLocalParams=0'
                warnMsg += '(monotonic decrease not guaranteed)\n'
                serious = False
            else:
                warnMsg = 'loss increased!\n'
            warnMsg += '    prev = % .15e\n' % (prev_loss)
            warnMsg += '     cur = % .15e\n' % (cur_loss)
            if lapFrac is None:
                prefix = "WARNING: "
            else:
                prefix = "WARNING @ %.3f: " % (lapFrac)

            if serious or not self.algParams['doShowSeriousWarningsOnly']:
                Log.error(prefix + warnMsg)

    def isSaveDiagnosticsCheckpoint(self, lap, nMstepUpdates):
        ''' Answer True/False whether to save trace stats now
        '''
        traceEvery = self.outputParams['traceEvery']
        if traceEvery <= 0:
            return False
        return isEvenlyDivisibleFloat(lap, traceEvery) \
            or nMstepUpdates < 3 \
            or lap in self.lap_list \
            or isEvenlyDivisibleFloat(lap, 1.0)

    def saveDiagnostics(self, lap, SS, loss, ActiveIDVec=None):
        ''' Save trace stats to disk
        '''
        if lap in self.lap_list:
            return

        self.lap_list.append(lap)
        self.loss_list.append(loss)

        # Exit here if we're not saving to disk
        if self.task_output_path is None:
            return

        # Record current state to plain-text files
        with open(self.mkfile('trace_lap.txt'), 'a') as f:
            f.write('%.4f\n' % (lap))
        with open(self.mkfile('trace_loss.txt'), 'a') as f:
            f.write('%.9e\n' % (loss))
        with open(self.mkfile('trace_elapsed_time_sec.txt'), 'a') as f:
            f.write('%.3f\n' % (self.get_elapsed_time()))
        with open(self.mkfile('trace_K.txt'), 'a') as f:
            f.write('%d\n' % (SS.K))
        with open(self.mkfile('trace_n_examples_total.txt'), 'a') as f:
            f.write('%d\n' % (self.totalDataUnitsProcessed))

        # Record active counts in plain-text files
        counts = SS.getCountVec()
        assert counts.ndim == 1
        counts = np.asarray(counts, dtype=np.float32)
        np.maximum(counts, 0, out=counts)
        with open(self.mkfile('active_counts.txt'), 'a') as f:
            flatstr = ' '.join(['%.3f' % x for x in counts])
            f.write(flatstr + '\n')

        with open(self.mkfile('active_uids.txt'), 'a') as f:
            if ActiveIDVec is None:
                if SS is None:
                    ActiveIDVec = np.arange(SS.K)
                else:
                    ActiveIDVec = SS.uids
            flatstr = ' '.join(['%d' % x for x in ActiveIDVec])
            f.write(flatstr + '\n')

        if SS.hasSelectionTerm('DocUsageCount'):
            ucount = SS.getSelectionTerm('DocUsageCount')
            flatstr = ' '.join(['%d' % x for x in ucount])
            with open(self.mkfile('active_doc_counts.txt'), 'a') as f:
                f.write(flatstr + '\n')

    def isCountVecConverged(self, Nvec, prevNvec, batchID=None):
        if Nvec.size != prevNvec.size:
            # Warning: the old value of maxDiff is still used for printing
            return False

        maxDiff = np.max(np.abs(Nvec - prevNvec))
        isConverged = maxDiff < self.algParams['convergeThr']
        if batchID is not None:
            if not hasattr(self, 'ConvergeInfoByBatch'):
                self.ConvergeInfoByBatch = dict()
            self.ConvergeInfoByBatch[batchID] = dict(
                isConverged=isConverged,
                maxDiff=maxDiff)
            isConverged = np.min([
                self.ConvergeInfoByBatch[b]['isConverged']
                for b in self.ConvergeInfoByBatch])
            maxDiff = np.max([
                self.ConvergeInfoByBatch[b]['maxDiff']
                for b in self.ConvergeInfoByBatch])
        self.ConvergeInfo = dict(isConverged=isConverged,
                                 maxDiff=maxDiff)
        return isConverged

    def isSaveParamsCheckpoint(self, lap, nMstepUpdates):
        ''' Answer True/False whether to save full model now
        '''
        s = self.outputParams['saveEveryLogScaleFactor']
        sE = self.outputParams['saveEvery']
        if s > 0:
            new_sE = np.maximum(np.maximum(sE, sE ** s), sE * s)
            if (lap >= new_sE):
                self.outputParams['saveEvery'] = new_sE
            if lap > 1.0:
                self.outputParams['saveEvery'] = \
                    np.ceil(self.outputParams['saveEvery'])
        saveEvery = self.outputParams['saveEvery']
        if saveEvery <= 0 or self.task_output_path is None:
            return False
        return isEvenlyDivisibleFloat(lap, saveEvery) \
            or (isEvenlyDivisibleFloat(lap, 1.0) and
                lap <= self.outputParams['saveEarly']) \
            or nMstepUpdates < 3 \
            or np.allclose(lap, 1.0) \
            or np.allclose(lap, 2.0) \
            or np.allclose(lap, 4.0) \
            or np.allclose(lap, 8.0)

    def saveParams(self, lap, hmodel, SS=None, **kwargs):
        ''' Save current model to disk
        '''
        if lap in self.SavedIters or self.task_output_path is None:
            return
        ElapsedTimeLogger.startEvent("io", "saveparams")
        self.SavedIters.add(lap)
        prefix = ModelWriter.makePrefixForLap(lap)
        with open(self.mkfile('snapshot_lap.txt'), 'a') as f:
            f.write('%.4f\n' % (lap))
        with open(self.mkfile('snapshot_elapsed_time_sec.txt'), 'a') as f:
            f.write('%.3f\n' % (self.get_elapsed_time()))
        if self.outputParams['doSaveFullModel']:
            ModelWriter.save_model(
                hmodel, self.task_output_path, prefix,
                doSavePriorInfo=np.allclose(lap, 0.0),
                doLinkBest=True,
                doSaveObsModel=self.outputParams['doSaveObsModel'])
        if self.outputParams['doSaveTopicModel']:
            ModelWriter.saveTopicModel(
                hmodel, SS, self.task_output_path, prefix, **kwargs)
        ElapsedTimeLogger.stopEvent("io", "saveparams")

    def mkfile(self, fname):
        """ Create full system path for provided file basename.

        Returns
        -------
        fpath : str

        Examples
        -------
        >>> mkfile("K.txt")
        "/path/to/output/K.txt"
        """
        return os.path.join(self.task_output_path, fname)

    def setStatus(self, lapFrac, isConverged):
        nLapsCompleted = lapFrac - self.algParams['startLap']

        nLap = self.algParams['nLap']
        minLapReq = np.minimum(nLap, self.algParams['minLaps'])
        minLapsCompleted = nLapsCompleted >= minLapReq
        if isConverged and minLapsCompleted:
            self.status = "done. converged."
        elif isConverged:
            self.status = "active. converged but minLaps unfinished."
        elif nLapsCompleted < nLap:
            self.status = "active. not converged."
        else:
            self.status = "done. not converged. max laps thru data exceeded."

        if self.task_output_path is not None:
            with open(self.mkfile('status.txt'), 'w') as f:
                f.write(self.status + '\n')

    def isLogCheckpoint(self, lap, nMstepUpdates):
        ''' Answer True/False whether to save full model now
        '''
        printEvery = self.outputParams['printEvery']
        if printEvery <= 0:
            return False
        return isEvenlyDivisibleFloat(lap, printEvery) \
            or nMstepUpdates < 3

    def printStateToLog(self, hmodel, loss, lap, iterid,
                        isFinal=0, rho=None):
        """ Print state of provided model and progress variables to log.
        """
        from bnpy import getCurMemCost_MiB
        if hasattr(self, 'ConvergeInfo') and 'maxDiff' in self.ConvergeInfo:
            countStr = 'Ndiff%9.3f' % (self.ConvergeInfo['maxDiff'])
        else:
            countStr = ''

        if rho is None:
            rhoStr = ''
        else:
            rhoStr = 'lrate %.4f' % (rho)

        if iterid == lap:
            lapStr = '%7d' % (lap)
        else:
            lapStr = '%7.3f' % (lap)
        maxLap = self.algParams['nLap'] + self.algParams['startLap']
        maxLapStr = '%d' % (maxLap)

        logmsg = '  %s/%s after %6.0f sec. |'
        logmsg += ' %8.1f MiB | K %4d | loss % .9e | %s %s'
        logmsg = logmsg % (lapStr,
                           maxLapStr,
                           self.get_elapsed_time(),
                           getCurMemCost_MiB(),
                           hmodel.allocModel.K,
                           loss,
                           countStr,
                           rhoStr)

        if iterid not in self.PrintIters:
            self.PrintIters.add(iterid)
            Log.info(logmsg)
        if isFinal:
            Log.info('... %s' % (self.status))

    def print_msg(self, msg):
        ''' Prints a string msg to the log for this training experiment.

        Avoids need to import all logging utilities into a subclass.
        '''
        Log.info(msg)

    # Checkpoints
    #########################################################
    def isFirstBatch(self, lapFrac):
        ''' Returns True/False if at first batch in the current lap.
        '''
        if self.lapFracInc == 1.0:  # Special case, nBatch == 1
            isFirstBatch = True
        else:
            lapRem = lapFrac - np.floor(lapFrac)
            isFirstBatch = np.allclose(lapRem, self.lapFracInc)
        return isFirstBatch

    def isLastBatch(self, lapFrac):
        ''' Returns True/False if at last batch in the current lap.
        '''
        lapRem = np.abs(lapFrac - np.round(lapFrac))
        return np.allclose(lapRem, 0.0, rtol=0, atol=1e-8)

    def do_birth_at_lap(self, lapFrac):
        ''' Returns True/False for whether birth happens at given lap
        '''
        if 'birth' not in self.algParams:
            return False
        nLapTotal = self.algParams['nLap']
        frac = self.algParams['birth']['fracLapsBirth']
        if lapFrac > nLapTotal:
            return False
        return (nLapTotal <= 5) or (lapFrac <= np.ceil(frac * nLapTotal))

    def doMergePrepAtLap(self, lapFrac):
        if 'merge' not in self.algParams:
            return False
        return lapFrac > self.algParams['merge']['mergeStartLap'] \
            and self.isFirstBatch(lapFrac)

    def eval_custom_func(self, isFinal=0, isInitial=0, lapFrac=0, **kwargs):
        ''' Evaluates a custom hook function
        '''

        cFuncPath = self.outputParams['customFuncPath']
        if cFuncPath is None or cFuncPath == 'None':
            return None

        cbName = str(cFuncPath)
        ElapsedTimeLogger.startEvent('callback', cbName)

        cFuncArgs_string = self.outputParams['customFuncArgs']
        nLapTotal = self.algParams['nLap']
        if isinstance(cFuncPath, str):
            cFuncPath = cFuncPath.replace(".py", "")
            pathParts = cFuncPath.split(os.path.sep)
            if len(pathParts) > 1:
                # Absolute path provided
                cFuncDir = os.path.expandvars(os.path.sep.join(pathParts[:-1]))
                sys.path.append(cFuncDir)
                cFuncModName = pathParts[-1]
                cFuncModule = __import__(cFuncModName, fromlist=[])
            else:
                # Treat as relative path to file in bnpy.callbacks
                cFuncModule = __import__(
                    'bnpy.callbacks.', fromlist=[cFuncPath])
                cFuncModule = getattr(cFuncModule, cFuncPath)
        else:
            cFuncModule = cFuncPath  # directly passed in as object

        kwargs['nLap'] = self.algParams['nLap']
        kwargs['lapFrac'] = lapFrac
        kwargs['isFinal'] = isFinal
        kwargs['isInitial'] = isInitial
        if isInitial:
            kwargs['lapFrac'] = 0
            kwargs['iterid'] = 0

        hasCBFuncs = hasattr(cFuncModule, 'onBatchComplete') or \
            hasattr(cFuncModule, 'onLapComplete') or \
            hasattr(cFuncModule, 'onAlgorithmComplete')
        if not hasCBFuncs:
            raise ValueError("Specified customFuncPath has no callbacks!")
        if hasattr(cFuncModule, 'onBatchComplete') and not isFinal:
            cFuncModule.onBatchComplete(args=cFuncArgs_string, **kwargs)
        if hasattr(cFuncModule, 'onLapComplete') \
           and isEvenlyDivisibleFloat(lapFrac, 1.0) and not isFinal:
            cFuncModule.onLapComplete(args=cFuncArgs_string, **kwargs)
        if hasattr(cFuncModule, 'onAlgorithmComplete') \
           and isFinal:
            cFuncModule.onAlgorithmComplete(args=cFuncArgs_string, **kwargs)
        ElapsedTimeLogger.stopEvent('callback', cbName)

    def saveDebugStateAtBatch(self, name, batchID, LPchunk=None, SS=None,
                              SSchunk=None, hmodel=None,
                              Dchunk=None):
        if self.outputParams['debugBatch'] == batchID:
            debugLap = self.outputParams['debugLap']
            debugLapBuffer = self.outputParams['debugLapBuffer']
            import joblib
            if self.lapFrac < 1:
                joblib.dump(dict(Dchunk=Dchunk),
                            os.path.join(self.task_output_path, 'Debug-Data.dump'))
            belowWindow = self.lapFrac < debugLap - debugLapBuffer
            aboveWindow = self.lapFrac > debugLap + debugLapBuffer
            if belowWindow or aboveWindow:
                return
            filename = 'DebugLap%04.0f-%s.dump' % (np.ceil(self.lapFrac), name)
            SaveVars = dict(LP=LPchunk, SS=SS, hmodel=hmodel,
                            SSchunk=SSchunk,
                            lapFrac=self.lapFrac)
            joblib.dump(SaveVars, os.path.join(self.task_output_path, filename))
            if self.lapFrac < 1:
                joblib.dump(dict(Dchunk=Dchunk),
                            os.path.join(self.task_output_path, 'Debug-Data.dump'))


def makeDictOfAllWorkspaceVars(**kwargs):
    ''' Create dict of all active variables in workspace

    Necessary to avoid call to self.

    Returns
    ------
    kwargs : dict
        key/value for every variable in the workspace.
    '''
    if 'self' in kwargs:
        kwargs['learnAlg'] = kwargs.pop('self')
    if 'lap' in kwargs:
        kwargs['lapFrac'] = kwargs['lap']
    for key in kwargs.keys():
        if key.startswith('_'):
            kwargs.pop(key)
    return kwargs
