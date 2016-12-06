'''
Implementation of parallel memoized variational algorithm for bnpy models.
'''
import numpy as np
import multiprocessing
import os
import ElapsedTimeLogger 
import scipy.sparse

from collections import defaultdict

from bnpy.birthmove.BCreateManyProposals \
    import makeSummariesForManyBirthProposals

from bnpy.birthmove import \
    BLogger, \
    selectShortListForBirthAtLapStart, \
    summarizeRestrictedLocalStep, \
    selectCompsForBirthAtCurrentBatch
from bnpy.mergemove import MLogger, SLogger
from bnpy.mergemove import selectCandidateMergePairs
from bnpy.deletemove import DLogger, selectCandidateDeleteComps
from bnpy.util import argsort_bigtosmall_stable
from bnpy.util.SparseRespUtil import sparsifyResp
from LearnAlg import makeDictOfAllWorkspaceVars
from LearnAlg import LearnAlg
from bnpy.viz.PrintTopics import count2str

# If abs val of two ELBOs differs by less than this small constant
# We figure its close enough and accept the model with smaller K
ELBO_GAP_ACCEPT_TOL = 0.000001

class MemoVBMovesAlg(LearnAlg):

    def __init__(self, **kwargs):
        ''' Constructor for LearnAlg.
        '''
        # Initialize instance vars related to
        # birth / merge / delete records
        LearnAlg.__init__(self, **kwargs)
        self.SSmemory = dict()
        self.LPmemory = dict()
        self.LastUpdateLap = dict()

    def makeNewUIDs(self, nMoves=1, b_Kfresh=0, **kwargs):
        newUIDs = np.arange(self.maxUID + 1,
                            self.maxUID + nMoves * b_Kfresh + 1)
        self.maxUID += newUIDs.size
        return newUIDs

    def fit(self, hmodel, DataIterator, **kwargs):
        ''' Run learning algorithm that fits parameters of hmodel to Data.

        Returns
        --------
        Info : dict of run information.

        Post Condition
        --------
        hmodel updated in place with improved global parameters.
        '''
        self.set_start_time_now()
        self.memoLPkeys = hmodel.allocModel.get_keys_for_memoized_local_params()

        origmodel = hmodel
        self.maxUID = hmodel.obsModel.K - 1

        # Initialize Progress Tracking vars like nBatch, lapFrac, etc.
        iterid, lapFrac = self.initProgressTrackVars(DataIterator)

        # Save initial state
        self.saveParams(lapFrac, hmodel)

        # Custom func hook
        self.eval_custom_func(
            isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))
        ElapsedTimeLogger.writeToLogOnLapCompleted(lapFrac)

        # Begin loop over batches of data...
        SS = None
        isConverged = False
        loss = np.inf
        MoveLog = list()
        MoveRecordsByUID = dict()
        ConvStatus = np.zeros(DataIterator.nBatch)
        while DataIterator.has_next_batch():

            batchID = DataIterator.get_next_batch(batchIDOnly=1)

            # Update progress-tracking variables
            iterid += 1
            lapFrac = (iterid + 1) * self.lapFracInc
            self.lapFrac = lapFrac
            self.set_random_seed_at_lap(lapFrac)

            # Debug print header
            if self.doDebugVerbose():
                self.print_msg('========================== lap %.2f batch %d'
                               % (lapFrac, batchID))

            # Reset at top of every lap
            if self.isFirstBatch(lapFrac):
                MovePlans = dict()
                if SS is not None and SS.hasSelectionTerms():
                    SS._SelectTerms.setAllFieldsToZero()
            MovePlans = self.makeMovePlans(
                hmodel, SS,
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                lapFrac=lapFrac)

            # Local/Summary step for current batch
            SSbatch = self.calcLocalParamsAndSummarize_withExpansionMoves(
                DataIterator, hmodel,
                SS=SS,
                batchID=batchID,
                lapFrac=lapFrac,
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                MoveLog=MoveLog)

            self.saveDebugStateAtBatch(
                'Estep', batchID, SSchunk=SSbatch, SS=SS, hmodel=hmodel)

            # Incremental update of whole-data SS given new SSbatch
            oldSSbatch = self.loadBatchAndFastForward(
                batchID, lapFrac, MoveLog)
            SS = self.incrementWholeDataSummary(
                SS, SSbatch, oldSSbatch, lapFrac=lapFrac, hmodel=hmodel)
            self.SSmemory[batchID] = SSbatch
            del SSbatch
            del oldSSbatch
            self.LastUpdateLap[batchID] = lapFrac

            # Global step
            hmodel, didUpdate = self.globalStep(hmodel, SS, lapFrac)

            # ELBO calculation
            loss = -1 * hmodel.calc_evidence(
                SS=SS, afterGlobalStep=didUpdate, doLogElapsedTime=True)

            # Birth moves!
            if self.hasMove('birth') and hasattr(SS, 'propXSS'):
                hmodel, SS, loss, MoveLog, MoveRecordsByUID = \
                    self.runMoves_Birth(
                        hmodel, SS, loss, MovePlans,
                        MoveLog=MoveLog,
                        MoveRecordsByUID=MoveRecordsByUID,
                        lapFrac=lapFrac)

            if self.isLastBatch(lapFrac):
                # Delete move!
                if self.hasMove('delete') and 'd_targetUIDs' in MovePlans:
                    hmodel, SS, loss, MoveLog, MoveRecordsByUID = \
                        self.runMoves_Delete(
                            hmodel, SS, loss, MovePlans,
                            MoveLog=MoveLog,
                            MoveRecordsByUID=MoveRecordsByUID,
                            lapFrac=lapFrac,)
                if hasattr(SS, 'propXSS'):
                    del SS.propXSS

                # Merge move!
                if self.hasMove('merge') and 'm_UIDPairs' in MovePlans:
                    hmodel, SS, loss, MoveLog, MoveRecordsByUID = \
                        self.runMoves_Merge(
                            hmodel, SS, loss, MovePlans,
                            MoveLog=MoveLog,
                            MoveRecordsByUID=MoveRecordsByUID,
                            lapFrac=lapFrac,)
                # Afterwards, always discard any tracked merge terms
                SS.removeMergeTerms()

                # Shuffle : Rearrange order (big to small)
                if self.hasMove('shuffle'):
                    hmodel, SS, loss, MoveLog, MoveRecordsByUID = \
                        self.runMoves_Shuffle(
                            hmodel, SS, loss, MovePlans,
                            MoveLog=MoveLog,
                            MoveRecordsByUID=MoveRecordsByUID,
                            lapFrac=lapFrac,)

            nLapsCompleted = lapFrac - self.algParams['startLap']
            if nLapsCompleted > 1.0:
                # loss decreases monotonically AFTER first lap
                # verify function warns if this isn't happening
                self.verify_monotonic_decrease(loss, prev_loss, lapFrac)

            # Debug
            if self.doDebug() and lapFrac >= 1.0:
                self.verifyELBOTracking(hmodel, SS, loss, 
                    MoveLog=MoveLog, lapFrac=lapFrac)
            self.saveDebugStateAtBatch(
                'Mstep', batchID,
                SSchunk=self.SSmemory[batchID],
                SS=SS, hmodel=hmodel)

            # Assess convergence
            countVec = SS.getCountVec()
            if nLapsCompleted > 1.0:
                ConvStatus[batchID] = self.isCountVecConverged(
                    countVec, prevCountVec, batchID=batchID)
                isConverged = np.min(ConvStatus) and not \
                    self.hasMoreReasonableMoves(SS, MoveRecordsByUID, lapFrac)
                self.setStatus(lapFrac, isConverged)

            # Display progress
            if self.isLogCheckpoint(lapFrac, iterid):
                self.printStateToLog(hmodel, loss, lapFrac, iterid)

            # Save diagnostics and params
            if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
                self.saveDiagnostics(lapFrac, SS, loss)
            if self.isSaveParamsCheckpoint(lapFrac, iterid):
                self.saveParams(lapFrac, hmodel, SS,
                    didExactUpdateWithSS=didUpdate)

            # Custom func hook
            self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

            if self.isLastBatch(lapFrac):
                ElapsedTimeLogger.writeToLogOnLapCompleted(lapFrac)

                if isConverged and \
                    nLapsCompleted >= self.algParams['minLaps']:
                    break
            prevCountVec = countVec.copy()
            prev_loss = loss
            # .... end loop over data

        # Finished! Save, print and exit
        self.printStateToLog(hmodel, loss, lapFrac, iterid, isFinal=1)
        self.saveParams(lapFrac, hmodel, SS)
        self.eval_custom_func(
            isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

        # Births and merges require copies of original model object
        #  we need to make sure original reference has updated parameters, etc.
        if id(origmodel) != id(hmodel):
            origmodel.allocModel = hmodel.allocModel
            origmodel.obsModel = hmodel.obsModel

        # Return information about this run
        return self.buildRunInfo(DataIterator, loss=loss, SS=SS,
                                 SSmemory=self.SSmemory)

    def calcLocalParamsAndSummarize_withExpansionMoves(
            self, DataIterator, curModel,
            SS=None,
            batchID=0,
            lapFrac=0,
            MovePlans=None,
            MoveRecordsByUID=dict(),
            MoveLog=None,
            **kwargs):
        ''' Execute local step and summary step, with expansion proposals.

        Returns
        -------
        SSbatch : bnpy.suffstats.SuffStatBag
        '''
        # Fetch the current batch of data
        ElapsedTimeLogger.startEvent('io', 'loadbatch')
        Dbatch = DataIterator.getBatch(batchID=batchID)
        ElapsedTimeLogger.stopEvent('io', 'loadbatch')

        # Prepare the kwargs for the local and summary steps
        # including args for the desired merges/deletes/etc.
        if not isinstance(MovePlans, dict):
            MovePlans = dict()
        LPkwargs = self.algParamsLP
        # MovePlans indicates which merge pairs to track in local step.
        LPkwargs.update(MovePlans)
        trackDocUsage = 0
        if self.hasMove('birth'):
            if self.algParams['birth']['b_debugWriteHTML']:
                trackDocUsage = 1

        if self.algParams['doMemoizeLocalParams'] and batchID in self.LPmemory:
            oldbatchLP = self.load_batch_local_params_from_memory(batchID)
        else:
            oldbatchLP = None

        # Do the real work here: calc local params
        # Pass lap and batch info so logging happens
        LPbatch = curModel.calc_local_params(Dbatch, oldbatchLP,
            lapFrac=lapFrac, batchID=batchID,
            doLogElapsedTime=True, **LPkwargs)
        if self.algParams['doMemoizeLocalParams']:
            self.save_batch_local_params_to_memory(batchID, LPbatch, Dbatch)
        # Summary time!
        SSbatch = curModel.get_global_suff_stats(
            Dbatch, LPbatch,
            doPrecompEntropy=1,
            doTrackTruncationGrowth=1,
            doLogElapsedTime=True,
            trackDocUsage=trackDocUsage,
            **MovePlans)
        if 'm_UIDPairs' in MovePlans:
            SSbatch.setMergeUIDPairs(MovePlans['m_UIDPairs'])

        if SS is not None:
            # Force newest stats to have same unique ids as whole stats
            # If merges/shuffles/other moves have happened,
            # we want to be sure the new local stats have the same labels
            SSbatch.setUIDs(SS.uids)

        # Prepare current snapshot of whole-dataset stats
        # These must reflect the latest assignment to this batch,
        # AND all previous batches
        if self.hasMove('birth') or self.hasMove('delete'):
            if SS is None:
                curSSwhole = SSbatch.copy()
            else:
                curSSwhole = SS.copy(includeELBOTerms=1, includeMergeTerms=0)
                curSSwhole += SSbatch
                if lapFrac > 1.0:
                    oldSSbatch = self.loadBatchAndFastForward(
                        batchID, lapFrac, MoveLog, doCopy=1)
                    curSSwhole -= oldSSbatch
        # Prepare plans for which births to try,
        # using recently updated stats.
        if self.hasMove('birth'):
            # Determine what integer position we are with respect to this lap
            batchPos = np.round(
                (lapFrac - np.floor(lapFrac)) / self.lapFracInc)

            ElapsedTimeLogger.startEvent('birth', 'plan')
            MovePlans = self.makeMovePlans_Birth_AtBatch(
                curModel, curSSwhole,
                SSbatch=SSbatch,
                lapFrac=lapFrac,
                batchID=batchID,
                isFirstBatch=self.isFirstBatch(lapFrac),
                nBatch=self.nBatch,
                batchPos=batchPos,
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                **kwargs)
            ElapsedTimeLogger.stopEvent('birth', 'plan')

        # Prepare some logging stats        
        if 'b_nFailedProp' not in MovePlans:
            MovePlans['b_nFailedProp'] = 0
        if 'b_nTrial' not in MovePlans:
            MovePlans['b_nTrial'] = 0

        # Create a place to store each proposal, indexed by UID
        SSbatch.propXSS = dict()
        # Try each planned birth
        if 'b_targetUIDs' in MovePlans and len(MovePlans['b_targetUIDs']) > 0:
            ElapsedTimeLogger.startEvent('birth', 'localexpansion')
            newUIDs = self.makeNewUIDs(
                nMoves=len(MovePlans['b_targetUIDs']),
                **self.algParams['birth'])
            SSbatch.propXSS, MovePlans, MoveRecordsByUID = \
                 makeSummariesForManyBirthProposals(
                    Dslice=Dbatch,
                    curModel=curModel,
                    curLPslice=LPbatch,
                    curSSwhole=curSSwhole,
                    curSSslice=SSbatch,
                    LPkwargs=LPkwargs,
                    newUIDs=newUIDs,
                    MovePlans=MovePlans,
                    MoveRecordsByUID=MoveRecordsByUID,
                    taskoutpath=self.task_output_path,
                    lapFrac=lapFrac,
                    batchID=batchID,
                    seed=self.seed,
                    nBatch=self.nBatch,
                    batchPos=batchPos,
                    **self.algParams['birth'])
            ElapsedTimeLogger.stopEvent('birth', 'localexpansion')


        # Prepare deletes
        if 'd_targetUIDs' in MovePlans:
            ElapsedTimeLogger.startEvent('delete', 'localexpansion')
            targetUID = MovePlans['d_targetUIDs'][0]
            if hasattr(curSSwhole, 'propXSS') and \
                    targetUID in curSSwhole.propXSS:
                xInitSS = curSSwhole.propXSS[targetUID].copy(
                    includeELBOTerms=False)
                doBuildOnInit = True
            else:
                doBuildOnInit = False
                # Make copy of current suff stats (minus target state)
                # to inspire reclustering of junk state.
                xInitSS = curSSwhole.copy(
                    includeELBOTerms=False, includeMergeTerms=False)
                for uid in xInitSS.uids:
                    if uid not in MovePlans['d_absorbingUIDSet']:
                        xInitSS.removeComp(uid=uid)
                MovePlans['d_absorbingUIDs'] = xInitSS.uids
            # Run restricted local step
            DKwargs = self.algParams['delete']
            SSbatch.propXSS[targetUID], rInfo = summarizeRestrictedLocalStep(
                Dbatch, curModel, LPbatch, 
                curSSwhole=curSSwhole,
                xInitSS=xInitSS,
                doBuildOnInit=doBuildOnInit,
                xUIDs=xInitSS.uids,
                targetUID=targetUID,
                LPkwargs=LPkwargs,
                emptyPiFrac=0,
                lapFrac=lapFrac,
                nUpdateSteps=DKwargs['d_nRefineSteps'],
                d_initTargetDocTopicCount=DKwargs['d_initTargetDocTopicCount'],
                d_initWordCounts=DKwargs['d_initWordCounts'],
                )
            ElapsedTimeLogger.stopEvent('delete', 'localexpansion')
        return SSbatch

    def load_batch_local_params_from_memory(self, batchID, doCopy=0):
        ''' Load local parameter dict stored in memory for provided batchID

        TODO: Fastforward so recent truncation changes are accounted for.

        Returns
        -------
        batchLP : dict of local parameters specific to batchID
        '''
        batchLP = self.LPmemory[batchID]
        if isinstance(batchLP, str):
            ElapsedTimeLogger.startEvent('io', 'loadlocal')
            batchLPpath = os.path.abspath(batchLP)
            assert os.path.exists(batchLPpath)
            F = np.load(batchLPpath)
            indptr = np.arange(
                0, (F['D']+1)*F['nnzPerDoc'],
                F['nnzPerDoc'])
            batchLP = dict()
            batchLP['DocTopicCount'] = scipy.sparse.csr_matrix(
                (F['data'], F['indices'], indptr),
                shape=(F['D'], F['K'])).toarray()
            ElapsedTimeLogger.stopEvent('io', 'loadlocal')
        if doCopy:
            # Duplicating to avoid changing the raw data stored in LPmemory
            # Usually for debugging only
            batchLP = copy.deepcopy(batchLP)
        return batchLP

    def save_batch_local_params_to_memory(self, batchID, batchLP, batchData):
        ''' Store certain fields of the provided local parameters dict
              into "memory" for later retrieval.
            Fields to save determined by the memoLPkeys attribute of this alg.
        '''
        batchLP = dict(**batchLP) # make a copy
        allkeys = batchLP.keys()
        for key in allkeys:
            if key not in self.memoLPkeys:
                del batchLP[key]
        if len(batchLP.keys()) > 0:
            if self.algParams['doMemoizeLocalParams'] == 1:
                self.LPmemory[batchID] = batchLP
            elif self.algParams['doMemoizeLocalParams'] == 2:
                ElapsedTimeLogger.startEvent('io', 'savelocal')
                spDTC = sparsifyResp(
                    batchLP['DocTopicCount'],
                    self.algParams['nnzPerDocForStorage'])
                wc_D = batchLP['DocTopicCount'].sum(axis=1)
                wc_U = np.repeat(wc_D, self.algParams['nnzPerDocForStorage'])
                spDTC.data *= wc_U
                savepath = self.savedir.replace(os.environ['BNPYOUTDIR'], '')
                if os.path.exists('/ltmp/'):
                    savepath = '/ltmp/%s/' % (savepath)
                else:
                    savepath = '/tmp/%s/' % (savepath)
                from distutils.dir_util import mkpath
                mkpath(savepath)
                savepath = os.path.join(savepath, 'batch%d.npz' % (batchID))
                # Now actually save it!
                np.savez(savepath,
                    data=spDTC.data,
                    indices=spDTC.indices,
                    D=spDTC.shape[0],
                    K=spDTC.shape[1],
                    nnzPerDoc=spDTC.indptr[1])
                self.LPmemory[batchID] = savepath
                del batchLP
                del spDTC
                ElapsedTimeLogger.stopEvent('io', 'savelocal')

    def incrementWholeDataSummary(
            self, SS, SSbatch, oldSSbatch,
            hmodel=None,
            lapFrac=0):
        ''' Update whole dataset sufficient stats object.

        Returns
        -------
        SS : SuffStatBag
            represents whole dataset seen thus far.
        '''
        ElapsedTimeLogger.startEvent('global', 'increment')
        if SS is None:
            SS = SSbatch.copy()
        else:
            if oldSSbatch is not None:
                SS -= oldSSbatch
            SS += SSbatch
            if hasattr(SSbatch, 'propXSS'):
                if not hasattr(SS, 'propXSS'):
                    SS.propXSS = dict()

                for uid in SSbatch.propXSS:
                    if uid in SS.propXSS:
                        SS.propXSS[uid] += SSbatch.propXSS[uid]
                    else:
                        SS.propXSS[uid] = SSbatch.propXSS[uid].copy()
        # Force aggregated suff stats to obey required constraints.
        # This avoids numerical issues caused by incremental updates
        if hmodel is not None:
            if hasattr(hmodel.allocModel, 'forceSSInBounds'):
                hmodel.allocModel.forceSSInBounds(SS)
            if hasattr(hmodel.obsModel, 'forceSSInBounds'):
                hmodel.obsModel.forceSSInBounds(SS)
        ElapsedTimeLogger.stopEvent('global', 'increment')
        return SS

    def loadBatchAndFastForward(self, batchID, lapFrac, MoveLog, doCopy=0):
        ''' Retrieve batch from memory, and apply any relevant moves to it.

        Returns
        -------
        oldSSbatch : SuffStatBag, or None if specified batch not in memory.

        Post Condition
        --------------
        LastUpdateLap attribute will indicate batchID was updated at lapFrac,
        unless working with a copy not raw memory (doCopy=1).
        '''
        ElapsedTimeLogger.startEvent('global', 'fastfwdSS')
        try:
            SSbatch = self.SSmemory[batchID]
        except KeyError:
            return None

        if doCopy:
            SSbatch = SSbatch.copy()

        for (lap, op, kwargs, beforeUIDs, afterUIDs) in MoveLog:
            if lap < self.LastUpdateLap[batchID]:
                continue
            assert np.allclose(SSbatch.uids, beforeUIDs)
            if op == 'merge':
                SSbatch.mergeComps(**kwargs)
            elif op == 'shuffle':
                SSbatch.reorderComps(kwargs['bigtosmallorder'])
            elif op == 'prune':
                for uid in kwargs['emptyCompUIDs']:
                    SSbatch.removeComp(uid=uid)
            elif op == 'birth':
                targetUID = kwargs['targetUID']
                hasStoredProposal = hasattr(SSbatch, 'propXSS') and \
                    targetUID in SSbatch.propXSS
                if hasStoredProposal:
                    cur_newUIDs = SSbatch.propXSS[targetUID].uids
                    expected_newUIDs = np.setdiff1d(afterUIDs, beforeUIDs)
                    sameSize = cur_newUIDs.size == expected_newUIDs.size
                    if sameSize and np.all(cur_newUIDs == expected_newUIDs):
                        SSbatch.transferMassFromExistingToExpansion(
                           uid=targetUID, xSS=SSbatch.propXSS[targetUID])
                    else:
                        hasStoredProposal = False

                if not hasStoredProposal:
                    Kfresh = afterUIDs.size - beforeUIDs.size
                    SSbatch.insertEmptyComps(Kfresh)
                    SSbatch.setUIDs(afterUIDs)
            elif op == 'delete':
                SSbatch.removeMergeTerms()
                targetUID = kwargs['targetUID']
                hasStoredProposal = hasattr(SSbatch, 'propXSS') and \
                    targetUID in SSbatch.propXSS
                assert hasStoredProposal
                SSbatch.replaceCompsWithContraction(
                    removeUIDs=[targetUID],
                    replaceUIDs=SSbatch.propXSS[targetUID].uids,
                    replaceSS=SSbatch.propXSS[targetUID],
                    )
                '''
                SSbatch.replaceCompWithExpansion(
                    uid=targetUID, xSS=SSbatch.propXSS[targetUID])
                for (uidA, uidB) in SSbatch.mUIDPairs:
                    SSbatch.mergeComps(uidA=uidA, uidB=uidB)
                '''
            else:
                raise NotImplementedError("TODO")
            assert np.allclose(SSbatch.uids, afterUIDs)
        # Discard merge terms, since all accepted merges have been incorporated
        SSbatch.removeMergeTerms()
        if not doCopy:
            self.LastUpdateLap[batchID] = lapFrac
        ElapsedTimeLogger.stopEvent('global', 'fastfwdSS')
        return SSbatch

    def globalStep(self, hmodel, SS, lapFrac):
        ''' Do global update, if appropriate at current lap.

        Post Condition
        ---------
        hmodel global parameters updated in place.
        '''
        doFullPass = self.algParams['doFullPassBeforeMstep']
        didUpdate = False
        if self.algParams['doFullPassBeforeMstep'] == 1:
            if lapFrac >= 1.0:
                hmodel.update_global_params(SS, doLogElapsedTime=True)
                didUpdate = True
        elif doFullPass > 1.0:
            if lapFrac >= 1.0 or (doFullPass < SS.nDoc):
                # update if we've seen specified num of docs, not before
                hmodel.update_global_params(SS, doLogElapsedTime=True)
                didUpdate = True
        else:
            hmodel.update_global_params(SS, doLogElapsedTime=True)
            didUpdate = True
        return hmodel, didUpdate

    def makeMovePlans(self, hmodel, SS,
                      MovePlans=dict(),
                      MoveRecordsByUID=dict(), 
                      lapFrac=-1,
                      **kwargs):
        ''' Plan which comps to target for each possible move.

        Returns
        -------
        MovePlans : dict
        '''
        isFirst = self.isFirstBatch(lapFrac)
        if isFirst:
            MovePlans = dict()
        if isFirst and self.hasMove('birth'):
           ElapsedTimeLogger.startEvent('birth', 'plan')
           MovePlans = self.makeMovePlans_Birth_AtLapStart(
               hmodel, SS, 
               lapFrac=lapFrac,
               MovePlans=MovePlans,
               MoveRecordsByUID=MoveRecordsByUID,
               **kwargs)
           ElapsedTimeLogger.stopEvent('birth', 'plan')
        if isFirst and self.hasMove('merge'):
            ElapsedTimeLogger.startEvent('merge', 'plan')
            MovePlans = self.makeMovePlans_Merge(
                hmodel, SS, 
                lapFrac=lapFrac,
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                **kwargs)
            ElapsedTimeLogger.stopEvent('merge', 'plan')
        if isFirst and self.hasMove('delete'):
            ElapsedTimeLogger.startEvent('delete', 'plan')
            MovePlans = self.makeMovePlans_Delete(
                hmodel, SS, 
                lapFrac=lapFrac,
                MovePlans=MovePlans,
                MoveRecordsByUID=MoveRecordsByUID,
                **kwargs)
            ElapsedTimeLogger.stopEvent('delete', 'plan')
        return MovePlans

    def makeMovePlans_Merge(self, hmodel, SS,
                            MovePlans=dict(),
                            MoveRecordsByUID=dict(),
                            lapFrac=0,
                            **kwargs):
        ''' Plan out which merges to attempt in current lap.

        Returns
        -------
        MovePlans : dict
            * m_UIDPairs : list of pairs of uids to merge
        '''
        ceilLap = np.ceil(lapFrac)
        if SS is None:
            msg = "MERGE @ lap %.2f: Disabled." + \
                " Cannot plan merge on first lap." + \
                " Need valid SS that represent whole dataset."
            MLogger.pprint(msg % (ceilLap), 'info')
            return MovePlans

        startLap = self.algParams['merge']['m_startLap']
        if np.ceil(lapFrac) < startLap:
            msg = "MERGE @ lap %.2f: Disabled." + \
                " Waiting for lap >= %d (--m_startLap)."
            MLogger.pprint(msg % (ceilLap, startLap), 'info')
            return MovePlans
        stopLap = self.algParams['merge']['m_stopLap']
        if stopLap > 0 and np.ceil(lapFrac) >= stopLap:
            msg = "MERGE @ lap %.2f: Disabled." + \
                " Beyond lap %d (--m_stopLap)."
            MLogger.pprint(msg % (ceilLap, stopLap), 'info')
            return MovePlans

        MArgs = self.algParams['merge']
        MPlan = selectCandidateMergePairs(
            hmodel, SS,
            MovePlans=MovePlans,
            MoveRecordsByUID=MoveRecordsByUID,
            lapFrac=lapFrac,
            **MArgs)
        # Do not track m_UIDPairs field unless it is non-empty
        if len(MPlan['m_UIDPairs']) < 1:
            del MPlan['m_UIDPairs']
            del MPlan['mPairIDs']
            msg = "MERGE @ lap %.2f: No promising candidates, so no attempts."
            MLogger.pprint(msg % (ceilLap), 'info')
        else:
            MPlan['doPrecompMergeEntropy'] = 1
        MovePlans.update(MPlan)
        return MovePlans

    def makeMovePlans_Delete(self, hmodel, SS,
                            MovePlans=dict(),
                            MoveRecordsByUID=dict(),
                            lapFrac=0,
                            **kwargs):
        ''' Plan out which deletes to attempt in current lap.

        Returns
        -------
        MovePlans : dict
            * d_targetUIDs : list of uids to delete
        '''
        ceilLap = np.ceil(lapFrac)
        if SS is None:
            msg = "DELETE @ lap %.2f: Disabled." + \
                " Cannot delete before first complete lap," + \
                " because SS that represents whole dataset is required."
            DLogger.pprint(msg % (ceilLap), 'info')
            return MovePlans

        startLap = self.algParams['delete']['d_startLap']
        if ceilLap < startLap:
            msg = "DELETE @ lap %.2f: Disabled." + \
                " Waiting for lap >= %d (--d_startLap)."
            DLogger.pprint(msg % (ceilLap, startLap), 'info')
            return MovePlans
        stopLap = self.algParams['delete']['d_stopLap']
        if stopLap > 0 and ceilLap >= stopLap:
            msg = "DELETE @ lap %.2f: Disabled." + \
                " Beyond lap %d (--d_stopLap)."
            DLogger.pprint(msg % (ceilLap, stopLap), 'info')
            return MovePlans

        if self.hasMove('birth'):
            BArgs = self.algParams['birth']
        else:
            BArgs = dict()
        DArgs = self.algParams['delete']
        DArgs.update(BArgs)
        DPlan = selectCandidateDeleteComps(
            hmodel, SS,
            MovePlans=MovePlans,
            MoveRecordsByUID=MoveRecordsByUID,
            lapFrac=lapFrac,
            **DArgs)
        if 'failMsg' in DPlan:
            DLogger.pprint(
                'DELETE @ lap %.2f: %s' % (ceilLap, DPlan['failMsg']),
                'info')
        else:
            MovePlans.update(DPlan)
        return MovePlans

    def makeMovePlans_Birth_AtLapStart(
            self, hmodel, SS,
            MovePlans=dict(),
            MoveRecordsByUID=dict(),
            lapFrac=-2,
            batchID=-1,
            **kwargs):
        ''' Select comps to target with birth at start of current lap.

        Returns
        -------
        MovePlans : dict
            * b_shortlistUIDs : list of uids (ints) off limits to other moves.
        '''
        ceilLap = np.ceil(lapFrac)
        startLap = self.algParams['birth']['b_startLap']
        stopLap = self.algParams['birth']['b_stopLap']

        assert self.isFirstBatch(lapFrac)

        if ceilLap < startLap:
            msg = "BIRTH @ lap %.2f: Disabled." + \
                " Waiting for lap >= %d (--b_startLap)."
            MovePlans['b_statusMsg'] = msg % (ceilLap, startLap)
            BLogger.pprint(MovePlans['b_statusMsg'], 'info')
            return MovePlans
        if stopLap > 0 and ceilLap >= stopLap:
            msg = "BIRTH @ lap %.2f: Disabled." + \
                " Beyond lap %d (--b_stopLap)."
            MovePlans['b_statusMsg'] = msg % (ceilLap, stopLap)
            BLogger.pprint(MovePlans['b_statusMsg'], 'info')
            return MovePlans

        BArgs = self.algParams['birth']
        if BArgs['b_useShortList']:
            msg = "PLANNING birth shortlist at lap %.3f"
            BLogger.pprint(msg % (lapFrac))
            MovePlans = selectShortListForBirthAtLapStart(
                hmodel, SS,
                MoveRecordsByUID=MoveRecordsByUID,
                MovePlans=MovePlans,
                lapFrac=lapFrac,
                **BArgs)
        else:
            MovePlans['b_shortlistUIDs'] = list()
        assert 'b_shortlistUIDs' in MovePlans
        assert isinstance(MovePlans['b_shortlistUIDs'], list)
        return MovePlans


    def makeMovePlans_Birth_AtBatch(
            self, hmodel, SS,
            SSbatch=None,
            MovePlans=dict(),
            MoveRecordsByUID=dict(),
            lapFrac=-2,
            batchID=0,
            batchPos=0,
            nBatch=0,
            isFirstBatch=False,
            **kwargs):
        ''' Select comps to target with birth at current batch.

        Returns
        -------
        MovePlans : dict
            * b_targetUIDs : list of uids (ints) indicating comps to target
        '''
        ceilLap = np.ceil(lapFrac)
        startLap = self.algParams['birth']['b_startLap']
        stopLap = self.algParams['birth']['b_stopLap']

        if ceilLap < startLap:
            return MovePlans
        if stopLap > 0 and ceilLap >= stopLap:
            return MovePlans

        if self.hasMove('birth'):
            BArgs = self.algParams['birth']    
            msg = "PLANNING birth at lap %.3f batch %d"
            BLogger.pprint(msg % (lapFrac, batchID))
            MovePlans = selectCompsForBirthAtCurrentBatch(
                hmodel, SS,
                SSbatch=SSbatch,
                MoveRecordsByUID=MoveRecordsByUID,
                MovePlans=MovePlans,
                lapFrac=lapFrac,
                batchID=batchID,
                nBatch=nBatch,
                batchPos=batchPos,
                isFirstBatch=isFirstBatch,
                **BArgs)
            if 'b_targetUIDs' in MovePlans:
                assert isinstance(MovePlans['b_targetUIDs'], list)
        return MovePlans

    def runMoves_Birth(self, hmodel, SS, loss, MovePlans,
                       MoveLog=list(),
                       MoveRecordsByUID=dict(),
                       lapFrac=0,
                       **kwargs):
        ''' Execute planned birth/split moves.

        Returns
        -------
        hmodel
        SS
        loss
        MoveLog
        MoveRecordsByUID
        '''
        ElapsedTimeLogger.startEvent('birth', 'eval')
        if 'b_targetUIDs' in MovePlans and len(MovePlans['b_targetUIDs']) > 0:
            b_targetUIDs = [u for u in MovePlans['b_targetUIDs']]
            BLogger.pprint(
                'EVALUATING birth proposals at lap %.2f' % (lapFrac))
            MovePlans['b_retainedUIDs'] = list()
        else:
            b_targetUIDs = list()

        if 'b_nFailedEval' in MovePlans:
            nFailedEval = MovePlans['b_nFailedEval']
        else:
            nFailedEval = 0
        if 'b_nAccept' in MovePlans:
            nAccept = MovePlans['b_nAccept']
        else:
            nAccept = 0
        if 'b_nTrial' in MovePlans:
            nTrial = MovePlans['b_nTrial']
        else:
            nTrial = 0
        if 'b_Knew' in MovePlans:
            totalKnew = MovePlans['b_Knew']
        else:
            totalKnew = 0
        nRetainedForNextLap = 0
        acceptedUIDs = list()
        curLdict = hmodel.calc_evidence(SS=SS, todict=1)
        for targetUID in b_targetUIDs:
            # Skip delete proposals, which are handled differently
            if 'd_targetUIDs' in MovePlans:
                if targetUID in MovePlans['d_targetUIDs']:
                    raise ValueError("WHOA! Cannot delete and birth same uid.")
            nTrial += 1

            BLogger.startUIDSpecificLog(targetUID)
            # Prepare record-keeping            
            if targetUID not in MoveRecordsByUID:
                MoveRecordsByUID[targetUID] = defaultdict(int)
            ktarget = SS.uid2k(targetUID)
            targetCount = SS.getCountVec()[ktarget]
            MoveRecordsByUID[targetUID]['b_nTrial'] += 1
            MoveRecordsByUID[targetUID]['b_latestLap'] = lapFrac

            # Construct proposal statistics
            propSS = SS.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=SS.propXSS[targetUID])
            # Create model via global step from proposed stats
            propModel = hmodel.copy()
            propModel.update_global_params(propSS)
            # Compute score of proposal
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)
            prop_loss = -1 * propLdict['Ltotal']
            # Decide accept or reject
            change_loss = prop_loss - loss
            if change_loss < 0:
                decision = 'ACCEPT'
                Knew_str = ' Knew %4d' % (propSS.K - SS.K)
            else:
                decision = 'REJECT'
                Knew_str = ''
            tUIDstr = "%15s" % ("targetUID %d" % (targetUID))
            decisionMsg = 'Eval %s at lap %.3f lapCeil %d | ' % (
                tUIDstr, lapFrac, np.ceil(lapFrac))
            decisionMsg += \
                decision + " change_loss % .3e" % (change_loss) + Knew_str
            BLogger.pprint(decisionMsg)
            # Record some details about final score
            msg = "   gainL % .3e" % (change_loss)
            msg += "\n    curL % .3e" % (loss)
            msg += "\n   propL % .3e" % (prop_loss)
            for key in sorted(curLdict.keys()):
                if key.count('_') or key.count('total'):
                    continue
                msg += "\n   gain_%8s % .3e" % (
                    key, propLdict[key] - curLdict[key])
            BLogger.pprint(msg)
            assert propLdict['Lentropy'] >= - 1e-6
            assert curLdict['Lentropy'] >= - 1e-6
            assert propLdict['Lentropy'] >= curLdict['Lentropy'] - 1e-6
            if prop_loss < loss:
                # Handle ACCEPTED case
                nAccept += 1
                BLogger.pprint(
                    '   Accepted. Jump up to loss % .3e ' % (prop_loss))
                BLogger.pprint(
                    "    Mass transfered to new comps: %.2f" % (
                        SS.getCountVec()[ktarget] - \
                            propSS.getCountVec()[ktarget]))
                BLogger.pprint(
                    "    Remaining mass at targetUID %d: %.2f" % (
                        targetUID, propSS.getCountVec()[ktarget]))
                totalKnew += propSS.K - SS.K
                MoveRecordsByUID[targetUID]['b_nSuccess'] += 1
                MoveRecordsByUID[targetUID]['b_nFailRecent'] = 0
                MoveRecordsByUID[targetUID]['b_nSuccessRecent'] += 1
                MoveRecordsByUID[targetUID]['b_latestLapAccept'] = lapFrac
                # Write necessary information to the log
                MoveArgs = dict(
                    targetUID=targetUID,
                    changedUIDs=np.asarray([targetUID]),
                    newUIDs=SS.propXSS[targetUID].uids)
                infoTuple = (
                    lapFrac, 'birth', MoveArgs,
                    SS.uids.copy(), propSS.uids.copy())
                MoveLog.append(infoTuple)
                # Set proposal values as new "current" values
                hmodel = propModel
                loss = prop_loss
                SS = propSS
                curLdict = propLdict
                MovePlans['b_targetUIDs'].remove(targetUID)
                del SS.propXSS[targetUID]
            else:
                # Rejected.
                BLogger.pprint(
                    '   Rejected. Remain at loss %.3e' % (loss))
                gainLdata = propLdict['Ldata'] - curLdict['Ldata']
                # Decide if worth pursuing in future batches, if necessary.
                subsetCountVec = SS.propXSS[targetUID].getCountVec()
                nSubset = subsetCountVec.sum()
                nTotal = SS.getCountVec()[ktarget]

                BKwargs = self.algParams['birth']
                doTryRetain = BKwargs['b_retainAcrossBatchesByLdata']
                if lapFrac > 1.0 and not self.isLastBatch(lapFrac):
                    doAlwaysRetain = \
                        BKwargs['b_retainAcrossBatchesAfterFirstLap']
                else:
                    doAlwaysRetain = False

                keepThr = BKwargs['b_minNumAtomsForRetainComp']
                hasTwoLargeOnes = np.sum(subsetCountVec >= keepThr) >= 2
                hasNotUsedMostData = nSubset < 0.75 * nTotal
                if hasTwoLargeOnes and hasNotUsedMostData and self.nBatch > 1:
                    couldUseMoreData = True
                else:
                    couldUseMoreData = False

                if doTryRetain and couldUseMoreData:
                    # If Ldata for subset of data reassigned so far looks good
                    # we hold onto this proposal for next time! 
                    propSSsubset = SS.propXSS[targetUID].copy(
                        includeELBOTerms=False, includeMergeTerms=False)
                    tmpModel = propModel
                    tmpModel.obsModel.update_global_params(propSSsubset)
                    propLdata_subset = tmpModel.obsModel.calcELBO_Memoized(
                        propSSsubset)
                    # Create current representation
                    curSSsubset = propSSsubset
                    while curSSsubset.K > 1:
                        curSSsubset.mergeComps(0, 1)
                    tmpModel.obsModel.update_global_params(curSSsubset)
                    curLdata_subset = tmpModel.obsModel.calcELBO_Memoized(
                        curSSsubset)
                    gainLdata_subset = propLdata_subset - curLdata_subset
                else:
                    gainLdata_subset = -42.0

                if doAlwaysRetain:
                    nTrial -= 1
                    BLogger.pprint(
                        '   Retained. Trying proposal across whole dataset.')
                    assert targetUID in SS.propXSS
                    MovePlans['b_retainedUIDs'].append(targetUID)
                elif doTryRetain and gainLdata_subset > 1e-6 and \
                        not self.isLastBatch(lapFrac):
                    nTrial -= 1
                    BLogger.pprint(
                        '   Retained. Promising gainLdata_subset % .2f' % (
                            gainLdata_subset))
                    assert targetUID in SS.propXSS
                    MovePlans['b_retainedUIDs'].append(targetUID)

                elif doTryRetain and gainLdata > 1e-6 and \
                        not self.isLastBatch(lapFrac):
                    nTrial -= 1
                    BLogger.pprint(
                        '   Retained. Promising value of gainLdata % .2f' % (
                            gainLdata))
                    assert targetUID in SS.propXSS
                    MovePlans['b_retainedUIDs'].append(targetUID)
                elif doTryRetain and gainLdata_subset > 1e-6 and \
                        self.isLastBatch(lapFrac) and couldUseMoreData:
                    nRetainedForNextLap += 1
                    BLogger.pprint(
                        '   Retain uid %d next lap! gainLdata_subset %.3e' % (
                            targetUID, gainLdata_subset))
                    assert targetUID in SS.propXSS
                    MoveRecordsByUID[targetUID]['b_tryAgainFutureLap'] = 1
                    MovePlans['b_retainedUIDs'].append(targetUID)
                else:
                    nFailedEval += 1
                    MovePlans['b_targetUIDs'].remove(targetUID)
                    MoveRecordsByUID[targetUID]['b_nFail'] += 1
                    MoveRecordsByUID[targetUID]['b_nFailRecent'] += 1
                    MoveRecordsByUID[targetUID]['b_nSuccessRecent'] = 0
                    MoveRecordsByUID[targetUID]['b_tryAgainFutureLap'] = 0
                    # Update batch-specific records for this uid
                    uidRec = MoveRecordsByUID[targetUID]
                    uidRec_b = uidRec['byBatch'][uidRec['b_proposalBatchID']]
                    uidRec_b['nFail'] += 1            
                    uidRec_b['nEval'] += 1
                    uidRec_b['proposalTotalSize'] = \
                        SS.propXSS[targetUID].getCountVec().sum()
                    del SS.propXSS[targetUID]

            BLogger.pprint('')
            BLogger.stopUIDSpecificLog(targetUID)

        if 'b_retainedUIDs' in MovePlans:
            assert np.allclose(MovePlans['b_retainedUIDs'],
                MovePlans['b_targetUIDs'])
            for uid in MovePlans['b_targetUIDs']:
                assert uid in SS.propXSS
        MovePlans['b_Knew'] = totalKnew
        MovePlans['b_nAccept'] = nAccept
        MovePlans['b_nTrial'] = nTrial
        MovePlans['b_nFailedEval'] = nFailedEval
        if self.isLastBatch(lapFrac) and 'b_statusMsg' not in MovePlans:
            usedNonEmptyShortList = \
                self.algParams['birth']['b_useShortList'] \
                    and len(MovePlans['b_shortlistUIDs']) > 0

            if nTrial > 0:
                msg = "BIRTH @ lap %.2f : Added %d states." + \
                    " %d/%d succeeded. %d/%d failed eval phase. " + \
                    "%d/%d failed build phase."
                msg = msg % (
                    lapFrac, totalKnew, 
                    nAccept, nTrial,
                    MovePlans['b_nFailedEval'], nTrial,
                    MovePlans['b_nFailedProp'], nTrial)
                if nRetainedForNextLap > 0:
                    msg += " %d retained!" % (nRetainedForNextLap)
                BLogger.pprint(msg, 'info')
            elif usedNonEmptyShortList:
                # Birth was eligible, but did not make it to eval stage.
                msg = "BIRTH @ lap %.3f : None attempted." + \
                    " Shortlist had %d possible clusters," + \
                    " but none met minimum requirements."
                msg = msg % (
                    lapFrac, len(MovePlans['b_shortlistUIDs']))
                BLogger.pprint(msg, 'info')
            else:
                msg = "BIRTH @ lap %.3f : None attempted."
                msg +=  " %d past failures. %d too small. %d too busy."
                msg = msg % (
                    lapFrac,
                    MovePlans['b_nDQ_pastfail'],
                    MovePlans['b_nDQ_toosmall'],
                    MovePlans['b_nDQ_toobusy'],
                    )
                BLogger.pprint(msg, 'info')

            # If any short-listed uids did not get tried in this lap
            # there are two possible reasons:
            # 1) No batch contains a sufficient size of that uid.
            # 2) Other uids were prioritized due to budget constraints.
            # We need to mark uids that failed for reason 1,
            # so that we don't avoid deleting/merging them in the future.
            if usedNonEmptyShortList:
                for uid in MovePlans['b_shortlistUIDs']:
                    if uid not in MoveRecordsByUID:
                        MoveRecordsByUID[uid] = defaultdict(int)
                    Rec = MoveRecordsByUID[uid]

                    lastEligibleLap = Rec['b_latestEligibleLap']
                    if np.ceil(lastEligibleLap) < np.ceil(lapFrac):
                        msg = "Marked uid %d ineligible for future shortlists."
                        msg += " It was never eligible this lap."
                        BLogger.pprint(msg % (uid))
                        k = SS.uid2k(uid)
                        Rec['b_latestLap'] = lapFrac
                        Rec['b_nFail'] += 1
                        Rec['b_nFailRecent'] += 1
                        Rec['b_nSuccessRecent'] = 0

        ElapsedTimeLogger.stopEvent('birth', 'eval')
        return hmodel, SS, loss, MoveLog, MoveRecordsByUID

    def runMoves_Merge(self, hmodel, SS, loss, MovePlans,
                       MoveLog=list(),
                       MoveRecordsByUID=dict(),
                       lapFrac=0,
                       **kwargs):
        ''' Execute planned merge moves.

        Returns
        -------
        hmodel
        SS : SuffStatBag
            Contains updated fields and ELBO terms for K-Kaccepted comps.
            All merge terms will be set to zero.
        loss
        MoveLog
        MoveRecordsByUID
        '''
        ElapsedTimeLogger.startEvent('merge', 'eval')
        acceptedUIDs = set()
        nTrial = 0
        nAccept = 0
        nSkip = 0
        Ndiff = 0.0
        MLogger.pprint("EVALUATING merges at lap %.2f" % (
            lapFrac), 'debug')
        for ii, (uidA, uidB) in enumerate(MovePlans['m_UIDPairs']):
            # Skip uids that we have already accepted in a previous merge.
            if uidA in acceptedUIDs or uidB in acceptedUIDs:
                nSkip += 1
                MLogger.pprint("%4d, %4d : skipped." % (
                    uidA, uidB), 'debug')
                continue
            uid_already_edited = False
            for log_tuple in MoveLog[::-1]:
                lap, name, move_args, orig_uids, prop_uids = log_tuple
                if lap != lapFrac:
                    break
                uid_already_edited = uid_already_edited or (
                    uidA in move_args['changedUIDs'] or
                    uidB in move_args['changedUIDs'])
                MLogger.pprint('Skip eval of uids %d,%d at lap %.2f' % (
                    uidA, uidB, lapFrac))
                if uid_already_edited:
                    break
            if uid_already_edited:
                continue


            nTrial += 1
            # Update records for when each uid was last attempted
            pairTuple = (uidA, uidB)
            if pairTuple not in MoveRecordsByUID:
                MoveRecordsByUID[pairTuple] = defaultdict(int)
            MoveRecordsByUID[pairTuple]['m_nTrial'] += 1
            MoveRecordsByUID[pairTuple]['m_latestLap'] = lapFrac
            minPairCount = np.minimum(
                SS.getCountForUID(uidA),
                SS.getCountForUID(uidB))
            MoveRecordsByUID[pairTuple]['m_latestMinCount'] = minPairCount
            propSS = SS.copy()
            propSS.mergeComps(uidA=uidA, uidB=uidB)
            propModel = hmodel.copy()
            propModel.update_global_params(propSS)
            proploss = -1 * propModel.calc_evidence(SS=propSS)
            assert np.isfinite(proploss)

            propSizeStr = count2str(propSS.getCountForUID(uidA))
            if proploss < loss + ELBO_GAP_ACCEPT_TOL:
                nAccept += 1
                Ndiff += minPairCount
                MLogger.pprint(
                    "%4d, %4d : accepted." % (uidA, uidB) +
                    " gain %.3e  " % (proploss - loss) +
                    " size %s  " % (propSizeStr),
                    'debug')

                acceptedUIDs.add(uidA)
                acceptedUIDs.add(uidB)
                if uidA not in MoveRecordsByUID:
                    MoveRecordsByUID[uidA] = defaultdict(int)
                MoveRecordsByUID[uidA]['m_latestLapAccept'] = lapFrac
                del MoveRecordsByUID[pairTuple]

                # Write necessary information to the log
                MoveArgs = dict(
                    uidA=uidA, uidB=uidB,
                    changedUIDs=np.hstack([uidA, uidB]))
                infoTuple = (lapFrac, 'merge', MoveArgs,
                             SS.uids.copy(), propSS.uids.copy())
                MoveLog.append(infoTuple)
                # Set proposal values as new "current" values
                SS = propSS
                hmodel = propModel
                loss = proploss
            else:
                MLogger.pprint(
                    "%4d, %4d : rejected." % (uidA, uidB) +
                    " gain %.3f  " % (proploss - loss) +
                    " size %s  " % (propSizeStr),
                    'debug')
                MoveRecordsByUID[pairTuple]['m_nFailRecent'] += 1

        if nTrial > 0:
            msg = "MERGE @ lap %.2f : %d/%d accepted." + \
                " Ndiff %.2f. %d skipped."
            msg = msg % (
                lapFrac, nAccept, nTrial, Ndiff, nSkip)
            MLogger.pprint(msg, 'info')
        # Finally, set all merge fields to zero,
        # since all possible merges have been accepted
        SS.removeMergeTerms()
        assert not hasattr(SS, 'M')
        ElapsedTimeLogger.stopEvent('merge', 'eval')
        return hmodel, SS, loss, MoveLog, MoveRecordsByUID

    def runMoves_Shuffle(self, hmodel, SS, loss, MovePlans,
                         MoveLog=list(),
                         MoveRecordsByUID=dict(),
                         lapFrac=0,
                         **kwargs):
        ''' Execute shuffle move, which need not be planned in advance.

        Returns
        -------
        hmodel
            Reordered copies of the K input states.
        SS : SuffStatBag
            Reordered copies of the K input states.
        loss
        MoveLog
        MoveRecordsByUID
        '''
        prev_loss = loss
        emptyCompLocs = np.flatnonzero(SS.getCountVec() < 0.001)
        emptyCompUIDs = [SS.uids[k] for k in emptyCompLocs]
        if emptyCompLocs.size > 0 and self.algParams['shuffle']['s_doPrune']:
            beforeUIDs = SS.uids.copy()
            for uid in emptyCompUIDs:
                SS.removeComp(uid=uid)
            afterUIDs = SS.uids.copy()
            moveTuple = (
                lapFrac, 'prune',
                dict(emptyCompUIDs=emptyCompUIDs),
                beforeUIDs,
                afterUIDs)
            MoveLog.append(moveTuple)

        if hasattr(SS, 'sumLogPiRemVec'):
            limits = np.flatnonzero(SS.sumLogPiRemVec) + 1
            assert limits.size > 0
            bigtosmallorder = argsort_bigtosmall_stable(
                SS.sumLogPi, limits=limits)
        else:
            bigtosmallorder = argsort_bigtosmall_stable(SS.getCountVec())
        sortedalready = np.arange(SS.K)
        if not np.allclose(bigtosmallorder, sortedalready):
            moveTuple = (
                lapFrac, 'shuffle',
                dict(bigtosmallorder=bigtosmallorder),
                SS.uids, SS.uids[bigtosmallorder])
            MoveLog.append(moveTuple)
            SS.reorderComps(bigtosmallorder)
            hmodel.update_global_params(SS, sortorder=bigtosmallorder)
            loss = -1 * hmodel.calc_evidence(SS=SS)
            # TODO Prevent shuffle if ELBO does not improve??
            SLogger.pprint(
                "SHUFFLED at lap %.3f." % (lapFrac) + \
                " diff % .4e   prev_loss % .4e   new_loss % .4e" % (
                    loss - prev_loss, prev_loss, loss))

        elif emptyCompLocs.size > 0 and self.algParams['shuffle']['s_doPrune']:
            hmodel.update_global_params(SS)
            loss = -1 * hmodel.calc_evidence(SS=SS)

        return hmodel, SS, loss, MoveLog, MoveRecordsByUID


    def runMoves_Delete(self, hmodel, SS, loss, MovePlans,
                        MoveLog=list(),
                        MoveRecordsByUID=dict(),
                        lapFrac=0,
                        **kwargs):
        ''' Execute planned delete move.

        Returns
        -------
        hmodel
        SS
        loss
        MoveLog
        MoveRecordsByUID
        '''
        ElapsedTimeLogger.startEvent('delete', 'eval')

        if len(MovePlans['d_targetUIDs']) > 0:
            DLogger.pprint('EVALUATING delete @ lap %.2f' % (lapFrac))

        nAccept = 0
        nTrial = 0
        Ndiff = 0.0
        curLdict = hmodel.calc_evidence(SS=SS, todict=1)
        for targetUID in MovePlans['d_targetUIDs']:
            nTrial += 1
            assert targetUID in SS.propXSS
            uid_already_edited = False
            for log_tuple in MoveLog[::-1]:
                lap, name, move_args, orig_uids, prop_uids = log_tuple
                if lap != lapFrac:
                    break
                if name == 'merge':
                    uid_already_edited = uid_already_edited or (
                        targetUID == move_args['uidA'] or
                        targetUID == move_args['uidB'])
                DLogger.pprint('Skip eval of targetUID %d at lap %.2f' % (
                    targetUID, lapFrac))
                if uid_already_edited:
                    break

            if uid_already_edited:
                continue

            # Prepare record keeping
            if targetUID not in MoveRecordsByUID:
                MoveRecordsByUID[targetUID] = defaultdict(int)
            targetCount = SS.getCountVec()[SS.uid2k(targetUID)]
            MoveRecordsByUID[targetUID]['d_nTrial'] += 1
            MoveRecordsByUID[targetUID]['d_latestLap'] = lapFrac
            MoveRecordsByUID[targetUID]['d_latestCount'] = targetCount
            # Construct proposed stats
            propSS = SS.copy()
            replaceUIDs = MovePlans['d_absorbingUIDs']
            propSS.replaceCompsWithContraction(
                replaceUIDs=replaceUIDs,
                removeUIDs=[targetUID],
                replaceSS=SS.propXSS[targetUID])
            # Construct proposed model and its ELBO score
            propModel = hmodel.copy()
            propModel.update_global_params(propSS)
            propLdict = propModel.calc_evidence(SS=propSS, todict=1)
            proploss = -1 * propLdict['Ltotal']
            msg = 'targetUID %d' % (targetUID)
            msg += '\n   gain_loss % .3e' % (proploss-loss)
            msg += "\n    cur_loss % .3e" % (loss)
            msg += "\n   prop_loss % .3e" % (proploss)
            for key in sorted(curLdict.keys()):
                if key.count('_') or key.count('total'):
                    continue
                msg += "\n   gain_%8s % .3e" % (
                    key, propLdict[key] - curLdict[key])

            DLogger.pprint(msg)
            # Make decision
            if proploss < loss + ELBO_GAP_ACCEPT_TOL:
                # Accept
                nAccept += 1
                Ndiff += targetCount
                MoveRecordsByUID[targetUID]['d_nFailRecent'] = 0
                MoveRecordsByUID[targetUID]['d_latestLapAccept'] = lapFrac
                # Write necessary information to the log
                MoveArgs = dict(
                    targetUID=targetUID,
                    changedUIDs=np.hstack([targetUID, replaceUIDs]))
                infoTuple = (lapFrac, 'delete', MoveArgs,
                             SS.uids.copy(), propSS.uids.copy())
                MoveLog.append(infoTuple)
                # Set proposal values as new "current" values
                hmodel = propModel
                loss = proploss
                SS = propSS
                curLdict = propLdict
            else:
                # Reject!
                MoveRecordsByUID[targetUID]['d_nFail'] += 1
                MoveRecordsByUID[targetUID]['d_nFailRecent'] += 1
            # Always cleanup evidence of the proposal
            del SS.propXSS[targetUID]

        if nTrial > 0:
            msg = 'DELETE @ lap %.2f: %d/%d accepted. Ndiff %.2f.' % (
                lapFrac, nAccept, nTrial, Ndiff)
            DLogger.pprint(msg, 'info')
        # Discard plans, because they have come to fruition.
        for key in MovePlans.keys():
            if key.startswith('d_'):
                del MovePlans[key]
        ElapsedTimeLogger.stopEvent('delete', 'eval')
        return hmodel, SS, loss, MoveLog, MoveRecordsByUID

    def initProgressTrackVars(self, DataIterator):
        ''' Initialize internal attributes tracking how many steps we've taken.

        Returns
        -------
        iterid : int
        lapFrac : float

        Post Condition
        --------------
        Creates attributes nBatch, lapFracInc
        '''
        # Define how much of data we see at each mini-batch
        nBatch = float(DataIterator.nBatch)
        self.nBatch = nBatch
        self.lapFracInc = 1.0 / nBatch

        # Set-up progress-tracking variables
        iterid = -1
        lapFrac = np.maximum(0, self.algParams['startLap'] - 1.0 / nBatch)
        if lapFrac > 0:
            # When restarting an existing run,
            #  need to start with last update for final batch from previous lap
            DataIterator.lapID = int(np.ceil(lapFrac)) - 1
            DataIterator.curLapPos = nBatch - 2
            iterid = int(nBatch * lapFrac) - 1
        return iterid, lapFrac

    def doDebug(self):
        debug = self.algParams['debug']
        return debug.count('q') or debug.count('on') or debug.count('interact')

    def doDebugVerbose(self):
        return self.doDebug() and self.algParams['debug'].count('q') == 0

    def hasMoreReasonableMoves(self, SS, MoveRecordsByUID, lapFrac, **kwargs):
        ''' Decide if more moves will feasibly change current configuration.

        Returns
        -------
        hasMovesLeft : boolean
            True means further iterations likely see births/merges accepted.
            False means all possible moves likely to be rejected.
        '''
        nLapsCompleted = lapFrac - self.algParams['startLap']
        if nLapsCompleted >= self.algParams['nLap']:
            # Time's up, so doesn't matter what other moves are possible.
            return False

        if self.hasMove('birth'):
            nStuck = self.algParams['birth']['b_nStuckBeforeQuit']
            startLap = self.algParams['birth']['b_startLap']
            stopLap = self.algParams['birth']['b_stopLap']
            if stopLap < 0:
                stopLap = np.inf

            if lapFrac > stopLap:
                hasMovesLeft_Birth = False
            elif startLap > self.algParams['nLap']:
                # Birth will never occur. User has effectively disabled it.
                hasMovesLeft_Birth = False
            elif (lapFrac > startLap + nStuck):
                # If tried for at least nStuck laps without accepting,
                # we consider the method exhausted and exit early.
                b_lapLastAcceptedVec = np.asarray(
                    [MoveRecordsByUID[u]['b_latestLapAccept']
                        for u in MoveRecordsByUID])
                if b_lapLastAcceptedVec.size == 0:
                    lapLastAccepted = 0
                else:
                    lapLastAccepted = np.max(b_lapLastAcceptedVec)
                if (lapFrac - lapLastAccepted) > nStuck:
                    hasMovesLeft_Birth = False
                else:
                    hasMovesLeft_Birth = True
            else:
                hasMovesLeft_Birth = True
        else:
            hasMovesLeft_Birth = False

        if self.hasMove('merge'):
            nStuck = self.algParams['merge']['m_nStuckBeforeQuit']
            startLap = self.algParams['merge']['m_startLap']
            stopLap = self.algParams['merge']['m_stopLap']
            if stopLap < 0:
                stopLap = np.inf
            if lapFrac > stopLap:
                hasMovesLeft_Merge = False
            elif startLap > self.algParams['nLap']:
                # Merge will never occur. User has effectively disabled it.
                hasMovesLeft_Merge = False
            elif (lapFrac > startLap + nStuck):
                # If tried for at least nStuck laps without accepting,
                # we consider the method exhausted and exit early.
                m_lapLastAcceptedVec = np.asarray(
                    [MoveRecordsByUID[u]['m_latestLapAccept']
                        for u in MoveRecordsByUID])
                if m_lapLastAcceptedVec.size == 0:
                    lapLastAccepted = 0
                else:
                    lapLastAccepted = np.max(m_lapLastAcceptedVec)
                if (lapFrac - lapLastAccepted) > nStuck:
                    hasMovesLeft_Merge = False
                else:
                    hasMovesLeft_Merge = True
            else:
                hasMovesLeft_Merge = True
        else:
            hasMovesLeft_Merge = False

        if self.hasMove('delete'):
            nStuck = self.algParams['delete']['d_nStuckBeforeQuit']
            startLap = self.algParams['delete']['d_startLap']
            stopLap = self.algParams['delete']['d_stopLap']
            if stopLap < 0:
                stopLap = np.inf
            if lapFrac > stopLap:
                hasMovesLeft_Delete = False
            elif startLap > self.algParams['nLap']:
                # Delete will never occur. User has effectively disabled it.
                hasMovesLeft_Delete = False
            elif lapFrac > startLap + nStuck:
                # If tried for at least nStuck laps without accepting,
                # we consider the method exhausted and exit early.
                d_lapLastAcceptedVec = np.asarray(
                    [MoveRecordsByUID[u]['d_latestLapAccept']
                        for u in MoveRecordsByUID])
                if d_lapLastAcceptedVec.size == 0:
                    lapLastAccepted = 0
                else:
                    lapLastAccepted = np.max(d_lapLastAcceptedVec)
                if (lapFrac - lapLastAccepted) > nStuck:
                    hasMovesLeft_Delete = False
                else:
                    hasMovesLeft_Delete = True
            else:
                hasMovesLeft_Delete = True
        else:
            hasMovesLeft_Delete = False
        return hasMovesLeft_Birth or hasMovesLeft_Merge or hasMovesLeft_Delete
        # ... end function hasMoreReasonableMoves


    def verifyELBOTracking(
            self, hmodel, SS,
            evBound=None, lapFrac=-1, MoveLog=None, **kwargs):
        ''' Verify current global SS consistent with batch-specific SS.
        '''
        if self.doDebugVerbose():
            self.print_msg(
                '>>>>>>>> BEGIN double-check @ lap %.2f' % (self.lapFrac))

        if evBound is None:
            evBound = hmodel.calc_evidence(SS=SS)

        for batchID in range(len(self.SSmemory.keys())):
            SSchunk = self.loadBatchAndFastForward(
                batchID, lapFrac=lapFrac, MoveLog=MoveLog, doCopy=1)
            if batchID == 0:
                SS2 = SSchunk.copy()
            else:
                SS2 += SSchunk
        evCheck = hmodel.calc_evidence(SS=SS2)

        if self.algParams['debug'].count('quiet') == 0:
            print '% 14.8f evBound from agg SS' % (evBound)
            print '% 14.8f evBound from sum over SSmemory' % (evCheck)
        if self.algParams['debug'].count('interactive'):
            isCorrect = np.allclose(SS.getCountVec(), SS2.getCountVec()) \
                and np.allclose(evBound, evCheck)
            if not isCorrect:
                from IPython import embed
                embed()
        else:
            assert np.allclose(SS.getCountVec(), SS2.getCountVec())
            assert np.allclose(evBound, evCheck)
        if self.doDebugVerbose():
            self.print_msg(
                '<<<<<<<< END   double-check @ lap %.2f' % (self.lapFrac))
