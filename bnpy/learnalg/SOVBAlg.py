'''
SOVBAlg.py

Implementation of stochastic online VB (soVB) for bnpy models
'''
from builtins import *
import os
import numpy as np
import scipy.sparse

from .LearnAlg import LearnAlg
from .LearnAlg import makeDictOfAllWorkspaceVars
from . import ElapsedTimeLogger
from bnpy.util.SparseRespUtil import sparsifyResp

class SOVBAlg(LearnAlg):

    def __init__(self, **kwargs):
        ''' Creates stochastic online learning algorithm,
            with fields rhodelay, rhoexp that define learning rate schedule.
        '''
        super(type(self), self).__init__(**kwargs)
        self.rhodelay = self.algParams['rhodelay']
        self.rhoexp = self.algParams['rhoexp']
        self.LPmemory = dict()

    def fit(self, hmodel, DataIterator, SS=None):
        ''' Run stochastic variational to fit hmodel parameters to Data.

        Returns
        --------
        Info : dict of run information.

        Post Condition
        --------
        hmodel updated in place with improved global parameters.
        '''
        self.set_start_time_now()
        LP = None
        rho = 1.0  # Learning rate
        nBatch = float(DataIterator.nBatch)

        # Set-up progress-tracking variables
        iterid = -1
        lapFrac = np.maximum(0, self.algParams['startLap'] - 1.0 / nBatch)
        if lapFrac > 0:
            # When restarting an existing run,
            #  need to start with last update for final batch from previous lap
            DataIterator.lapID = int(np.ceil(lapFrac)) - 1
            DataIterator.curLapPos = nBatch - 2
            iterid = int(nBatch * lapFrac) - 1

        # Save initial state
        self.saveParams(lapFrac, hmodel)

        # Custom func hook
        self.eval_custom_func(
            isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))
        ElapsedTimeLogger.writeToLogOnLapCompleted(lapFrac)

        if self.algParams['doMemoELBO']:
            SStotal = None
            SSPerBatch = dict()
        else:
            loss_running_sum = 0
            loss_per_batch = np.zeros(nBatch)
        while DataIterator.has_next_batch():

            # Grab new data
            Dchunk = DataIterator.get_next_batch()
            batchID = DataIterator.batchID
            Dchunk.batchID = batchID

            # Update progress-tracking variables
            iterid += 1
            lapFrac += 1.0 / nBatch
            self.lapFrac = lapFrac
            nLapsCompleted = lapFrac - self.algParams['startLap']
            self.set_random_seed_at_lap(lapFrac)

            # E step
            self.algParamsLP['batchID'] = batchID
            self.algParamsLP['lapFrac'] = lapFrac  # logging
            if batchID in self.LPmemory:
                batchLP = self.load_batch_local_params_from_memory(batchID)
            else:
                batchLP = None
            LP = hmodel.calc_local_params(Dchunk, batchLP,
                doLogElapsedTime=True,
                **self.algParamsLP)
            rho = (1 + iterid + self.rhodelay) ** (-1.0 * self.rhoexp)
            if self.algParams['doMemoELBO']:
                # SS step. Scale at size of current batch.
                SS = hmodel.get_global_suff_stats(Dchunk, LP,
                                                  doLogElapsedTime=True,
                                                  doPrecompEntropy=True)
                if self.algParams['doMemoizeLocalParams']:
                    self.save_batch_local_params_to_memory(
                        batchID, LP)
                # Incremental updates for whole-dataset stats
                # Must happen before applification.
                if batchID in SSPerBatch:
                    SStotal -= SSPerBatch[batchID]
                if SStotal is None:
                    SStotal = SS.copy()
                else:
                    SStotal += SS
                SSPerBatch[batchID] = SS.copy()

                # Scale up to size of whole dataset.
                if hasattr(Dchunk, 'nDoc'):
                    ampF = Dchunk.nDocTotal / float(Dchunk.nDoc)
                    SS.applyAmpFactor(ampF)
                else:
                    ampF = Dchunk.nObsTotal / float(Dchunk.nObs)
                    SS.applyAmpFactor(ampF)
                # M step with learning rate
                hmodel.update_global_params(SS, rho, doLogElapsedTime=True)
                # ELBO step
                assert not SStotal.hasAmpFactor()
                loss = -1 * hmodel.calc_evidence(
                    SS=SStotal,
                    doLogElapsedTime=True,
                    afterGlobalStep=not self.algParams['useSlackTermsInELBO'])
            else:
                # SS step. Scale at size of current batch.
                SS = hmodel.get_global_suff_stats(Dchunk, LP,
                    doLogElapsedTime=True)

                # Scale up to size of whole dataset.
                if hasattr(Dchunk, 'nDoc'):
                    ampF = Dchunk.nDocTotal / float(Dchunk.nDoc)
                    SS.applyAmpFactor(ampF)
                else:
                    ampF = Dchunk.nObsTotal / float(Dchunk.nObs)
                    SS.applyAmpFactor(ampF)

                # M step with learning rate
                hmodel.update_global_params(SS, rho, doLogElapsedTime=True)

                # ELBO step
                assert SS.hasAmpFactor()
                cur_batch_loss = -1 * hmodel.calc_evidence(
                    Dchunk, SS, LP, doLogElapsedTime=True)
                if loss_per_batch[batchID] != 0:
                    loss_running_sum -= loss_per_batch[batchID]
                loss_running_sum += cur_batch_loss
                loss_per_batch[batchID] = cur_batch_loss
                loss = loss_running_sum / nBatch

            # Display progress
            self.updateNumDataProcessed(Dchunk.get_size())
            if self.isLogCheckpoint(lapFrac, iterid):
                self.printStateToLog(hmodel, loss, lapFrac, iterid, rho=rho)

            # Save diagnostics and params
            if self.isSaveDiagnosticsCheckpoint(lapFrac, iterid):
                self.saveDiagnostics(lapFrac, SS, loss)
            if self.isSaveParamsCheckpoint(lapFrac, iterid):
                self.saveParams(lapFrac, hmodel, tryToSparsifyOutput=1)
                # don't save SS here, since its for one batch only
            self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

            if self.isLastBatch(lapFrac):
                ElapsedTimeLogger.writeToLogOnLapCompleted(lapFrac)
            # .... end loop over data

        # Finished! Save, print and exit
        self.printStateToLog(hmodel, loss, lapFrac, iterid, isFinal=1)
        self.saveParams(lapFrac, hmodel, SS)
        self.eval_custom_func(
            isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

        return self.buildRunInfo(Data=DataIterator, loss=loss, SS=SS)


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

    def save_batch_local_params_to_memory(self, batchID, batchLP):
        ''' Store certain fields of the provided local parameters dict
              into "memory" for later retrieval.
            Fields to save determined by the memoLPkeys attribute of this alg.
        '''
        batchLP = dict(**batchLP) # make a copy
        allkeys = list(batchLP.keys())
        for key in allkeys:
            if key != 'DocTopicCount':
                del batchLP[key]
        if len(list(batchLP.keys())) > 0:
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
