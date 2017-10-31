from builtins import *
import numpy as np

from .LearnAlg import LearnAlg, makeDictOfAllWorkspaceVars


class EMAlg(LearnAlg):

    """ Implementation of expectation-maximization learning algorithm.

    Key Methods
    ------
    fit : fit a provided model object to data.

    Attributes
    ------
    See LearnAlg.py
    """

    def __init__(self, **kwargs):
        ''' Create EMAlg instance, subtype of generic LearnAlg
        '''
        super(type(self), self).__init__(**kwargs)

    def fit(self, hmodel, Data, LP=None):
        ''' Fit point estimates of global parameters of hmodel to Data

        Returns
        --------
        Info : dict of run information.

        Post Condition
        --------
        hmodel updated in place with improved global parameters.
        '''
        self.set_start_time_now()
        isConverged = False
        prev_loss = -np.inf

        # Save initial state
        self.saveParams(0, hmodel)

        # Custom func hook
        self.eval_custom_func(
            isInitial=1, **makeDictOfAllWorkspaceVars(**vars()))

        for iterid in range(1, self.algParams['nLap'] + 1):
            lap = self.algParams['startLap'] + iterid
            nLapsCompleted = lap - self.algParams['startLap']
            self.set_random_seed_at_lap(lap)

            # Local/E step
            LP = hmodel.calc_local_params(Data, LP, **self.algParamsLP)

            # Summary step
            SS = hmodel.get_global_suff_stats(Data, LP)

            # ELBO calculation (needs to be BEFORE Mstep for EM)
            cur_loss = -1 * hmodel.calc_evidence(Data, SS, LP)
            if lap > 1.0:
                # Report warning if bound isn't increasing monotonically
                self.verify_monotonic_decrease(cur_loss, prev_loss)

            # Global/M step
            hmodel.update_global_params(SS)

            # Check convergence of expected counts
            countVec = SS.getCountVec()
            if lap > 1.0:
                isConverged = self.isCountVecConverged(countVec, prevCountVec)
                self.setStatus(lap, isConverged)

            # Display progress
            self.updateNumDataProcessed(Data.get_size())
            if self.isLogCheckpoint(lap, iterid):
                self.printStateToLog(hmodel, cur_loss, lap, iterid)

            # Save diagnostics and params
            if self.isSaveDiagnosticsCheckpoint(lap, iterid):
                self.saveDiagnostics(lap, SS, cur_loss)
            if self.isSaveParamsCheckpoint(lap, iterid):
                self.saveParams(lap, hmodel, SS)

            # Custom func hook
            self.eval_custom_func(**makeDictOfAllWorkspaceVars(**vars()))

            if nLapsCompleted >= self.algParams['minLaps'] and isConverged:
                break
            prev_loss = cur_loss
            prevCountVec = countVec.copy()
            # .... end loop over laps

        # Finished! Save, print and exit
        self.saveParams(lap, hmodel, SS)
        self.printStateToLog(hmodel, cur_loss, lap, iterid, isFinal=1)
        self.eval_custom_func(
            isFinal=1, **makeDictOfAllWorkspaceVars(**vars()))

        return self.buildRunInfo(Data=Data, loss=cur_loss, SS=SS, LP=LP)
