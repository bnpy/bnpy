from builtins import *
import numpy as np

from .LearnAlg import LearnAlg, makeDictOfAllWorkspaceVars


class VBAlg(LearnAlg):

    """ Variational Bayes (VB) learning algorithm.

    Extends
    -------
    LearnAlg
    """

    def __init__(self, **kwargs):
        ''' Create VBLearnAlg, subtype of generic LearnAlg
        '''
        LearnAlg.__init__(self, **kwargs)

    def fit(self, hmodel, Data, LP=None):
        ''' Run VB learning to fit global parameters of hmodel to Data

        Returns
        --------
        Info : dict of run information.

        Post Condition
        --------
        hmodel updated in place with improved global parameters.
        '''
        self.set_start_time_now()
        prev_loss = np.inf
        isConverged = False
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
            self.algParamsLP['lapFrac'] = lap  # logging
            self.algParamsLP['batchID'] = 1
            LP = hmodel.calc_local_params(Data, LP, **self.algParamsLP)

            # Summary step
            SS = hmodel.get_global_suff_stats(Data, LP)

            # Global/M step
            hmodel.update_global_params(SS)

            # ELBO calculation
            cur_loss = -1 * hmodel.calc_evidence(Data, SS, LP)
            if lap > 1.0:
                # Report warning if loss function isn't behaving monotonically
                self.verify_monotonic_decrease(cur_loss, prev_loss)

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
