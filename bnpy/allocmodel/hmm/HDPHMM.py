from builtins import *
import copy
import numpy as np
import logging

from . import HMMUtil
from .HDPHMMUtil import ELBOTermDimMap, calcELBO
from .HDPHMMUtil import calcELBO_LinearTerms, calcELBO_NonlinearTerms

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import digamma, gammaln
from bnpy.util import StickBreakUtil
from bnpy.allocmodel.topics import OptimizerRhoOmega
from bnpy.allocmodel.topics.HDPTopicUtil import c_Beta, c_Dir, L_top
from bnpy.util import sharedMemToNumpyArray, numpyToSharedMemArray

Log = logging.getLogger('bnpy')


class HDPHMM(AllocModel):

    """ Hierarchical Dirichlet process Hidden Markov model (HDP-HMM)

    Truncated to finite number of K active states.

    Attributes
    -------
    inferType : string {'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    K : int
        number of states
    startAlpha : float
        scalar pseudo-count
        used in Dirichlet prior on starting state probabilities.
    transAlpha : float
        scalar pseudo-count
        used in Dirichlet prior on state-to-state transition probabilities.
    kappa : float
        scalar pseudo-count
        adds mass to probability of self-transition

    Attributes for VB
    ---------
    startTheta : 1D array, K
        Vector parameterizes Dirichlet posterior q(\pi_{0})
    transTheta : 2D array, K x K
        Vector that parameterizes Dirichlet posterior q(\pi_k), k>0

    Local Parameters
    --------
    resp :  2D array, T x K
        q(z_t=k) = resp[t,k]
    respPair : 3D array, T x K x K
        q(z_t=k, z_t-1=j) = respPair[t,j,k]
    """

    def __init__(self, inferType, priorDict=dict()):
        if inferType == 'EM':
            raise ValueError('EM is not supported for HDPHMM')

        self.set_prior(**priorDict)
        self.inferType = inferType
        self.K = 0

    def set_prior(self, gamma=10, alpha=0.5,
                  startAlpha=5.0, hmmKappa=0.0,
                  nGlobalIters=1, nGlobalItersBigChange=10, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.startAlpha = startAlpha
        self.kappa = hmmKappa
        self.nGlobalIters = nGlobalIters
        self.nGlobalItersBigChange = nGlobalItersBigChange

    def get_active_comp_probs(self):
        ''' Return K vector of appearance probabilities for each of the K comps
        '''
        return StickBreakUtil.rho2beta_active(self.rho)

    def get_init_prob_vector(self):
        ''' Get vector of initial probabilities for all K active states
        '''
        expELogPi0 = digamma(
            self.startTheta) - digamma(np.sum(self.startTheta))
        np.exp(expELogPi0, out=expELogPi0)
        return expELogPi0[0:self.K]

    def get_trans_prob_matrix(self):
        ''' Get matrix of transition probabilities for all K active states
        '''
        digammaSumVec = digamma(np.sum(self.transTheta, axis=1))
        expELogPi = digamma(self.transTheta) - digammaSumVec[:, np.newaxis]
        np.exp(expELogPi, out=expELogPi)
        return expELogPi[0:self.K, 0:self.K]

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Calculate local parameters for each data item and each component.

        This is part of the E-step.

        Args
        -------
        Data : bnpy data object with Data.nObs observations
        LP : local param dict with fields
            * E_log_soft_ev : Data.nObs x K array where
            E_log_soft_ev[n,k] = log p(data obs n | comp k)

        Returns
        -------
        LP : dict of local parameters.
        '''
        return HMMUtil.calcLocalParams(Data, LP,
                                       transTheta=self.transTheta,
                                       startTheta=self.startTheta,
                                       **kwargs)

    def initLPFromResp(self, Data, LP, limitMemoryLP=1):
        ''' Fill in remaining local parameters given resp.

        Returns
        --------
        LP : dict, with fields
            * respPair
        '''
        K = LP['resp'].shape[1]
        if limitMemoryLP:
            LP['TransCount'] = np.zeros((Data.nDoc, K, K))
        else:
            LP['respPair'] = np.zeros((Data.doc_range[-1], K, K))
        for n in range(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n + 1]
            if limitMemoryLP:
                for t in range(start + 1, stop):
                    respPair_t = np.outer(
                        LP['resp'][
                            t - 1,
                            :],
                        LP['resp'][
                            t,
                            :])
                    LP['TransCount'][n] += respPair_t
            else:
                R = LP['resp']
                LP['respPair'][start + 1:stop] = \
                    R[start:stop - 1][:, :, np.newaxis] \
                    * R[start + 1:stop][:, np.newaxis, :]
        return LP

    def selectSubsetLP(self, Data, LP, relIDs):
        ''' Create local parameter dict for subset of sequences in Data

        Returns
        -------
        subsetLP : local params dict
        '''
        relIDs = np.asarray(relIDs, dtype=np.int32)
        if relIDs.size == Data.nDoc:
            if np.allclose(relIDs, np.arange(Data.nDoc)):
                return copy.deepcopy(LP)
        T_all = np.sum(Data.doc_range[relIDs + 1] - Data.doc_range[relIDs])
        K = LP['resp'].shape[1]
        resp = np.zeros((T_all, K))
        if 'respPair' in LP:
            respPair = np.zeros((T_all, K, K))
        else:
            TransCount = np.zeros((len(relIDs), K, K))
            Htable = np.zeros((len(relIDs), K, K))
        nstart = 0
        for ii, n in enumerate(relIDs):
            start = Data.doc_range[n]
            stop = Data.doc_range[n + 1]
            nstop = nstart + stop - start
            resp[nstart:nstop] = LP['resp'][start:stop]
            if 'respPair' in LP:
                respPair[nstart:nstop] = LP['respPair'][start:stop]
            else:
                TransCount[ii] = LP['TransCount'][n]
                Htable[ii] = LP['Htable'][n]
            nstart = nstop
        if 'respPair' in LP:
            subsetLP = dict(resp=resp, respPair=respPair)
        else:
            subsetLP = dict(resp=resp, TransCount=TransCount, Htable=Htable)
        return subsetLP

    def fillSubsetLP(self, Data, LP, targetLP, targetIDs):
        ''' Fill in local parameters for a subset of sequences/documents.

        Args
        -----
        LP : dict of local params
            represents K states and nDoc sequences
        targetLP : dict of local params
            represents K+Kx states and a subset of nDoc sequences

        Returns
        -------
        newLP : dict of local params, with K + Kx components
        '''
        nAtom = LP['resp'].shape[0]
        Knew = targetLP['resp'].shape[1]
        Kold = LP['resp'].shape[1]
        newResp = np.zeros((nAtom, Knew))
        newResp[:, :Kold] = LP['resp']
        newTransCount = np.zeros((Data.nDoc, Knew, Knew))
        newTransCount[:, :Kold, :Kold] = LP['TransCount']
        newHtable = np.zeros((Data.nDoc, Knew, Knew))
        newHtable[:, :Kold, :Kold] = LP['Htable']
        start_t = 0
        for ii, n in enumerate(targetIDs):
            assert n >= 0
            assert n < Data.nDoc
            start = Data.doc_range[n]
            stop = Data.doc_range[n+1]
            stop_t = start_t + (stop-start)
            newResp[start:stop] = targetLP['resp'][start_t:stop_t]
            newTransCount[n] = targetLP['TransCount'][ii]
            newHtable[n] = targetLP['Htable'][ii]
            start_t = stop_t
        return dict(resp=newResp, TransCount=newTransCount, Htable=newHtable)

    def getSummaryFieldNames(self):
        return ['StartStateCount', 'TransStateCount']

    def getSummaryFieldDims(self):
        return [('K'), ('K', 'K')]

    def get_global_suff_stats(self, Data, LP, **kwargs):

        return calcSummaryStats(Data, LP, **kwargs)

    def forceSSInBounds(self, SS):
        ''' Force TransStateCount and StartStateCount to be >= 0.

        This avoids numerical issues in memoized updates
        where SS "chunks" are added and subtracted incrementally
        such as:
          x = 10
          x += 1e-15
          x -= 10
          x -= 1e-15
        resulting in x < 0.

        Returns
        -------
        Nothing.  SS is updated in-place.
        '''
        np.maximum(SS.TransStateCount, 0, out=SS.TransStateCount)
        np.maximum(SS.StartStateCount, 0, out=SS.StartStateCount)

    def find_optimum_rhoOmega(self, **kwargs):
        ''' Performs numerical optimization of rho and omega for M-step update.

        Note that the optimizer forces rho to be in [EPS, 1-EPS] for
        the sake of numerical stability

        Returns
        -------
        rho : 1D array, size K
        omega : 1D array, size K
        Info : dict of information about optimization.
        '''

        # Calculate expected log transition probability
        # using theta vectors for all K states plus initial state
        ELogPi = digamma(self.transTheta) \
            - digamma(np.sum(self.transTheta, axis=1))[:, np.newaxis]
        sumELogPi = np.sum(ELogPi, axis=0)
        startELogPi = digamma(self.startTheta) \
            - digamma(np.sum(self.startTheta))

        # Select initial rho, omega values for gradient descent
        if hasattr(self, 'rho') and self.rho.size == self.K:
            initRho = self.rho
        else:
            initRho = None

        if hasattr(self, 'omega') and self.omega.size == self.K:
            initOmega = self.omega
        else:
            initOmega = None

        # Do the optimization
        try:
            rho, omega, fofu, Info = \
                OptimizerRhoOmega.find_optimum_multiple_tries(
                    sumLogPi=sumELogPi,
                    sumLogPiActiveVec=None,
                    sumLogPiRemVec=None,
                    startAlphaLogPi=self.startAlpha * startELogPi,
                    nDoc=self.K + 1,
                    gamma=self.gamma,
                    alpha=self.alpha,
                    kappa=self.kappa,
                    initrho=initRho,
                    initomega=initOmega)
            self.OptimizerInfo = Info
            self.OptimizerInfo['fval'] = fofu

        except ValueError as error:
            if hasattr(self, 'rho') and self.rho.size == self.K:
                Log.error(
                    '***** Optim failed. Remain at cur val. ' +
                    str(error))
                rho = self.rho
                omega = self.omega
            else:
                Log.error('***** Optim failed. Set to prior. ' + str(error))
                omega = (self.gamma + 1) * np.ones(SS.K)
                rho = 1 / float(1 + self.gamma) * np.ones(SS.K)

        return rho, omega

    def update_global_params_EM(self, SS, **kwargs):
        raise ValueError('HDPHMM does not support EM')

    def update_global_params_VB(self, SS,
                                mergeCompA=None, mergeCompB=None,
                                **kwargs):
        ''' Update global parameters.
        '''
        self.K = SS.K
        if not hasattr(self, 'rho') or self.rho.size != SS.K:
            # Big change from previous model is being proposed.
            # We'll init rho from scratch, and need more iters to improve.
            nGlobalIters = self.nGlobalItersBigChange
        else:
            # Small change required. Current rho is good initialization.
            nGlobalIters = self.nGlobalIters

        # Special update case for merges:
        # Fast, heuristic update for new rho given original value
        if mergeCompA is not None:
            beta = OptimizerRhoOmega.rho2beta_active(self.rho)
            beta[mergeCompA] += beta[mergeCompB]
            beta = np.delete(beta, mergeCompB, axis=0)
            self.rho = OptimizerRhoOmega.beta2rho(beta, SS.K)
            omega = self.omega
            omega[mergeCompA] += omega[mergeCompB]
            self.omega = np.delete(omega, mergeCompB, axis=0)
        # TODO think about smarter init for rho/omega??

        # Update theta with recently updated info from suff stats
        self.transTheta, self.startTheta = self._calcTheta(SS)

        for giter in range(nGlobalIters):
            # Update rho, omega through numerical optimization
            self.rho, self.omega = self.find_optimum_rhoOmega(**kwargs)
            # Update theta again to reflect the new rho, omega
            self.transTheta, self.startTheta = self._calcTheta(SS)

    def update_global_params_soVB(self, SS, rho, **kwargs):
        ''' Updates global parameters when learning with stochastic online VB.
            Note that the rho here is the learning rate parameter, not
            the global stick weight parameter rho
        '''
        self.K = SS.K

        # Update theta (1/2), incorporates recently updated suff stats
        transThetaStar, startThetaStar = self._calcTheta(SS)
        self.transTheta = rho * transThetaStar + (1 - rho) * self.transTheta
        self.startTheta = rho * startThetaStar + (1 - rho) * self.startTheta

        # Update rho/omega
        rhoStar, omegaStar = self.find_optimum_rhoOmega(**kwargs)
        g1 = (1 - rho) * (self.rho * self.omega) + rho * (rhoStar * omegaStar)
        g0 = (1 - rho) * ((1 - self.rho) * self.omega) + \
            rho * ((1 - rhoStar) * omegaStar)
        self.rho = g1 / (g1 + g0)
        self.omega = g1 + g0

        # TODO: update theta (2/2)?? incorporates recent rho/omega updates

    def _calcTheta(self, SS):
        ''' Update parameters theta to maximize objective given suff stats.

        Returns
        ---------
        transTheta : 2D array, size K x K+1
        startTheta : 1D array, size K
        '''
        K = SS.K
        if not hasattr(self, 'rho') or self.rho.size != K:
            self.rho = OptimizerRhoOmega.create_initrho(K)

        # Calculate E_q[alpha * Beta_l] for l = 1, ..., K+1
        Ebeta = StickBreakUtil.rho2beta(self.rho)
        alphaEBeta = self.alpha * Ebeta

        # transTheta_kl = M_kl + E_q[alpha * Beta_l] + kappa * 1_{k==l}
        transTheta = np.zeros((K, K + 1))
        transTheta += alphaEBeta[np.newaxis, :]
        transTheta[:K, :K] += SS.TransStateCount + self.kappa * np.eye(self.K)

        # startTheta_k = r_1k + E_q[alpha * Beta_l] (where r_1,>K = 0)
        startTheta = self.startAlpha * Ebeta
        startTheta[:K] += SS.StartStateCount
        return transTheta, startTheta

    def init_global_params(self, Data, K=0, **initArgs):
        ''' Initialize rho, omega, and theta to reasonable values.

        This is only called by "from scratch" init routines.
        '''
        self.K = K
        self.rho = OptimizerRhoOmega.create_initrho(K)
        self.omega = (1.0 + self.gamma) * np.ones(K)

        # To initialize theta, perform standard update given rho, omega
        # but with "empty" sufficient statistics.
        SS = SuffStatBag(K=self.K, D=Data.dim)
        SS.setField('StartStateCount', np.ones(K), dims=('K'))
        SS.setField('TransStateCount', np.ones((K, K)), dims=('K', 'K'))
        self.transTheta, self.startTheta = self._calcTheta(SS)

    def set_global_params(self, hmodel=None,
                          rho=None, omega=None,
                          startTheta=None, transTheta=None,
                          **kwargs):
        ''' Set rho, omega to provided values.
        '''
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            if hasattr(hmodel.allocModel, 'rho'):
                self.rho = hmodel.allocModel.rho
                self.omega = hmodel.allocModel.omega
            else:
                raise AttributeError('Unrecognized hmodel. No field rho.')
            if hasattr(hmodel.allocModel, 'startTheta'):
                self.startTheta = hmodel.allocModel.startTheta
                self.transTheta = hmodel.allocModel.transTheta
            else:
                raise AttributeError(
                    'Unrecognized hmodel. No field startTheta.')
        elif rho is not None \
                and omega is not None \
                and startTheta is not None:
            self.rho = rho
            self.omega = omega
            self.startTheta = startTheta
            self.transTheta = transTheta
            self.K = omega.size
            assert self.K == self.startTheta.size - 1
        else:
            self._set_global_params_from_scratch(**kwargs)

    def _set_global_params_from_scratch(self, beta=None,
                                        Data=None, nDoc=None, **kwargs):
        ''' Set rho, omega to values that reproduce provided appearance probs

        Args
        --------
        beta : 1D array, size K
            beta[k] gives top-level probability for active comp k
        '''
        if nDoc is None:
            nDoc = Data.nDoc
        if nDoc is None:
            raise ValueError('Bad parameters. nDoc not specified.')
        if beta is not None:
            beta = beta / beta.sum()
        if beta is None:
            raise ValueError('Bad parameters. Vector beta not specified.')
        Ktmp = beta.size
        rem = np.minimum(0.05, 1. / (Ktmp))
        beta = np.hstack([np.squeeze(beta), rem])
        beta = beta / np.sum(beta)
        self.K = beta.size - 1
        self.rho, self.omega = self._convert_beta2rhoomega(beta)
        assert self.rho.size == self.K
        assert self.omega.size == self.K

    def _convert_beta2rhoomega(self, beta, nDoc=10):
        ''' Find vectors rho, omega that are probable given beta

        Returns
        --------
        rho : 1D array, size K
        omega : 1D array, size K
        '''
        assert abs(np.sum(beta) - 1.0) < 0.001
        rho = OptimizerRhoOmega.beta2rho(beta, self.K)
        omega = (nDoc + self.gamma) * np.ones(rho.size)
        return rho, omega

    def calc_evidence(self, Data, SS, LP, todict=False, **kwargs):
        ''' Calculate ELBO objective function value for provided state.

        Returns
        -------
        L : float
        '''
        assert hasattr(self, 'rho')
        return calcELBO(Data=Data, SS=SS, LP=LP,
                        startAlpha=self.startAlpha, alpha=self.alpha,
                        kappa=self.kappa, gamma=self.gamma,
                        rho=self.rho, omega=self.omega,
                        transTheta=self.transTheta, startTheta=self.startTheta,
                        todict=todict, **kwargs)

    def calcELBO_LinearTerms(self, **kwargs):
        ''' Compute sum of ELBO terms that are linear/const wrt suff stats

        Returns
        -------
        L : float
        '''
        return calcELBO_LinearTerms(
            startAlpha=self.startAlpha, alpha=self.alpha,
            kappa=self.kappa, gamma=self.gamma,
            rho=self.rho, omega=self.omega,
            transTheta=self.transTheta, startTheta=self.startTheta,
            **kwargs)

    def calcELBO_NonlinearTerms(self, **kwargs):
        ''' Compute sum of ELBO terms that are NONlinear wrt suff stats

        Returns
        -------
        L : float
        '''
        return calcELBO_NonlinearTerms(**kwargs)

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate scalar improvement in ELBO for hard merge of comps kA, kB

        Does *not* include any entropy.

        Returns
        ---------
        L : scalar
        '''
        m_K = SS.K - 1
        m_SS = SuffStatBag(K=SS.K, D=0)
        m_SS.setField('StartStateCount', SS.StartStateCount.copy(), dims='K')
        m_SS.setField('TransStateCount', SS.TransStateCount.copy(),
                      dims=('K', 'K'))
        m_SS.mergeComps(kA, kB)

        # Create candidate beta vector
        m_beta = StickBreakUtil.rho2beta(self.rho)
        m_beta[kA] += m_beta[kB]
        m_beta = np.delete(m_beta, kB, axis=0)

        # Create candidate rho and omega vectors
        m_rho = StickBreakUtil.beta2rho(m_beta, m_K)
        m_omega = np.delete(self.omega, kB)

        # Create candidate startTheta
        m_startTheta = self.startAlpha * m_beta.copy()
        m_startTheta[:m_K] += m_SS.StartStateCount

        # Create candidate transTheta
        m_transTheta = self.alpha * np.tile(m_beta, (m_K, 1))
        if self.kappa > 0:
            m_transTheta[:, :m_K] += self.kappa * np.eye(m_K)
        m_transTheta[:, :m_K] += m_SS.TransStateCount

        # Evaluate objective func. for both candidate and current model
        Lcur = calcELBO_LinearTerms(
            SS=SS, rho=self.rho, omega=self.omega,
            startTheta=self.startTheta, transTheta=self.transTheta,
            alpha=self.alpha, startAlpha=self.startAlpha,
            gamma=self.gamma, kappa=self.kappa)

        Lprop = calcELBO_LinearTerms(
            SS=m_SS, rho=m_rho, omega=m_omega,
            startTheta=m_startTheta, transTheta=m_transTheta,
            alpha=self.alpha, startAlpha=self.startAlpha,
            gamma=self.gamma, kappa=self.kappa)

        # Note: This gap relies on fact that all nonlinear terms are entropies,
        return Lprop - Lcur

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calc matrix of improvement in ELBO for all possible pairs of comps
        '''
        Gap = np.zeros((SS.K, SS.K))
        for kB in range(1, SS.K):
            for kA in range(0, kB):
                Gap[kA, kB] = self.calcHardMergeGap(SS, kA, kB)
        return Gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc matrix of improvement in ELBO for all possible pairs of comps
        '''
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
        return Gaps

    def to_dict(self):
        return dict(transTheta=self.transTheta,
                    startTheta=self.startTheta,
                    omega=self.omega, rho=self.rho)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.transTheta = myDict['transTheta']
        self.startTheta = myDict['startTheta']
        self.omega = myDict['omega']
        self.rho = myDict['rho']

    def get_prior_dict(self):
        return dict(gamma=self.gamma, alpha=self.alpha, K=self.K,
                    hmmKappa=self.kappa, startAlpha=self.startAlpha)

    def getSerializableParamsForLocalStep(self):
        """ Get compact dict of params for parallel local step.

        Returns
        -------
        Info : dict
        """
        return dict(inferType=self.inferType,
                    K=self.K)

    def fillSharedMemDictForLocalStep(self, ShMem=None):
        """ Get dict of shared mem arrays needed for parallel local step.

        Returns
        -------
        ShMem : dict of RawArray objects
        """
        # No shared memory required here.
        if not isinstance(ShMem, dict):
            ShMem = dict()

        K = self.K
        if 'startTheta' in ShMem:
            shared_startTheta = sharedMemToNumpyArray(ShMem['startTheta'])
            assert shared_startTheta.size >= K + 1
            shared_startTheta[:K + 1] = self.startTheta

            shared_transTheta = sharedMemToNumpyArray(ShMem['transTheta'])
            assert shared_transTheta.shape[0] >= K
            assert shared_transTheta.shape[1] >= K + 1
            shared_transTheta[:K, :K + 1] = self.transTheta
        else:
            ShMem['startTheta'] = numpyToSharedMemArray(self.startTheta)
            ShMem['transTheta'] = numpyToSharedMemArray(self.transTheta)
        return ShMem

    def getLocalAndSummaryFunctionHandles(self):
        """ Get function handles for local step and summary step

        Useful for parallelized algorithms.

        Returns
        -------
        calcLocalParams : f handle
        calcSummaryStats : f handle
        """
        return HMMUtil.calcLocalParams, calcSummaryStats
    # .... end class HDPHMM


def calcSummaryStats(Data, LP,
                     doPrecompEntropy=0,
                     doPrecompMergeEntropy=0,
                     mPairIDs=None,
                     trackDocUsage=0,
                     **kwargs):
    ''' Calculate summary statistics for given data slice and local params.

    Returns
    -------
    SS : SuffStatBag
    '''
    if mPairIDs is None:
        M = 0
    else:
        M = len(mPairIDs)

    resp = LP['resp']
    K = resp.shape[1]
    startLocIDs = Data.doc_range[:-1]
    StartStateCount = np.sum(resp[startLocIDs], axis=0)
    N = np.sum(resp, axis=0)

    if 'TransCount' in LP:
        TransStateCount = np.sum(LP['TransCount'], axis=0)
    else:
        respPair = LP['respPair']
        TransStateCount = np.sum(respPair, axis=0)

    SS = SuffStatBag(K=K, D=Data.dim, M=M)
    SS.setField('StartStateCount', StartStateCount, dims=('K'))
    SS.setField('TransStateCount', TransStateCount, dims=('K', 'K'))
    SS.setField('N', N, dims=('K'))
    SS.setField('nDoc', Data.nDoc, dims=None)

    if doPrecompEntropy or 'Htable' in LP:
        # Compute entropy terms!
        # 'Htable', 'Hstart' will both be in Mdict
        Mdict = calcELBO_NonlinearTerms(Data=Data,
                                        LP=LP, returnMemoizedDict=1)
        SS.setELBOTerm('Htable', Mdict['Htable'], dims=('K', 'K'))
        SS.setELBOTerm('Hstart', Mdict['Hstart'], dims=('K'))

    if doPrecompMergeEntropy:
        subHstart, subHtable = HMMUtil.PrecompMergeEntropy_SpecificPairs(
            LP, Data, mPairIDs)
        SS.setMergeTerm('Hstart', subHstart, dims=('M'))
        SS.setMergeTerm('Htable', subHtable, dims=('M', 2, 'K'))
        SS.mPairIDs = np.asarray(mPairIDs)

    if trackDocUsage:
        # Track how often topic appears in a seq. with mass > thresh.
        DocUsage = np.zeros(K)
        for n in range(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n + 1]
            DocUsage += np.sum(LP['resp'][start:stop], axis=0) > 0.01
        SS.setSelectionTerm('DocUsageCount', DocUsage, dims='K')
    return SS
