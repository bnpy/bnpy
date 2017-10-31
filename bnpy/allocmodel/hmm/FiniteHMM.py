from builtins import *
import numpy as np

from . import HMMUtil
from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import digamma, gammaln, as2D


def log_pdf_dirichlet(PiMat, alphavec):
    ''' Return scalar log probability for Dir(PiMat | alphavec)
    '''
    PiMat = as2D(PiMat + 1e-100)
    J, K = PiMat.shape
    if isinstance(alphavec, float):
        alphavec = alphavec * np.ones(K)
    elif alphavec.ndim == 0:
        alphavec = alphavec * np.ones(K)
    assert alphavec.size == K
    cDir = gammaln(np.sum(alphavec)) - np.sum(gammaln(alphavec))
    return K * cDir + np.sum(np.dot(np.log(PiMat), alphavec - 1.0))


class FiniteHMM(AllocModel):

    ''' Hidden Markov model (HMM) with finite number of hidden states.

    Attributes
    -------
    inferType : string {'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    K : int
        number of components
    startAlpha : float
        scalar pseudo-count
        used in Dirichlet prior on starting state probabilities.
    transAlpha : float
        scalar pseudo-count
        used in Dirichlet prior on state-to-state transition probabilities.
    kappa : float
        scalar pseudo-count
        adds mass to probability of self-transition

    Attributes for EM
    ---------
    startPi : 1D array, K
        Probability of starting sequence in each state.
    transPi : 2D array, K x K
        Probability of transition between each pair of states.

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

    '''

    def __init__(self, inferType, priorDict):
        self.inferType = inferType
        self.set_prior(**priorDict)
        self.K = 0  # Number of states

    def set_prior(self, startAlpha=.1, transAlpha=.1, hmmKappa=0.0,
                  **kwargs):
        ''' Set hyperparameters that control state transition probs
        '''
        self.startAlpha = startAlpha
        self.transAlpha = transAlpha
        self.kappa = hmmKappa

    def get_active_comp_probs(self):
        ''' Get vector of appearance probabilities for each active state
        '''
        if self.inferType == 'EM':
            return self.transPi.mean(axis=0)
        else:
            EPiMat = self.transTheta / \
                self.transTheta.sum(axis=1)[:, np.newaxis]
            return EPiMat.mean(axis=0)

    def get_init_prob_vector(self):
        ''' Get vector of initial probabilities for all K active states
        '''
        if self.inferType == 'EM':
            pi0 = self.startPi
        else:
            pi0 = np.exp(digamma(self.startTheta) -
                         digamma(np.sum(self.startTheta)))
        return pi0

    def get_trans_prob_matrix(self):
        ''' Get matrix of transition probabilities for all K active states
        '''
        if self.inferType == 'EM':
            EPiMat = self.transPi
        else:
            digammasumVec = digamma(np.sum(self.transTheta, axis=1))
            EPiMat = np.exp(digamma(self.transTheta) -
                            digammasumVec[:, np.newaxis])
        return EPiMat

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Local update step

        Args
        -------
        Data : bnpy data object

        Returns
        -------
        LP : dict
            Local parameters, with updated fields
            * resp : 2D array T x K
            * respPair : 3D array, T x K x K

        Notes
        -----
        Runs the forward backward algorithm (from HMMUtil) to calculate resp
        and respPair and adds them to the LP dict
        '''
        logSoftEv = LP['E_log_soft_ev']
        K = logSoftEv.shape[1]

        # First calculate input params for forward-backward alg,
        # These calculations are different for EM and VB
        if self.inferType.count('VB') > 0:
            # Row-wise subtraction
            digammasumVec = digamma(np.sum(self.transTheta, axis=1))
            expELogTrans = np.exp(digamma(self.transTheta) -
                                  digammasumVec[:, np.newaxis])
            ELogPi0 = (digamma(self.startTheta) -
                       digamma(np.sum(self.startTheta)))
            transParam = expELogTrans
        elif self.inferType == 'EM' > 0:
            ELogPi0 = np.log(self.startPi + 1e-40)
            transParam = self.transPi
        else:
            raise ValueError('Unrecognized inferType')

        initParam = np.ones(K)

        # Run forward-backward algorithm on each sequence
        logMargPr = np.empty(Data.nDoc)
        resp = np.empty((Data.nObs, K))
        respPair = np.zeros((Data.nObs, K, K))
        for n in range(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n + 1]
            logSoftEv_n = logSoftEv[start:stop]
            logSoftEv_n[0] += ELogPi0  # adding in start state log probs

            seqResp, seqRespPair, seqLogMargPr = \
                HMMUtil.FwdBwdAlg(initParam, transParam, logSoftEv_n)

            resp[start:stop] = seqResp
            respPair[start:stop] = seqRespPair
            logMargPr[n] = seqLogMargPr

        LP['resp'] = resp
        LP['respPair'] = respPair
        if self.inferType == 'EM':
            LP['evidence'] = np.sum(logMargPr)
        return LP

    def initLPFromResp(self, Data, LP, deleteCompID=None):
        ''' Fill in remaining local parameters given token-topic resp.

        Args
        ----
        LP : dict with fields
            * resp : 2D array, size T x K

        Returns
        -------
        LP : dict with fields
            * respPair
        '''
        resp = LP['resp']
        N, K = resp.shape
        respPair = np.zeros((N, K, K))

        # Loop over each sequence,
        # and define pair-wise responsibilities via an outer-product
        for n in range(Data.nDoc):
            start = Data.doc_range[n]
            stop = Data.doc_range[n + 1]
            R = resp[start:stop]
            respPair[start + 1:stop] = R[:-1, :, np.newaxis] \
                * R[1:, np.newaxis, :]
        LP['respPair'] = respPair
        return LP

    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
        ''' Create sufficient stats needed for global param updates

        Args
        -------
        Data : bnpy data object
        LP : Dictionary containing the local parameters. Expected to contain:
            resp : Data.nObs x K array
            respPair : Data.nObs x K x K array (from the def. of respPair, note
                       respPair[0,:,:] is undefined)

        Returns
        -------
        SS : SuffStatBag with fields
            StartStateCount : A vector of length K with entry i being
                             resp(z_{1k}) = resp[0,:]
            TransStateCount : A K x K matrix where TransStateCount[i,j] =
                           sum_{n=2}^K respPair(z_{n-1,j}, z_{nk})
            N : A vector of length K with entry k being
                sum_{n=1}^Data.nobs resp(z_{nk})

            The first two of these are used by FiniteHMM.update_global_params,
            and the third is used by ObsModel.update_global_params.

        (see the documentation for information about resp and respPair)
        '''
        resp = LP['resp']
        respPair = LP['respPair']
        K = resp.shape[1]
        startLocIDs = Data.doc_range[:-1]

        StartStateCount = np.sum(resp[startLocIDs], axis=0)
        N = np.sum(resp, axis=0)
        TransStateCount = np.sum(respPair, axis=0)

        SS = SuffStatBag(K=K, D=Data.dim)
        SS.setField('StartStateCount', StartStateCount, dims=('K'))
        SS.setField('TransStateCount', TransStateCount, dims=('K', 'K'))
        SS.setField('N', N, dims=('K'))

        if doPrecompEntropy is not None:
            entropy = self.elbo_entropy(Data, LP)
            SS.setELBOTerm('Elogqz', entropy, dims=None)
        return SS

    def forceSSInBounds(self, SS):
        ''' Force SS fields to avoid numerical badness in fields.

        This avoids numerical issues in moVB like the one below
        due to SS "chunks" being added and subtracted incrementally.
              x = 10
              x += 1e-15
              x -= 10
              x -= 1e-15
            resulting in x < 0, when x should be exactly 0.

        Post Condition
        --------------
        Fields of SS updated in-place.
        '''
        np.maximum(SS.N, 0, out=SS.N)
        np.maximum(SS.TransStateCount, 0, out=SS.TransStateCount)
        np.maximum(SS.StartStateCount, 0, out=SS.StartStateCount)

    def update_global_params_EM(self, SS, **kwargs):
        ''' Perform global step using EM objective.

        Args
        -------
        SS : bnpy SuffStatBag with K components.
            Required fields
            * StartStateCount
            * TransStateCount

        Post Condition
        --------------
        Fields startPi and transPi updated in place.
        '''
        self.K = SS.K
        if self.startAlpha <= 1.0:
            self.startPi = SS.StartStateCount
        else:
            self.startPi = SS.StartStateCount + self.startAlpha - 1.0
        self.startPi /= self.startPi.sum()

        if self.transAlpha <= 1.0:
            self.transPi = SS.TransStateCount
        else:
            self.transPi = SS.TransStateCount + self.transAlpha - 1.0
        rowSums = self.transPi.sum(axis=1) + 1e-15
        self.transPi /= rowSums[:, np.newaxis]

    def update_global_params_VB(self, SS, **kwargs):
        ''' Perform global step using EM objective.

        Args
        -------
        SS : bnpy SuffStatBag with K components.
            Required fields
            * StartStateCount
            * TransStateCount

        Post Condition
        --------------
        Fields transTheta, startTheta updated in place.
        '''
        self.startTheta = self.startAlpha + SS.StartStateCount
        self.transTheta = self.transAlpha + SS.TransStateCount + \
            self.kappa * np.eye(self.K)
        self.K = SS.K

    def update_global_params_soVB(self, SS, rho, **kwargs):
        startNew = self.startAlpha + SS.StartStateCount
        transNew = self.transAlpha + SS.TransStateCount + \
            self.kappa * np.eye(self.K)
        self.startTheta = rho * startNew + (1 - rho) * self.startTheta
        self.transTheta = rho * transNew + (1 - rho) * self.transTheta
        self.K = SS.K

    def init_global_params(self, Data, K=0, **kwargs):
        ''' Default initialization of global parameters when

        Not used for local-first initializations, such as
        * contigBlocksLP
        * randexamples
        * kmeansplusplus
        '''
        self.K = K
        if self.inferType == 'EM':
            self.startPi = 1.0 / K * np.ones(K)
            self.transPi = 1.0 / K * np.ones((K, K))
        else:
            self.startTheta = self.startAlpha + np.ones(K)
            self.transTheta = self.transAlpha + np.ones((K, K)) + \
                self.kappa * np.eye(self.K)

    def set_global_params(self, hmodel=None, K=None,
                          startPi=None, transPi=None,
                          **kwargs):
        """ Set global parameters to provided values.

        Post Condition for EM
        ---------------------
        startPi, transPi define valid probability parameters w/ K states.

        Post Condition for VB
        -------
        startTheta, transTheta define valid posterior over K components.
        """
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            if self.inferType == 'EM':
                self.startPi = hmodel.allocModel.startPi
                self.transPi = hmodel.allocModel.transPi
            elif self.inferType.count('VB'):
                self.startTheta = hmodel.allocModel.startTheta
                self.transTheta = hmodel.allocModel.transTheta
        else:
            self.K = K
            if self.inferType == 'EM':
                self.startPi = startPi
                self.transPi = transPi
            elif self.inferType.count('VB'):
                self.startTheta = startTheta
                self.transTheta = transTheta

    def calc_evidence(self, Data, SS, LP, todict=False, **kwargs):
        if self.inferType == 'EM':
            if self.startAlpha < 1.0:
                logprior_init = 0
            else:
                logprior_init = log_pdf_dirichlet(
                    self.startPi,
                    self.startAlpha)
            if self.transAlpha < 1.0:
                logprior_trans = 0
            else:
                logprior_trans = log_pdf_dirichlet(
                    self.transPi,
                    self.transAlpha)

            return LP['evidence'] + logprior_init + logprior_trans

        elif self.inferType.count('VB') > 0:
            if SS.hasELBOTerm('Elogqz'):
                entropy = SS.getELBOTerm('Elogqz')
            else:
                entropy = self.elbo_entropy(Data, LP)
            # For stochastic (soVB), we need to scale up the entropy
            # Only used when --doMemoELBO is set to 0 (not recommended)
            if SS.hasAmpFactor():
                entropy *= SS.ampF
            return entropy + self.elbo_alloc()
        else:
            emsg = 'Unrecognized inferType: ' + self.inferType
            raise NotImplementedError(emsg)

    def elbo_entropy(self, Data, LP):
        return HMMUtil.calcEntropyFromResp(LP['resp'], LP['respPair'], Data)

    def elbo_alloc(self):
        K = self.K
        normPinit = gammaln(self.K * self.startAlpha) \
            - self.K * gammaln(self.startAlpha)

        normQinit = gammaln(np.sum(self.startTheta)) \
            - np.sum(gammaln(self.startTheta))

        normPtrans = K * gammaln(K * self.transAlpha + self.kappa) - \
            self.K * (self.K - 1) * gammaln(self.transAlpha) - \
            self.K * gammaln(self.transAlpha + self.kappa)

        normQtrans = np.sum(gammaln(np.sum(self.transTheta, axis=1))) \
            - np.sum(gammaln(self.transTheta))

        return normPinit + normPtrans - normQinit - normQtrans

    def to_dict(self):
        if self.inferType == 'EM':
            return dict(startPi=self.startPi,
                        transPi=self.transPi)
        elif self.inferType.count('VB') > 0:
            return dict(startTheta=self.startTheta,
                        transTheta=self.transTheta)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        if self.inferType.count('VB') > 0:
            self.startTheta = myDict['startTheta']
            self.transTheta = myDict['transTheta']
        elif self.inferType == 'EM':
            self.startPi = myDict['startPi']
            self.transPi = myDict['transPi']

    def get_prior_dict(self):
        return dict(startAlpha=self.startAlpha,
                    transAlpha=self.transAlpha,
                    kappa=self.kappa,
                    K=self.K)
