'''
Bayesian parametric mixture model with finite number of components K.
'''
from builtins import *
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil
from bnpy.util import logsumexp
from bnpy.util import gammaln, digamma
from bnpy.util.SparseRespUtil import sparsifyLogResp
from .DPMixtureModel import calcSummaryStats, calcHrespFromLP

class FiniteMixtureModel(AllocModel):

    """ Parametric mixture model with finite number of components K

    Attributes
    -------
    * inferType : string {'EM', 'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    * K : int
        number of components
    * gamma : float
        scalar symmetric Dirichlet prior on mixture weights

    Attributes for EM
    --------
    * w : 1D array, size K
        estimated mixture weights for each component
        w[k] > 0 for all k, sum of vector w is equal to one

    Attributes for VB
    ---------
    * theta : 1D array, size K
        Estimated parameters for Dirichlet posterior over mix weights
        theta[k] > 0 for all k
    * Elogw : 1D array, size K
        Expected value E[ log w[k] ] for each component
        This is a deterministic function of theta
    """

    def __init__(self, inferType, priorDict=dict()):
        self.inferType = inferType
        self.set_prior(**priorDict)
        self.K = 0

    def set_prior(self, gamma=1.0, **kwargs):
        self.gamma = float(gamma)
        if self.gamma < 1.0 and self.inferType == 'EM':
            raise ValueError("Cannot perform MAP inference if param gamma < 1")

    def get_active_comp_probs(self):
        ''' Get vector of appearance probabilities for each active comp.

            Returns
            -------
            beta : 1D array, size K
                beta[k] gives probability of comp. k under this model.
        '''
        if self.inferType == 'EM':
            return self.w
        else:
            return self.theta / np.sum(self.theta)

    def get_keys_for_memoized_local_params(self):
        ''' Return LP field names required for warm starts of local step
        '''
        return list()

    def calc_local_params(self, Data, LP, nnzPerRowLP=0, **kwargs):
        ''' Compute local parameters for each data item and component.

        Parameters
        -------
        Data : bnpy.data.DataObj subclass

        LP : dict
            Local parameters as key-value string/array pairs
            * E_log_soft_ev : 2D array, N x K
                E_log_soft_ev[n,k] = log p(data obs n | comp k)

        Returns
        -------
        LP : dict
            Local parameters, with updated fields
            * resp : 2D array, size N x K array
                Posterior responsibility each comp has for each item
                resp[n, k] = p(z[n] = k | x[n])
        '''
        lpr = LP['E_log_soft_ev']
        K = lpr.shape[1]
        if self.inferType.count('EM') > 0:
            # Using point estimates, for EM algorithm
            lpr += np.log(self.w)
            if nnzPerRowLP and (nnzPerRowLP > 0 and nnzPerRowLP < K):
                # SPARSE Assignments
                LP['nnzPerRow'] = nnzPerRowLP
                LP['spR'] = sparsifyLogResp(lpr, nnzPerRow=nnzPerRowLP)
                assert np.all(np.isfinite(LP['spR'].data))
            else:
                lprPerItem = logsumexp(lpr, axis=1)
                lpr -= lprPerItem[:, np.newaxis]
                np.exp(lpr, out=lpr)
                LP['resp'] = lpr
                LP['evidence'] = lprPerItem.sum()
        else:
            # Full Bayesian approach, for VB or GS algorithms
            lpr += self.Elogw
            if nnzPerRowLP and (nnzPerRowLP > 0 and nnzPerRowLP < K):
                # SPARSE Assignments
                LP['nnzPerRow'] = nnzPerRowLP
                LP['spR'] = sparsifyLogResp(lpr, nnzPerRow=nnzPerRowLP)
                assert np.all(np.isfinite(LP['spR'].data))
            else:
                # DENSE Assignments
                # Calculate exp in numerically safe way,
                # in-place so no new allocations occur
                NumericUtil.inplaceExpAndNormalizeRows(lpr)
                LP['resp'] = lpr
                assert np.allclose(lpr.sum(axis=1), 1)
        return LP

    def get_global_suff_stats(self, Data, LP, **kwargs):
        ''' Calculate sufficient statistics for global updates.

        Parameters
        -------
        Data : bnpy data object
        LP : local param dict with fields
            resp : Data.nObs x K array,
                where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag
            indicates whether to precompute ELBO terms in advance
            used for memoized learning algorithms (moVB)

        Returns
        -------
        SS : SuffStatBag with K components
            Summarizes for this mixture model, with fields
            * N : 1D array, size K
                N[k] = expected number of items assigned to comp k

            Also has optional ELBO field when precompELBO is True
            * Hresp : 1D array, size K
                Vector of entropy contributions from each comp.
                Hresp[k] = \sum_{n=1}^N resp[n,k] log resp[n,k]
        '''
        SS = calcSummaryStats(Data, LP, **kwargs)
        return SS
        """
        Nvec = np.sum(LP['resp'], axis=0)
        if hasattr(Data, 'dim'):
            SS = SuffStatBag(K=Nvec.size, D=Data.dim)
        elif hasattr(Data, 'vocab_size'):
            SS = SuffStatBag(K=Nvec.size, D=Data.vocab_size)

        SS.setField('N', Nvec, dims=('K'))
        if doPrecompEntropy is not None:
            ElogqZ_vec = self.E_logqZ(LP)
            SS.setELBOTerm('ElogqZ', ElogqZ_vec, dims=('K'))
        return SS
        """

    def update_global_params_EM(self, SS, **kwargs):
        """ Update attribute w to optimize the ELBO ML/MAP objective.

        Post Condition for EM
        -------
        w set to valid vector of size SS.K.
        """
        w = SS.N + (self.gamma / SS.K) - 1.0  # MAP estimate. Requires gamma>1
        self.w = w / w.sum()
        self.K = SS.K

    def update_global_params_VB(self, SS, **kwargs):
        """ Update attribute theta to optimize the ELBO objective.

        Post Condition for VB
        -------
        theta set to valid posterior for SS.K components.
        """
        self.theta = self.gamma / SS.K + SS.N
        self.Elogw = digamma(self.theta) - digamma(self.theta.sum())
        self.K = SS.K

    def update_global_params_soVB(self, SS, rho, **kwargs):
        """ Update attribute theta to optimize stochastic ELBO objective.

        Post Condition for VB
        -------
        theta set to valid posterior for SS.K components.
        """
        thetaStar = self.gamma / SS.K + SS.N
        self.theta = rho * thetaStar + (1 - rho) * self.theta
        self.Elogw = digamma(self.theta) - digamma(self.theta.sum())
        self.K = SS.K

    def init_global_params(self, Data=None, K=0, **kwargs):
        """ Initialize global parameters to reasonable default values.

        Post Condition for EM
        -------
        w set to valid K vector.

        Post Condition for VB
        -------
        theta set to valid K vector.
        """
        self.setParamsFromCountVec(K, np.ones(K))

    def set_global_params(self, hmodel=None, K=None,
                          w=None, beta=None,
                          theta=None, **kwargs):
        """ Set global parameters to provided values.

        Post Condition for EM
        -------
        w set to valid vector with K components.

        Post Condition for VB
        -------
        theta set to define valid posterior over K components.
        """
        if hmodel is not None:
            self.setParamsFromHModel(hmodel)
        elif beta is not None:
            self.setParamsFromBeta(K, beta=beta)
        elif w is not None:
            self.setParamsFromBeta(K, beta=w)
        elif theta is not None and self.inferType.count('VB'):
            self.K = int(K)
            self.theta = theta
            self.Elogw = digamma(self.theta) - digamma(self.theta.sum())
        else:
            raise ValueError("Unrecognized set_global_params args")

    def setParamsFromCountVec(self, K, N=None):
        """ Set params to reasonable values given counts for each comp.

        Parameters
        --------
        K : int
            number of components
        N : 1D array, size K. optional, default=[1 1 1 1 ... 1]
            size of each component

        Post Condition for EM
        --------
        Attribute w is set to posterior mean given provided vector N.
        Default behavior sets w to uniform distribution.

        Post Condition for VB
        ---------
        Attribute theta is set so q(w) equals posterior given vector N.
        Default behavior has q(w) with mean of uniform and moderate variance.
        """
        if N is None:
            N = 1.0 * np.ones(K)
        assert N.ndim == 1
        assert N.size == K

        self.K = int(K)
        if self.inferType == 'EM':
            self.w = N + (self.gamma / K)
            self.w /= self.w.sum()
        else:
            self.theta = N + self.gamma / K
            self.Elogw = digamma(self.theta) - digamma(self.theta.sum())

    def setParamsFromBeta(self, K, beta=None):
        """ Set params to reasonable values given comp probabilities.

        Parameters
        --------
        K : int
            number of components
        beta : 1D array, size K. optional, default=[1 1 1 1 ... 1]
            probability of each component

        Post Condition for EM
        --------
        Attribute w is set to posterior mean given provided vector N.
        Default behavior sets w to uniform distribution.

        Post Condition for VB
        ---------
        Attribute theta is set so q(w) has mean of beta and moderate variance.
        """
        if beta is None:
            beta = 1.0 / K * np.ones(K)
        assert beta.ndim == 1
        assert beta.size == K

        self.K = int(K)
        if self.inferType == 'EM':
            self.w = beta.copy()
        else:
            self.theta = self.K * beta
            self.Elogw = digamma(self.theta) - digamma(self.theta.sum())

    def setParamsFromHModel(self, hmodel):
        """ Set parameters exactly as in provided HModel object.

        Parameters
        ------
        hmodel : bnpy.HModel
            The model to copy parameters from.

        Post Condition
        ------
        w or theta will be set exactly equal to hmodel's allocModel.
        """
        self.K = hmodel.allocModel.K
        if self.inferType == 'EM':
            self.w = hmodel.allocModel.w.copy()
        else:
            self.theta = hmodel.allocModel.theta.copy()
            self.Elogw = digamma(self.theta) - digamma(self.theta.sum())

    def calc_evidence(self, Data, SS, LP, todict=False, **kwargs):
        """ Calculate ELBO objective function value for provided state.

        Returns
        -------
        L : scalar
            represents sum of all terms in objective
        """
        if self.inferType == 'EM':
            return LP['evidence'] + self.log_pdf_dirichlet(self.w)
        elif self.inferType.count('VB') > 0:
            L_alloc = Lalloc(SS=SS, theta=self.theta, Elogw=self.Elogw)
            if SS.hasELBOTerm('Hresp'):
                L_entropy = np.sum(SS.getELBOTerm('Hresp'))
            else:
                L_entropy = np.sum(calcHrespFromLP(LP=LP))
            if SS.hasAmpFactor():
                L_entropy *= SS.ampF
            return L_entropy + L_alloc
            '''
            evW = self.E_logpW() - self.E_logqW()
            if SS.hasELBOTerm('Hresp'):
                Hresp = np.sum(SS.getELBOTerm('Hresp'))
            else:
                Hresp = np.sum(calcHrespFromLP(LP=LP))
            if SS.hasAmpFactor():
                Hresp *= SS.ampF
            evZ = self.E_logpZ(SS) +
            return evZ + evW
            '''
        else:
            raise NotImplementedError(
                'Unrecognized inferType ' + self.inferType)

    def E_logpZ(self, SS):
        ''' Bishop PRML eq. 10.72
        '''
        return np.inner(SS.N, self.Elogw)

    def E_logpW(self):
        ''' Bishop PRML eq. 10.73
        '''
        return gammaln(self.gamma) \
            - self.K * gammaln(self.gamma/self.K) + \
            (self.gamma / self.K - 1) * self.Elogw.sum()

    def E_logqW(self):
        ''' Bishop PRML eq. 10.76
        '''
        return gammaln(self.theta.sum()) - gammaln(self.theta).sum() \
            + np.inner((self.theta - 1), self.Elogw)

    def log_pdf_dirichlet(self, wvec=None, avec=None):
        ''' Return scalar log probability for Dir(wvec | avec)
        '''
        if wvec is None:
            wvec = self.w
        if avec is None:
            avec = (self.gamma / self.K) * np.ones(self.K)
        logC = gammaln(np.sum(avec)) - np.sum(gammaln(avec))
        return logC + np.sum((avec - 1.0) * np.log(wvec))

    def get_info_string(self):
        ''' Returns one-line human-readable terse description of this object
        '''
        msgPattern = 'Finite mixture model. Dir prior param %.2f'
        return msgPattern % (self.gamma)

    def to_dict(self):
        if self.inferType == 'EM':
            return dict(w=self.w)
        elif self.inferType.count('VB') > 0:
            return dict(theta=self.theta)
        elif self.inferType.count('GS') > 0:
            return dict(theta=self.theta)
        return dict()

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        if self.inferType == 'EM':
            self.w = myDict['w']
        else:
            self.theta = myDict['theta']
            self.Elogw = digamma(self.theta) - digamma(self.theta.sum())

    def get_prior_dict(self):
        return dict(gamma=self.gamma, K=self.K)

    # ----    Sampler functions
    def sample_local_params(self, obsModel, Data, SS, LP, PRNG, **kwargs):
        ''' Sample local assignments for each data item

        Returns
        --------
        LP : dict
            Local parameters, with updated fields
            * Z : 1D array, size N
                Z[n] = k iff item n is assigned to component k
        '''
        Z = LP['Z']
        # Iteratively sample data allocations
        for dataindex in range(Data.nObs):
            x = Data.X[dataindex]

            # de-update current assignment and suff stats
            kcur = Z[dataindex]
            SS.N[kcur] -= 1
            obsModel.decrementSS(SS, kcur, x)

            # Calculate probs
            alloc_prob = self.getConditionalProbVec_Unnorm(SS)
            pvec = obsModel.calcPredProbVec_Unnorm(SS, x)
            pvec *= alloc_prob
            pvec /= np.sum(pvec)

            # sample new allocation
            knew = PRNG.choice(SS.K, p=pvec)

            # update with new assignment
            SS.N[knew] += 1
            obsModel.incrementSS(SS, knew, x)
            Z[dataindex] = knew

        LP['Z'] = Z
        return LP, SS

    def getConditionalProbVec_Unnorm(self, SS):
        ''' Returns a K vector of positive values \propto p(z_i|z_-i)
        '''
        return SS.N + self.gamma / SS.K

    def calcMargLik(self, SS):
        ''' Calculate marginal likelihood of assignments, summed over all comps
        '''
        theta = self.gamma / SS.K + SS.N
        cPrior = gammaln(self.gamma) - SS.K * gammaln(self.gamma / SS.K)
        cPost = gammaln(np.sum(theta)) - np.sum(gammaln(theta))
        return cPrior - cPost

def c_Dir(tvec):
    return gammaln(tvec.sum()) - gammaln(tvec).sum()

def Lalloc(Nvec=None, SS=None, gamma=0.5, theta=None, Elogw=None):
    assert theta is not None
    K = theta.size
    if Elogw is None:
        Elogw = digamma(theta) - digamma(theta.sum())
    if Nvec is None:
        Nvec = SS.N
    Lalloc = c_Dir(gamma/K * np.ones(K)) - c_Dir(theta)
    Lalloc_slack = np.inner(Nvec + gamma/K - theta, Elogw)
    return Lalloc + Lalloc_slack
