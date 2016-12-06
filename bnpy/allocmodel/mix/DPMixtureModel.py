'''
Bayesian nonparametric mixture model with Dirichlet process prior.
'''

import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil
from bnpy.util import gammaln, digamma, EPS
from bnpy.util.StickBreakUtil import beta2rho
from bnpy.util.SparseRespUtil import sparsifyLogResp
from bnpy.util.SparseRespStatsUtil import calcSparseRlogR, calcSparseMergeRlogR
from bnpy.util.ShapeUtil import toCArray, as1D
ELBOTermDimMap = dict(
    Hresp='K',
)


def calcELBO(**kwargs):
    """ Calculate ELBO objective for provided model state.
    """
    Llinear = calcELBO_LinearTerms(**kwargs)
    Lnon = calcELBO_NonlinearTerms(**kwargs)
    if 'todict' in kwargs and kwargs['todict']:
        assert isinstance(Llinear, dict)
        Llinear.update(Lnon)
        return Llinear
    return Lnon + Llinear


def calcELBO_LinearTerms(SS=None,
                         N=None,
                         eta1=None, eta0=None, ElogU=None, Elog1mU=None,
                         gamma1=1.0, gamma0=None,
                         afterGlobalStep=0, todict=0, **kwargs):
    """ Calculate ELBO objective terms that are linear in suff stats.
    """
    if SS is not None:
        N = SS.N
    K = N.size
    Lglobal = K * c_Beta(gamma1, gamma0) - c_Beta(eta1, eta0)
    if afterGlobalStep:
        if todict:
            return dict(Lalloc=Lglobal)
        return Lglobal
    # Slack term only needed when not immediately after a global step.
    N0 = convertToN0(N)
    if ElogU is None or Elog1mU is None:
        ElogU, Elog1mU = calcBetaExpectations(eta1, eta0)
    Lslack = np.inner(N + gamma1 - eta1, ElogU) + \
        np.inner(N0 + gamma0 - eta0, Elog1mU)
    if todict:
        return dict(Lalloc=Lglobal)
    return Lglobal + Lslack


def calcELBO_NonlinearTerms(SS=None, LP=None,
                            resp=None, Hresp=None,
                            returnMemoizedDict=0, todict=0, **kwargs):
    """ Calculate ELBO objective terms non-linear in suff stats.
    """
    if Hresp is None:
        if SS is not None and SS.hasELBOTerm('Hresp'):
            Hresp = SS.getELBOTerm('Hresp')
        else:
            Hresp = calcHrespFromLP(LP=LP, resp=resp)
    if returnMemoizedDict:
        return dict(Hresp=Hresp)
    Lentropy = np.sum(Hresp)
    if SS is not None and SS.hasAmpFactor():
        Lentropy *= SS.ampF
    if todict:
        return dict(Lentropy=Lentropy)
    return Lentropy

def calcHrespFromLP(LP=None, resp=None):
    if LP is not None and 'spR' in LP:
        nnzPerRow = LP['nnzPerRow']
        if nnzPerRow > 1:
            # Handles multiply by -1 already
            Hresp = calcSparseRlogR(**LP) 
            assert np.all(np.isfinite(Hresp))
        else:
            Hresp = 0.0
    else:
        if LP is not None and 'resp' in LP:
            resp = LP['resp']
        Hresp = -1 * NumericUtil.calcRlogR(resp)
    return Hresp

def calcELBOGain_NonlinearTerms(beforeSS=None, afterSS=None):
    """ Compute gain in ELBO score by transition from before to after values.
    """
    L_before = beforeSS.getELBOTerm('Hresp').sum()
    L_after = afterSS.getELBOTerm('Hresp').sum()
    return L_after - L_before


def convertToN0(N):
    """ Convert count vector to vector of "greater than" counts.

    Parameters
    -------
    N : 1D array, size K
        each entry k represents the count of items assigned to comp k.

    Returns
    -------
    N0 : 1D array, size K
        each entry k gives the total count of items at index above k
        N0[k] = np.sum(N[k:])

    Example
    -------
    >>> convertToN0([1., 3., 7., 2])
    array([ 12.,   9.,   2.,   0.])
    """
    N = np.asarray(N)
    N0 = np.zeros_like(N)
    N0[:-1] = N[::-1].cumsum()[::-1][1:]
    return N0


def c_Beta(eta1, eta0):
    ''' Evaluate cumulant function of Beta distribution

    Parameters
    -------
    eta1 : 1D array, size K
        represents ON pseudo-count parameter of the Beta
    eta0 : 1D array, size K
        represents OFF pseudo-count parameter of the Beta

    Returns
    -------
    c : float
        = \sum_k c_B(eta1[k], eta0[k])
    '''
    return np.sum(gammaln(eta1 + eta0) - gammaln(eta1) - gammaln(eta0))


def c_Beta_ReturnVec(eta1, eta0):
    ''' Evaluate cumulant of Beta distribution for vector of parameters

    Parameters
    -------
    eta1 : 1D array, size K
        represents ON pseudo-count parameter of the Beta
    eta0 : 1D array, size K
        represents OFF pseudo-count parameter of the Beta

    Returns
    -------
    cvec : 1D array, size K
    '''
    return gammaln(eta1 + eta0) - gammaln(eta1) - gammaln(eta0)


def calcBetaExpectations(eta1, eta0):
    ''' Evaluate expected value of log u under Beta(u | eta1, eta0)

    Returns
    -------
    ElogU : 1D array, size K
    Elog1mU : 1D array, size K
    '''
    digammaBoth = digamma(eta0 + eta1)
    ElogU = digamma(eta1) - digammaBoth
    Elog1mU = digamma(eta0) - digammaBoth
    return ElogU, Elog1mU


def calcCachedELBOGap_SinglePair(SS, kA, kB,
                                 delCompID=None, dtargetMinCount=None):
    """ Compute (lower bound on) gap in cacheable ELBO

    Returns
    ------
    gap : scalar
        L'_entropy - L_entropy >= gap
    """
    assert SS.hasELBOTerms()
    # Hvec : 1D array, size K
    Hvec = -1 * SS.getELBOTerm('ElogqZ')
    if delCompID is None:
        # Use bound - r log r >= 0
        gap = -1 * (Hvec[kA] + Hvec[kB])
    else:
        # Use bound - (1-r) log (1-r) >= r for small values of r
        assert delCompID == kA or delCompID == kB
        gap1 = -1 * Hvec[delCompID] - SS.N[delCompID]
        gap2 = -1 * (Hvec[kA] + Hvec[kB])
        gap = np.maximum(gap1, gap2)
    return gap


def calcCachedELBOTerms_SinglePair(SS, kA, kB, delCompID=None):
    """ Calculate all cached ELBO terms under proposed merge.
    """
    assert SS.hasELBOTerms()
    # Hvec : 1D array, size K
    Hvec = -1 * SS.getELBOTerm('ElogqZ')
    newHvec = np.delete(Hvec, kB)
    if delCompID is None:
        newHvec[kA] = 0
    else:
        assert delCompID == kA or delCompID == kB
        if delCompID == kA:
            newHvec[kA] = Hvec[kB]
        newHvec[kA] -= SS.N[delCompID]
        newHvec[kA] = np.maximum(0, newHvec[kA])
    return dict(ElogqZ=-1 * newHvec)


class DPMixtureModel(AllocModel):

    """ Nonparametric mixture model with K active components.

    Attributes
    ----------
    * inferType : string {'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    * K : int
        number of components
    * gamma1 : float
        scalar pseudo-count of ON values
        used in Beta prior on stick-breaking lengths.
    * gamma0 : float
        scalar pseudo-count of OFF values
        used in Beta prior on stick-breaking lengths.

    Attributes : Variational learning
    ---------------------------------
    * eta1 : 1D array, size K
        Posterior ON parameters for Beta posterior factor q(u).
        eta1[k] > 0 for all k
    * eta0 : 1D array, size K
        Posterior OFF parameters for Beta posterior factor q(u).
        eta0[k] > 0 for all k

    Secondary Attributes for VB
    ---------------------------
    * ElogU : 1D array, size K
        Expected value E[log u[k]] under current q(u[k])
    * Elog1mU : 1D array, size K
        Expected value E[log 1 - u[k]] under current q(u[k])
    """

    def __init__(self, inferType, priorDict=None, **priorKwargs):
        if inferType == 'EM':
            raise ValueError('EM not supported.')
        self.inferType = inferType
        if priorDict is not None:
            self.set_prior(**priorDict)
        else:
            self.set_prior(**priorKwargs)
        self.K = 0

    def set_prior(self, gamma1=1.0, gamma0=5.0, **kwargs):
        self.gamma1 = gamma1
        self.gamma0 = gamma0

    def set_helper_params(self):
        ''' Set dependent attributes given primary global params.

        This means storing digamma function evaluations.
        '''
        self.ElogU, self.Elog1mU = calcBetaExpectations(self.eta1, self.eta0)

        # Calculate expected mixture weights E[ log \beta_k ]
        # Using copy() allows += without modifying ElogU
        self.Elogbeta = self.ElogU.copy()
        self.Elogbeta[1:] += self.Elog1mU[:-1].cumsum()

    def get_active_comp_probs(self):
        ''' Get vector of appearance probabilities for each active comp.

        Returns
        -------
        beta : 1D array, size K
            beta[k] gives probability of comp. k under this model.
        '''
        Eu = self.eta1 / (self.eta1 + self.eta0)
        Ebeta = Eu.copy()
        Ebeta[1:] *= np.cumprod(1.0 - Eu[:-1])
        return Ebeta

    def get_keys_for_memoized_local_params(self):
        ''' Return LP field names required for warm starts of local step
        '''
        return list()

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Compute local parameters for provided data.

        Args
        ----
        Data : :class:`bnpy.data.DataObj`
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
        return calcLocalParams(
            Data, LP, Elogbeta=self.Elogbeta, **kwargs)

    def selectSubsetLP(self, Data, LP, relIDs):
        ''' Make subset of provided local params for certain data items.

        Returns
        ------
        LP : dict
            New local parameter dict for subset of data, with fields
            * resp : 2D array, size Nsubset x K
        '''
        resp = LP['resp'][relIDs].copy()
        relLP = dict(resp=resp)
        return relLP

    def fillSubsetLP(self, Data, LP, targetLP, targetIDs=[]):
        ''' Replace subset of local params with provided updated values.

        Returns
        ------
        LP : dict, with fields
            * resp : 2D array, size N x K
        '''
        targetK = targetLP['resp'].shape[-1]
        curK = LP['resp'].shape[-1]
        Kx = targetK - curK
        N = LP['resp'].shape[0]
        if Kx > 0:
            resp = np.zeros((N, targetK))
            resp[:, :curK] = LP['resp']
        else:
            resp = LP['resp']
        resp[targetIDs] = targetLP['resp']
        newLP = dict(resp=resp)
        return newLP

    def calcMergeTermsFromSeparateLP(
            self, Data=None, LPa=None, SSa=None,
            LPb=None, SSb=None, mUIDPairs=None):
        ''' Compute merge terms for case of expansion LP proposals.

        Returns
        -------
        Mdict : dict, with fields
        * Hresp
        '''
        M = len(mUIDPairs)
        m_Hresp = np.zeros(M)
        for m, (uidA, uidB) in enumerate(mUIDPairs):
            kA = SSa.uid2k(uidA)
            kB = SSb.uid2k(uidB)
            respAB = LPa['resp'][:, kA] + LPb['resp'][:, kB]
            m_Hresp[m] = -1 * NumericUtil.calcRlogR(respAB)
        assert m_Hresp.min() > -1e-9
        return dict(Hresp=m_Hresp)

    def get_global_suff_stats(self, Data, LP,
                              **kwargs):
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
        doPrecompMergeEntropy : boolean flag
            indicates whether to precompute ELBO terms in advance
            for certain merge candidates.

        Returns
        -------
        SS : SuffStatBag with K components
            Summarizes for this mixture model, with fields
            * N : 1D array, size K
                N[k] = expected number of items assigned to comp k

            Also has optional ELBO field when precompELBO is True
            * ElogqZ : 1D array, size K
                Vector of entropy contributions from each comp.
                ElogqZ[k] = \sum_{n=1}^N resp[n,k] log resp[n,k]

            Also has optional Merge field when precompMergeELBO is True
            * ElogqZ : 2D array, size K x K
                Each term is scalar entropy of merge candidate
        '''
        return calcSummaryStats(Data, LP, **kwargs)

    def forceSSInBounds(self, SS):
        ''' Enforce known bounds on SS fields for numerical stability.

        Post Condition for SS fields
        --------
        N : will have no entries below zero

        Post Condition for SS ELBO fields
        --------
        ElogqZ : will have no entries above zero
        '''
        np.maximum(SS.N, 0, out=SS.N)
        if SS.hasELBOTerm('ElogqZ'):
            Hvec = SS.getELBOTerm('ElogqZ')
            Hmax = Hvec.max()
            assert Hmax < 1e-10  # should be all negative
            if Hmax > 0:  # fix numerical errors to force entropy negative
                np.minimum(Hvec, 0, out=Hvec)
        if SS.hasMergeTerm('ElogqZ'):
            Hmat = SS.getMergeTerm('ElogqZ')
            Hmax = Hmat.max()
            assert Hmax < 1e-10  # should be all negative
            if Hmax > 0:
                np.minimum(Hmat, 0, out=Hmat)

    def update_global_params_VB(self, SS, **kwargs):
        """ Update eta1, eta0 to optimize the ELBO objective.

        Post Condition for VB
        -------
        eta1 and eta0 set to valid posterior for SS.K components.
        """
        N = SS.getCountVec()
        self.K = SS.K
        eta1 = self.gamma1 + N
        eta0 = self.gamma0 + convertToN0(N)
        self.eta1 = eta1
        self.eta0 = eta0
        self.set_helper_params()

    def update_global_params_soVB(self, SS, rho, **kwargs):
        """ Update eta1, eta0 to optimize stochastic ELBO objective.

        Post Condition for VB
        -------
        eta1 and eta0 set to valid posterior for SS.K components.
        """
        N = SS.getCountVec()
        assert self.K == SS.K
        eta1 = self.gamma1 + N
        eta0 = self.gamma0 + convertToN0(N)
        self.eta1 = rho * eta1 + (1 - rho) * self.eta1
        self.eta0 = rho * eta0 + (1 - rho) * self.eta0
        self.set_helper_params()

    def init_global_params(self, Data, K=0, **kwargs):
        """ Initialize global parameters to reasonable default values.

        Post Condition for VB
        -------
        eta1 and eta0 set to valid K vector.
        """
        self.setParamsFromCountVec(K, np.ones(K))

    def set_global_params(self, hmodel=None, K=None,
                          beta=None,
                          eta1=None, eta0=None, **kwargs):
        """ Set global parameters to provided values.

        Post Condition for EM
        -------
        w set to valid vector with K components.

        Post Condition for VB
        -------
        eta1/eta0 set to define valid posterior over K components.
        """
        if hmodel is not None:
            self.setParamsFromHModel(hmodel)
        elif beta is not None:
            self.setParamsFromBeta(K, beta=beta)
        elif eta1 is not None:
            self.K = int(K)
            self.eta1 = eta1
            self.eta0 = eta0
            self.set_helper_params()
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

        Post Condition for VB
        ---------
        Attributes eta1, eta0 are set so q(beta) equals its posterior
        given count vector N.
        """
        self.K = int(K)

        if N is None:
            N = 1.0 * np.ones(K)
        assert N.ndim == 1
        assert N.size == K
        self.eta1 = self.gamma1 + N
        self.eta0 = self.gamma0 + convertToN0(N)
        self.set_helper_params()

    def setParamsFromBeta(self, K, beta=None):
        """ Set params to reasonable values given comp probabilities.

        Parameters
        --------
        K : int
            number of components
        beta : 1D array, size K. optional, default=[1 1 1 1 ... 1]
            probability of each component

        Post Condition for VB
        ---------
        Attribute eta1, eta0 is set so q(beta) has properties:
        * mean of (nearly) beta, allowing for some small remaining mass.
        * moderate variance.
        """
        if beta is None:
            beta = 1.0 / K * np.ones(K)
        assert beta.ndim == 1
        assert beta.size == K
        assert np.sum(beta) < 1.0
        beta = beta / np.sum(beta)
        self.K = int(K)

        # Append in small remaining/leftover mass
        betaRem = np.minimum(1.0 / (2 * K), 0.05)
        betaWithRem = np.hstack([beta * (1.0 - betaRem), betaRem])

        theta = self.K * betaWithRem
        self.eta1 = theta[:-1].copy()
        self.eta0 = theta[::-1].cumsum()[::-1][1:]
        self.set_helper_params()

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
        if hasattr(hmodel.allocModel, 'eta1'):        
            self.eta1 = hmodel.allocModel.eta1.copy()
            self.eta0 = hmodel.allocModel.eta0.copy()
        elif hasattr(hmodel.allocModel, 'rho'):
            rho = hmodel.allocModel.rho
            omega = hmodel.allocModel.omega
            self.eta1 = omega * rho
            self.eta0 = omega * (1-rho)
        else:
            beta = hmodel.allocModel.get_active_comp_probs()
            self.setParamsFromBeta(K=beta.size, beta=beta)
        self.set_helper_params()

    def calc_evidence(self, Data, SS, LP=None, todict=0, **kwargs):
        """ Calculate ELBO objective function value for provided state.

        Returns
        -------
        L : float
        """
        return calcELBO(SS=SS, LP=LP,
                        eta1=self.eta1, eta0=self.eta0,
                        ElogU=self.ElogU, Elog1mU=self.Elog1mU,
                        gamma1=self.gamma1, gamma0=self.gamma0,
                        todict=todict,
                        **kwargs)

    def calcELBO_LinearTerms(self, **kwargs):
        ''' Compute sum of ELBO terms that are linear/const wrt suff stats

        Returns
        -------
        L : float
        '''
        return calcELBO_LinearTerms(eta1=self.eta1, eta0=self.eta0,
                                    gamma1=self.gamma1, gamma0=self.gamma0,
                                    **kwargs)

    def calcELBO_NonlinearTerms(self, **kwargs):
        ''' Compute sum of ELBO terms that are NONlinear wrt suff stats

        Returns
        -------
        L : float
        '''
        return calcELBO_NonlinearTerms(**kwargs)

    def calcHardMergeEntropyGap(self, SS, kA, kB):
        ''' Calc scalar improvement in entropy for merge of kA, kB
        '''
        Hmerge = SS.getMergeTerm('ElogqZ')
        Hcur = SS.getELBOTerm('ElogqZ')
        if Hmerge.ndim == 1:
            gap = Hcur[kB] - Hmerge[kB]
        else:
            gap = - Hmerge[kA, kB] + Hcur[kA] + Hcur[kB]
        return gap

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate scalar improvement in ELBO for merge of kA, kB

        For speed, use one of
        * calcHardMergeGapFast
        * calcHardMergeGapFastSinglePair.

        Does *not* include the entropy term for soft assignments.

        Returns
        -------
        L : float
            difference of partial ELBO functions
        '''
        cPrior = c_Beta(self.gamma1, self.gamma0)
        cB = c_Beta(self.eta1[kB], self.eta0[kB])

        gap = cB - cPrior
        # Add terms for changing kA to kA+kB
        gap += c_Beta(self.eta1[kA], self.eta0[kA]) \
            - c_Beta(self.eta1[kA] + SS.N[kB], self.eta0[kA] - SS.N[kB])

        # Add terms for each index kA+1, kA+2, ... kB-1
        # where only \gamma_0 has changed
        for k in xrange(kA + 1, kB):
            a1 = self.eta1[k]
            a0old = self.eta0[k]
            a0new = self.eta0[k] - SS.N[kB]
            gap += c_Beta(a1, a0old) - c_Beta(a1, a0new)
        return gap

    def calcHardMergeGapFast(self, SS, kA, kB):
        ''' Calculate scalar improvement in ELBO for merge of kA, kB

            Returns
            -------
            gap : float
        '''
        if not hasattr(self, 'cPrior'):
            self.cPrior = c_Beta(self.gamma1, self.gamma0)
        if not hasattr(self, 'cBetaCur'):
            self.cBetaCur = c_Beta_ReturnVec(self.eta1, self.eta0)
        if not hasattr(self, 'cBetaNewB') \
           or not (hasattr(self, 'kB') and self.kB == kB):
            self.kB = kB
            self.cBetaNewB = c_Beta_ReturnVec(self.eta1[:kB],
                                              self.eta0[:kB] - SS.N[kB])
        cDiff_A = self.cBetaCur[kA] \
            - c_Beta(self.eta1[kA] + SS.N[kB], self.eta0[kA] - SS.N[kB])
        cDiff_AtoB = np.sum(self.cBetaCur[kA + 1:kB] - self.cBetaNewB[kA + 1:])
        gap = self.cBetaCur[kB] - self.cPrior + cDiff_A + cDiff_AtoB
        return gap

    def calcHardMergeGapFastSinglePair(self, SS, kA, kB):
        ''' Calculate scalar improvement in ELBO for merge of kA, kB

            Returns
            -------
            gap : float
        '''
        if not hasattr(self, 'cPrior'):
            self.cPrior = c_Beta(self.gamma1, self.gamma0)
        if not hasattr(self, 'cBetaCur'):
            self.cBetaCur = c_Beta_ReturnVec(self.eta1, self.eta0)

        cBetaNew_AtoB = c_Beta_ReturnVec(self.eta1[kA + 1:kB],
                                         self.eta0[kA + 1:kB] - SS.N[kB])
        cDiff_A = self.cBetaCur[kA] \
            - c_Beta(self.eta1[kA] + SS.N[kB], self.eta0[kA] - SS.N[kB])
        cDiff_AtoB = np.sum(self.cBetaCur[kA + 1:kB] - cBetaNew_AtoB)
        gap = self.cBetaCur[kB] - self.cPrior + cDiff_A + cDiff_AtoB
        return gap

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calc matrix of improvement in ELBO for all possible pairs of comps
        '''
        Gap = np.zeros((SS.K, SS.K))
        for kB in xrange(1, SS.K):
            for kA in xrange(0, kB):
                Gap[kA, kB] = self.calcHardMergeGapFast(SS, kA, kB)
        if hasattr(self, 'cBetaNewB'):
            del self.cBetaNewB
            del self.kB
        if hasattr(self, 'cPrior'):
            del self.cPrior
        if hasattr(self, 'cBetaCur'):
            del self.cBetaCur
        return Gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc matrix of improvement in ELBO for all possible pairs of comps
        '''
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGapFastSinglePair(SS, kA, kB)
        if hasattr(self, 'cPrior'):
            del self.cPrior
        if hasattr(self, 'cBetaCur'):
            del self.cBetaCur
        return Gaps

    def calcCachedELBOTerms_SinglePair(self, SS, kA, kB, delCompID=None):
        """ Compute ELBO terms after merge of kA, kB

        Returns
        -------
        ELBOTerms : dict
            Key/value pairs are field name (str) and array
        """
        return calcCachedELBOTerms_SinglePair(SS, kA, kB, delCompID=delCompID)

    def calcCachedELBOGap_SinglePair(self, SS, kA, kB,
                                     delCompID=None):
        """ Compute (lower bound on) gap in cacheable ELBO

        Returns
        ------
        gap : scalar
            L'_entropy - L_entropy >= gap
        """
        return calcCachedELBOGap_SinglePair(SS, kA, kB, delCompID=delCompID)

    def get_info_string(self):
        ''' Returns one-line human-readable terse description of this object
        '''
        msgPattern = 'DP mixture with K=%d. Concentration gamma0= %.2f'
        return msgPattern % (self.K, self.gamma0)

    def to_dict(self):
        return dict(eta1=self.eta1, eta0=self.eta0)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.eta1 = myDict['eta1']
        self.eta0 = myDict['eta0']
        if self.eta0.ndim == 0:
            self.eta0 = self.eta1[np.newaxis]
        if self.eta0.ndim == 0:
            self.eta0 = self.eta0[np.newaxis]
        self.set_helper_params()

    def get_prior_dict(self):
        return dict(gamma1=self.gamma1,
                    gamma0=self.gamma0,
                    K=self.K,
                    )

    def make_hard_asgn_local_params(self, LP):
        ''' Convert soft assignments to hard for provided local params

            Returns
            --------
            LP : local params dict, with new fields
                 Z : 1D array, size N
                        Z[n] is an integer in range {0, 1, 2, ... K-1}
                 resp : 2D array, size N x K+1 (with final column empty)
                        resp[n,k] = 1 iff Z[n] == k
        '''
        LP['Z'] = np.argmax(LP['resp'], axis=1)
        K = LP['resp'].shape[1]
        LP['resp'].fill(0)
        for k in xrange(K):
            LP['resp'][LP['Z'] == k, k] = 1
        return LP

    def removeEmptyComps_SSandLP(self, SS, LP):
        ''' Remove all parameters related to empty components from SS and LP

            Returns
            --------
            SS : bnpy SuffStatBag
            LP : dict for local params
        '''
        badks = np.flatnonzero(SS.N[:-1] < 1)
        # Remove in order, from largest index to smallest
        for k in badks[::-1]:
            SS.removeComp(k)
            mask = LP['Z'] > k
            LP['Z'][mask] -= 1
        if 'resp' in LP:
            del LP['resp']
        return SS, LP

    def insertEmptyCompAtLastIndex_SSandLP(self, SS, LP):
        ''' Create empty component and insert last in order into SS

            Returns
            --------
            SS
            LP
        '''
        SS.insertEmptyComps(1)
        return SS, LP

    def sample_local_params(self, obsModel, Data, SS, LP, PRNG, **algParams):
        ''' Sample local assignments of all data items to components
        '''
        Z = LP['Z']
        # Iteratively sample data allocations
        for dataindex in xrange(Data.nObs):
            x = Data.X[dataindex]

            # de-update current assignment and suff stats
            kcur = Z[dataindex]
            SS.N[kcur] -= 1
            obsModel.decrementSS(SS, kcur, x)

            SS, LP = self.removeEmptyComps_SSandLP(SS, LP)

            doKeepFinalCompEmpty = SS.K < algParams['Kmax']
            if SS.N[-1] > 0 and doKeepFinalCompEmpty:
                SS, LP = self.insertEmptyCompAtLastIndex_SSandLP(SS, LP)

            # Calculate probs
            alloc_prob = self.getConditionalProbVec_Unnorm(
                SS, doKeepFinalCompEmpty)
            pvec = obsModel.calcPredProbVec_Unnorm(SS, x)
            pvec *= alloc_prob
            psum = np.sum(pvec)

            if np.isnan(psum) or psum <= 0:
                print pvec
                print psum
                raise ValueError('BAD VALUES FOR PROBS!')

            pvec /= psum
            # sample new allocation
            knew = PRNG.choice(SS.K, p=pvec)

            # update with new assignment
            SS.N[knew] += 1
            obsModel.incrementSS(SS, knew, x)
            Z[dataindex] = knew

        LP['Z'] = Z
        print ' '.join(['%.1f' % (x) for x in SS.N])
        return LP, SS

    def getConditionalProbVec_Unnorm(self, SS, doKeepFinalCompEmpty):
        ''' Returns a K vector of positive values \propto p(z_i|z_-i)
        '''
        if doKeepFinalCompEmpty:
            assert SS.N[-1] == 0
            return np.hstack([SS.N[:-1], self.gamma0])
        else:
            return np.hstack([SS.N[:-1], np.maximum(SS.N[-1], self.gamma0)])

    def calcMargLik(self, SS):
        ''' Calculate marginal likelihood of assignments, summed over all comps
        '''
        mask = SS.N > 0
        Nvec = SS.N[mask]
        K = Nvec.size
        return gammaln(self.gamma0) \
            + K * np.log(self.gamma0) \
            + np.sum(gammaln(Nvec)) \
            - gammaln(np.sum(Nvec) + self.gamma0)

    def getSerializableParamsForLocalStep(self):
        """ Get compact dict of params for parallel local step.

        Returns
        -------
        Info : dict
        """
        return dict(inferType=self.inferType,
                    Elogbeta=self.Elogbeta,
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
        return ShMem

    def getLocalAndSummaryFunctionHandles(self):
        """ Get function handles for local step and summary step

        Useful for parallelized algorithms.

        Returns
        -------
        calcLocalParams : f handle
        calcSummaryStats : f handle
        """
        return calcLocalParams, calcSummaryStats

    # .... end class DPMixtureModel


def calcLocalParams(Data, LP, Elogbeta=None, nnzPerRowLP=None, **kwargs):
    ''' Compute local parameters for each data item.

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
    lpr += Elogbeta
    K = LP['E_log_soft_ev'].shape[1]
    if nnzPerRowLP and (nnzPerRowLP > 0 and nnzPerRowLP < K):
        # SPARSE Assignments
        LP['spR'] = sparsifyLogResp(lpr, nnzPerRow=nnzPerRowLP)
        assert np.all(np.isfinite(LP['spR'].data))
        LP['nnzPerRow'] = nnzPerRowLP
    else:
        # DENSE Assignments
        # Calculate exp in numerically stable manner (first subtract the max)
        #  perform this in-place so no new allocations occur
        NumericUtil.inplaceExpAndNormalizeRows(lpr)
        LP['resp'] = lpr
    return LP


def calcSummaryStats(Data, LP,
                     doPrecompEntropy=False,
                     doPrecompMergeEntropy=False, mPairIDs=None,
                     mergePairSelection=None,
                     trackDocUsage=False,
                     **kwargs):
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
    doPrecompMergeEntropy : boolean flag
        indicates whether to precompute ELBO terms in advance
        for certain merge candidates.

    Returns
    -------
    SS : SuffStatBag with K components
        Summarizes for this mixture model, with fields
        * N : 1D array, size K
            N[k] = expected number of items assigned to comp k

        Also has optional ELBO field when precompELBO is True
        * ElogqZ : 1D array, size K
            Vector of entropy contributions from each comp.
            ElogqZ[k] = \sum_{n=1}^N resp[n,k] log resp[n,k]

        Also has optional Merge field when precompMergeELBO is True
        * ElogqZ : 2D array, size K x K
            Each term is scalar entropy of merge candidate
    '''
    if mPairIDs is not None and len(mPairIDs) > 0:
        M = len(mPairIDs)
    else:
        M = 0
    if 'resp' in LP:
        Nvec = np.sum(LP['resp'], axis=0)
        K = Nvec.size
    else:
        # Sparse assignment case
        Nvec = as1D(toCArray(LP['spR'].sum(axis=0)))
        K = LP['spR'].shape[1]

    if hasattr(Data, 'dim'):
        SS = SuffStatBag(K=K, D=Data.dim, M=M)
    else:
        SS = SuffStatBag(K=K, D=Data.vocab_size, M=M)
    SS.setField('N', Nvec, dims=('K'))
    if doPrecompEntropy:
        Mdict = calcELBO_NonlinearTerms(LP=LP, returnMemoizedDict=1)
        if type(Mdict['Hresp']) == float:
            # SPARSE HARD ASSIGNMENTS
            SS.setELBOTerm('Hresp', Mdict['Hresp'], dims=None)
        else:
            SS.setELBOTerm('Hresp', Mdict['Hresp'], dims=('K',))

    if doPrecompMergeEntropy:
        m_Hresp = None
        if 'resp' in LP:
            m_Hresp = -1 * NumericUtil.calcRlogR_specificpairs(
                LP['resp'], mPairIDs)
        elif 'spR' in LP:
            if LP['nnzPerRow'] > 1:
                m_Hresp = calcSparseMergeRlogR(
                    spR_csr=LP['spR'],
                    nnzPerRow=LP['nnzPerRow'], 
                    mPairIDs=mPairIDs)
        else:
            raise ValueError("Need resp or spR in LP")
        if m_Hresp is not None:
            assert m_Hresp.size == len(mPairIDs)
            SS.setMergeTerm('Hresp', m_Hresp, dims=('M'))
    if trackDocUsage:
        Usage = np.sum(LP['resp'] > 0.01, axis=0)
        SS.setSelectionTerm('DocUsageCount', Usage, dims='K')

    return SS
