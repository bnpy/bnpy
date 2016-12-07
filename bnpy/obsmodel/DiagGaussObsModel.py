import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, toCArray, np2flatstr
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray
from bnpy.util.SparseRespStatsUtil import calcRXX_withDenseResp, calcSpRXX
from AbstractObsModel import AbstractObsModel
from GaussObsModel import createECovMatFromUserInput

class DiagGaussObsModel(AbstractObsModel):

    ''' Diagonal gaussian data generation model for real vectors.

    Attributes for Prior (Normal-Wishart)
    --------
    nu : float
        degrees of freedom
    beta : 1D array, size D
        scale parameters that set mean of parameter sigma
    m : 1D array, size D
        mean of the parameter mu
    kappa : float
        scalar precision on parameter mu

    Attributes for k-th component of EstParams (EM point estimates)
    ---------
    mu[k] : 1D array, size D
    sigma[k] : 1D array, size D

    Attributes for k-th component of Post (VB parameter)
    ---------
    nu[k] : float
    beta[k] : 1D array, size D
    m[k] : 1D array, size D
    kappa[k] : float

    '''

    def __init__(self, inferType='EM', D=0, min_covar=None,
                 Data=None, **PriorArgs):
        ''' Initialize bare obsmodel with valid prior hyperparameters.

        Resulting object lacks either EstParams or Post,
        which must be created separately (see init_global_params).
        '''
        if Data is not None:
            self.D = Data.dim
        else:
            self.D = int(D)
        self.K = 0
        self.inferType = inferType
        self.min_covar = min_covar
        self.Prior = createParamBagForPrior(Data, D=D, **PriorArgs)
        self.Cache = dict()


    def get_mean_for_comp(self, k=None):
        if hasattr(self, 'EstParams'):
            return self.EstParams.mu[k]
        elif k is None or k == 'prior':
            return self.Prior.m
        else:
            return self.Post.m[k]

    def get_covar_mat_for_comp(self, k=None):
        if hasattr(self, 'EstParams'):
            return np.diag(self.EstParams.sigma[k])
        elif k is None or k == 'prior':
            return self._E_CovMat()
        else:
            return self._E_CovMat(k)

    def get_name(self):
        return 'DiagGauss'

    def get_info_string(self):
        return 'Gaussian with diagonal covariance.'

    def get_info_string_prior(self):
        return getStringSummaryOfPrior(self.Prior)

    def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                     mu=None, sigma=None, Sigma=None,
                     **kwargs):
        ''' Create EstParams ParamBag with fields mu, Sigma
        '''
        self.ClearCache()
        if obsModel is not None:
            self.EstParams = obsModel.EstParams.copy()
            self.K = self.EstParams.K
            return

        if LP is not None and Data is not None:
            SS = self.calcSummaryStats(Data, None, LP)

        if SS is not None:
            self.updateEstParams(SS)
        else:
            K = mu.shape[0]
            if Sigma is not None:
                assert Sigma.ndim == 3
                sigma = np.empty((Sigma.shape[0], Sigma.shape[1]))
                for k in xrange(K):
                    sigma[k] = np.diag(Sigma[k])
            assert sigma.ndim == 2
            self.EstParams = ParamBag(K=K, D=mu.shape[1])
            self.EstParams.setField('mu', mu, dims=('K', 'D'))
            self.EstParams.setField('sigma', sigma, dims=('K', 'D'))
        self.K = self.EstParams.K

    def setEstParamsFromPost(self, Post):
        ''' Convert from Post (nu, beta, m, kappa) to EstParams (mu, Sigma),
             each EstParam is set to its posterior mean.
        '''
        self.EstParams = ParamBag(K=Post.K, D=self.D)
        mu = Post.m.copy()
        sigma = Post.beta / (Post.nu - 2)[:, np.newaxis]
        self.EstParams.setField('mu', mu, dims=('K', 'D'))
        self.EstParams.setField('sigma', sigma, dims=('K', 'D'))
        self.K = self.EstParams.K

    def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                       **param_kwargs):
        ''' Set attribute Post to provided values.
        '''
        self.ClearCache()
        if obsModel is not None:
            if hasattr(obsModel, 'Post'):
                self.Post = obsModel.Post.copy()
                self.K = self.Post.K
            else:
                self.setPostFromEstParams(obsModel.EstParams)
            return

        if LP is not None and Data is not None:
            SS = self.calcSummaryStats(Data, None, LP)

        if SS is not None:
            self.updatePost(SS)
        else:
            if 'D' not in param_kwargs:
                param_kwargs['D'] = self.D
            self.Post = packParamBagForPost(**param_kwargs)
        self.K = self.Post.K

    def setPostFromEstParams(self, EstParams, Data=None, N=None):
        ''' Set attribute Post based on values in EstParams.
        '''
        K = EstParams.K
        D = EstParams.D
        if Data is not None:
            N = Data.nObsTotal

        N = np.asarray(N, dtype=np.float)
        if N.ndim == 0:
            N = float(N) / K * np.ones(K)

        nu = self.Prior.nu + N
        beta = np.zeros((K, D))
        beta = (nu - 2)[:, np.newaxis] * EstParams.sigma
        m = EstParams.mu.copy()
        kappa = self.Prior.kappa + N

        self.Post = ParamBag(K=K, D=D)
        self.Post.setField('nu', nu, dims=('K'))
        self.Post.setField('beta', beta, dims=('K', 'D'))
        self.Post.setField('m', m, dims=('K', 'D'))
        self.Post.setField('kappa', kappa, dims=('K'))
        self.K = self.Post.K

    def calcSummaryStats(self, Data, SS, LP, **kwargs):
        ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        return calcSummaryStats(Data, SS, LP, **kwargs)

    def forceSSInBounds(self, SS):
        ''' Force count vector N to remain positive
        '''
        np.maximum(SS.N, 0, out=SS.N)

    def incrementSS(self, SS, k, x):
        SS.x[k] += x
        SS.xx[k] += np.square(x)

    def decrementSS(self, SS, k, x):
        SS.x[k] -= x
        SS.xx[k] -= np.square(x)

    def calcLogSoftEvMatrix_FromEstParams(self, Data, **kwargs):
        ''' Compute log soft evidence matrix for Dataset under EstParams.

        Returns
        ---------
        L : 2D array, N x K
        '''
        K = self.EstParams.K
        L = np.zeros((Data.nObs, K))
        for k in xrange(K):
            L[:, k] = - 0.5 * self.D * LOGTWOPI \
                - 0.5 * np.sum(np.log(self.EstParams.sigma[k])) \
                - 0.5 * self._mahalDist_EstParam(Data.X, k)
        return L

    def _mahalDist_EstParam(self, X, k):
        ''' Calculate distance to every row of matrix X

            Args
            -------
            X : 2D array, size N x D

            Returns
            ------
            dist : 1D array, size N
        '''
        Xdiff = X - self.EstParams.mu[k]
        np.square(Xdiff, out=Xdiff)
        dist = np.sum(Xdiff / self.EstParams.sigma[k], axis=1)
        return dist

    def updateEstParams_MaxLik(self, SS):
        ''' Update attribute EstParams for all comps given suff stats.

        Update uses the maximum likelihood objective for point estimation.

        Post Condition
        ---------
        Attributes K and EstParams updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
            self.EstParams = ParamBag(K=SS.K, D=SS.D)

        mu = SS.x / SS.N[:, np.newaxis]
        sigma = self.min_covar \
            + SS.xx / SS.N[:, np.newaxis] \
            - np.square(mu)

        self.EstParams.setField('mu', mu, dims=('K', 'D'))
        self.EstParams.setField('sigma', sigma, dims=('K', 'D'))
        self.K = SS.K

    def updateEstParams_MAP(self, SS):
        ''' Update attribute EstParams for all comps given suff stats.

        Update uses the MAP objective for point estimation.

        Post Condition
        ---------
        Attributes K and EstParams updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
            self.EstParams = ParamBag(K=SS.K, D=SS.D)

        Prior = self.Prior
        nu = Prior.nu + SS.N
        kappa = Prior.kappa + SS.N
        PB = Prior.beta + Prior.kappa * np.square(Prior.m)

        m = np.empty((SS.K, SS.D))
        beta = np.empty((SS.K, SS.D))
        for k in xrange(SS.K):
            km_x = Prior.kappa * Prior.m + SS.x[k]
            m[k] = 1.0 / kappa[k] * km_x
            beta[k] = PB + SS.xx[k] - 1.0 / kappa[k] * np.square(km_x)

        mu, sigma = MAPEstParams_inplace(nu, beta, m, kappa)
        self.EstParams.setField('mu', mu, dims=('K', 'D'))
        self.EstParams.setField('sigma', sigma, dims=('K', 'D'))
        self.K = SS.K

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Update uses the variational objective.

        Post Condition
        ---------
        Attributes K and Post updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D)

        nu, beta, m, kappa = self.calcPostParams(SS)
        self.Post.setField('nu', nu, dims=('K'))
        self.Post.setField('kappa', kappa, dims=('K'))
        self.Post.setField('m', m, dims=('K', 'D'))
        self.Post.setField('beta', beta, dims=('K', 'D'))
        self.K = SS.K

    def calcPostParams(self, SS):
        ''' Calc posterior parameters for all comps given suff stats.

        Returns
        --------
        nu : 1D array, size K
        beta : 2D array, size K x D
        m : 2D array, size K x D
        kappa : 1D array, size K
        '''
        Prior = self.Prior
        nu = Prior.nu + SS.N
        kappa = Prior.kappa + SS.N
        m = (Prior.kappa * Prior.m + SS.x) / kappa[:, np.newaxis]
        beta = Prior.beta + SS.xx \
            + Prior.kappa * np.square(Prior.m) \
            - kappa[:, np.newaxis] * np.square(m)
        return nu, beta, m, kappa

    def calcPostParamsForComp(self, SS, kA, kB=None):
        ''' Calc posterior parameters for specific comp given suff stats.

        Returns
        --------
        nu : positive scalar
        beta : 1D array, size D
        m : 1D array, size D
        kappa : positive scalar
        '''
        return calcPostParamsFromSSForComp(SS, kA, kB)
        '''
        if kB is None:
            SN = SS.N[kA]
            Sx = SS.x[kA]
            Sxx = SS.xx[kA]
        else:
            SN = SS.N[kA] + SS.N[kB]
            Sx = SS.x[kA] + SS.x[kB]
            Sxx = SS.xx[kA] + SS.xx[kB]
        Prior = self.Prior
        nu = Prior.nu + SN
        kappa = Prior.kappa + SN
        m = (Prior.kappa * Prior.m + Sx) / kappa
        beta = Prior.beta + Sxx \
            + Prior.kappa * np.square(Prior.m) \
            - kappa * np.square(m)
        return nu, beta, m, kappa
        '''

    def updatePost_stochastic(self, SS, rho):
        ''' Update attribute Post for all comps given suff stats

        Update uses the stochastic variational formula.

        Post Condition
        ---------
        Attributes K and Post updated in-place.
        '''
        assert hasattr(self, 'Post')
        assert self.Post.K == SS.K
        self.ClearCache()

        self.convertPostToNatural()
        nu, b, km, kappa = self.calcNaturalPostParams(SS)
        Post = self.Post
        Post.nu[:] = (1 - rho) * Post.nu + rho * nu
        Post.b[:] = (1 - rho) * Post.b + rho * b
        Post.km[:] = (1 - rho) * Post.km + rho * km
        Post.kappa[:] = (1 - rho) * Post.kappa + rho * kappa
        self.convertPostToCommon()

    def calcNaturalPostParams(self, SS):
        ''' Calc natural posterior params for all comps given suff stats.


        Returns
        --------
        nu : 1D array, size K
        b : 2D array, size K x D
        km : 2D array, size K x D
        kappa : 1D array, size K
        '''
        Prior = self.Prior
        nu = Prior.nu + SS.N
        kappa = Prior.kappa + SS.N
        km = Prior.kappa * Prior.m + SS.x
        b = Prior.beta + Prior.kappa * np.square(Prior.m) + SS.xx
        return nu, b, km, kappa

    def convertPostToNatural(self):
        ''' Convert current posterior params from common to natural form
        '''
        Post = self.Post
        assert hasattr(Post, 'nu')
        assert hasattr(Post, 'kappa')
        km = Post.m * Post.kappa[:, np.newaxis]
        b = Post.beta + (np.square(km) / Post.kappa[:, np.newaxis])
        Post.setField('km', km, dims=('K', 'D'))
        Post.setField('b', b, dims=('K', 'D'))

    def convertPostToCommon(self):
        ''' Convert current posterior params from natural to common form
        '''
        Post = self.Post
        assert hasattr(Post, 'nu')
        assert hasattr(Post, 'kappa')
        if hasattr(Post, 'm'):
            Post.m[:] = Post.km / Post.kappa[:, np.newaxis]
        else:
            m = Post.km / Post.kappa[:, np.newaxis]
            Post.setField('m', m, dims=('K', 'D'))

        if hasattr(Post, 'beta'):
            Post.beta[:] = Post.b - \
                (np.square(Post.km) / Post.kappa[:, np.newaxis])
        else:
            beta = Post.b - (np.square(Post.km) / Post.kappa[:, np.newaxis])
            Post.setField('beta', beta, dims=('K', 'D'))

    def calcLogSoftEvMatrix_FromPost(self, Data, **kwargs):
        return calcLogSoftEvMatrix_FromPost(Data, Post=self.Post)

    def zzz_calcLogSoftEvMatrix_FromPost(self, Data, **kwargs):
        ''' Calculate expected log soft ev matrix under Post.

        Returns
        ------
        L : 2D array, size N x K
        '''
        K = self.Post.K
        L = np.zeros((Data.nObs, K))
        for k in xrange(K):
            L[:, k] = - 0.5 * self.D * LOGTWOPI \
                + 0.5 * np.sum(self.GetCached('E_logL', k)) \
                - 0.5 * self._mahalDist_Post(Data.X, k)
        return L

    def _mahalDist_Post(self, X, k):
        ''' Calc expected mahalonobis distance from comp k to each data atom

            Returns
            --------
            distvec : 1D array, size nObs
                   distvec[n] gives E[ \Lam (x-\mu)^2 ] for comp k
        '''
        Xdiff = X - self.Post.m[k]
        np.square(Xdiff, out=Xdiff)
        dist = np.dot(Xdiff, self.Post.nu[k] / self.Post.beta[k])
        dist += self.D / self.Post.kappa[k]
        return dist

    def calcELBO_Memoized(self, SS, returnVec=0, afterMStep=False, **kwargs):
        return calcELBOFromSSAndPost(SS=SS, Post=self.Post, Prior=self.Prior)

    
    def _zzz_calcELBO_Memoized(self, SS, returnVec=0, afterMStep=False, **kwargs):
        """ Calculate obsModel's objective using suff stats SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag
        afterMStep : boolean flag
            if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        obsELBO : scalar float
            Equal to E[ log p(x) + log p(phi) - log q(phi)]
        """
        elbo = np.zeros(SS.K)
        Post = self.Post
        Prior = self.Prior
        for k in xrange(SS.K):
            elbo[k] = c_Diff(Prior.nu, Prior.beta, Prior.m, Prior.kappa,
                             Post.nu[k], Post.beta[
                                 k], Post.m[k], Post.kappa[k],
                             )
            if not afterMStep:
                aDiff = SS.N[k] + Prior.nu - Post.nu[k]
                bDiff = SS.xx[k] + Prior.beta \
                    + Prior.kappa * np.square(Prior.m) \
                    - Post.beta[k] \
                    - Post.kappa[k] * np.square(Post.m[k])
                cDiff = SS.x[k] + Prior.kappa * Prior.m \
                    - Post.kappa[k] * Post.m[k]
                dDiff = SS.N[k] + Prior.kappa - Post.kappa[k]
                elbo[k] += 0.5 * aDiff * np.sum(self._E_logL(k)) \
                    - 0.5 * np.inner(bDiff, self._E_L(k)) \
                    + np.inner(cDiff, self.GetCached('E_Lmu', k)) \
                    - 0.5 * dDiff * np.sum(self.GetCached('E_muLmu', k))
        
        elbo += -(0.5 * SS.D * LOGTWOPI) * SS.N
        if returnVec:
            return elbo
        return elbo.sum()
    

    def getDatasetScale(self, SS):
        ''' Get number of observed scalars in dataset from suff stats.

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
        '''
        return SS.N.sum() * SS.D

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate change in ELBO after a hard merge applied to this model

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
        '''
        gap, _, _ = calcHardMergeGapForPair(
            SS=SS, Prior=self.Prior, Post=self.Post, kA=kA, kB=kB)
        return gap        

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calculate change in ELBO for all possible hard merge pairs

        Returns
        ---------
        Gap : 2D array, size K x K, upper-triangular entries non-zero
              Gap[j,k] : scalar change in ELBO after merge of k into j
        '''
        Gap2D = np.zeros((SS.K, SS.K))
        cPrior = None
        cPost_K = [None for k in range(SS.K)]
        for kA in xrange(SS.K):
            for kB in xrange(kA + 1, SS.K):
                    Gap2D[kA, kB], cPost_K, cPrior = calcHardMergeGapForPair(
                        SS=SS, Post=self.Post, Prior=self.Prior, kA=kA, kB=kB,
                        cPrior=cPrior, cPost_K=cPost_K)
        return Gap2D

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
        '''
        Gaps = np.zeros(len(PairList))
        cPrior = None
        cPost_K = [None for k in range(SS.K)]
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii], cPost_K, cPrior = calcHardMergeGapForPair(
                SS=SS, Post=self.Post, Prior=self.Prior, kA=kA, kB=kB,
                cPrior=cPrior, cPost_K=cPost_K)
        return Gaps


    def calcLogMargLikForComp(self, SS, kA, kB=None, **kwargs):
        ''' Calc log marginal likelihood of data assigned to given component

        Args
        -------
        SS : bnpy suff stats object
        kA : integer ID of target component to compute likelihood for
        kB : (optional) integer ID of second component.
             If provided, we merge kA, kB into one component for calculation.
        Returns
        -------
        logM : scalar real
               logM = log p( data assigned to comp kA | Prior )
                      computed up to an additive constant
        '''
        nu, B, m, kappa = self.calcPostParamsForComp(SS, kA, kB)
        return -1 * c_Func(nu, B, m, kappa)

    def calcMargLik(self, SS):
        ''' Calc log marginal likelihood additively across all comps.

        Returns
        --------
        logM : scalar real
               logM = \sum_{k=1}^K log p( data assigned to comp k | Prior)
        '''
        return self.calcMargLik_CFuncForLoop(SS)

    def calcMargLik_CFuncForLoop(self, SS):
        Prior = self.Prior
        logp = np.zeros(SS.K)
        for k in xrange(SS.K):
            nu, beta, m, kappa = self.calcPostParamsForComp(SS, k)
            logp[k] = c_Diff(Prior.nu, Prior.beta, Prior.m, Prior.kappa,
                             nu, beta, m, kappa)
        return np.sum(logp) - 0.5 * np.sum(SS.N) * LOGTWOPI

    def calcPredProbVec_Unnorm(self, SS, x):
        ''' Calculate K-vector of positive entries \propto p( x | SS[k] )
        '''
        return self._calcPredProbVec_Fast(SS, x)

    def _calcPredProbVec_Naive(self, SS, x):
        nu, beta, m, kappa = self.calcPostParams(SS)
        pSS = SS.copy()
        pSS.N += 1
        pSS.x += x
        pSS.xx += np.square(x)
        pnu, pbeta, pm, pkappa = self.calcPostParams(pSS)
        logp = np.zeros(SS.K)
        for k in xrange(SS.K):
            logp[k] = c_Diff(nu[k], beta[k], m[k], kappa[k],
                             pnu[k], pbeta[k], pm[k], pkappa[k])
        return np.exp(logp - np.max(logp))

    def _calcPredProbVec_Fast(self, SS, x):
        p = np.zeros(SS.K)
        nu, beta, m, kappa = self.calcPostParams(SS)
        kbeta = beta
        kbeta *= ((kappa + 1) / kappa)[:, np.newaxis]
        base = np.square(x - m)
        base /= kbeta
        base += 1
        # logp : 2D array, size K x D
        logp = (-0.5 * (nu + 1))[:, np.newaxis] * np.log(base)
        logp += (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu))[:, np.newaxis]
        logp -= 0.5 * np.log(kbeta)

        # p : 1D array, size K
        p = np.sum(logp, axis=1)
        p -= np.max(p)
        np.exp(p, out=p)
        return p

    def _calcPredProbVec_ForLoop(self, SS, x):
        ''' For-loop version
        '''
        p = np.zeros(SS.K)
        for k in xrange(SS.K):
            nu, beta, m, kappa = self.calcPostParamsForComp(SS, k)
            kbeta = (kappa + 1) / kappa * beta
            base = np.square(x - m)
            base /= kbeta
            base += 1
            p_k = np.exp(gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu)) \
                * 1.0 / np.sqrt(kbeta) \
                * base ** (-0.5 * (nu + 1))
            p[k] = np.prod(p_k)
        return p

    def _Verify_calcPredProbVec(self, SS, x):
        ''' Verify that the predictive prob vector is correct,
              by comparing 3 very different implementations
        '''
        pA = self._calcPredProbVec_Fast(SS, x)
        pB = self._calcPredProbVec_Naive(SS, x)
        pC = self._calcPredProbVec_ForLoop(SS, x)
        pA /= np.sum(pA)
        pB /= np.sum(pB)
        pC /= np.sum(pC)
        assert np.allclose(pA, pB)
        assert np.allclose(pA, pC)

    def _E_CovMat(self, k=None):
        ''' Get expected value of Sigma under specified distribution.

        Returns
        --------
        E[ Sigma ] : 2D array, size DxD
        '''
        return np.diag(self._E_Cov(k))

    def _E_Cov(self, k=None):
        ''' Get expected value of sigma vector under specified distribution.

        Returns
        --------
        E[ sigma^2 ] : 1D array, size D
        '''
        if k is None:
            nu = self.Prior.nu
            beta = self.Prior.beta
        else:
            nu = self.Post.nu[k]
            beta = self.Post.beta[k]
        return beta / (nu - 2)

    def _E_logL(self, k=None):
        '''
        Returns
        -------
        E_logL : 1D array, size D
        '''
        if k == 'all':
            # retVec : K x D
            retVec = LOGTWO - np.log(self.Post.beta.copy())  # no strided!
            retVec += digamma(0.5 * self.Post.nu)[:, np.newaxis]
            return retVec
        elif k is None:
            nu = self.Prior.nu
            beta = self.Prior.beta
        else:
            nu = self.Post.nu[k]
            beta = self.Post.beta[k]
        return LOGTWO - np.log(beta) + digamma(0.5 * nu)

    def _E_L(self, k=None):
        '''
        Returns
        --------
        EL : 1D array, size D
        '''
        if k is None:
            nu = self.Prior.nu
            beta = self.Prior.beta
        else:
            nu = self.Post.nu[k]
            beta = self.Post.beta[k]
        return nu / beta

    def _E_Lmu(self, k=None):
        '''
        Returns
        --------
        ELmu : 1D array, size D
        '''
        if k is None:
            nu = self.Prior.nu
            beta = self.Prior.beta
            m = self.Prior.m
        else:
            nu = self.Post.nu[k]
            beta = self.Post.beta[k]
            m = self.Post.m[k]
        return (nu / beta) * m

    def _E_muLmu(self, k=None):
        ''' Calc expectation E[lam * mu^2]

        Returns
        --------
        EmuLmu : 1D array, size D
        '''
        if k is None:
            nu = self.Prior.nu
            kappa = self.Prior.kappa
            m = self.Prior.m
            beta = self.Prior.beta
        else:
            nu = self.Post.nu[k]
            kappa = self.Post.kappa[k]
            m = self.Post.m[k]
            beta = self.Post.beta[k]
        return 1.0 / kappa + (nu / beta) * (m * m)

    def getSerializableParamsForLocalStep(self):
        """ Get compact dict of params for local step.

        Returns
        -------
        Info : dict
        """
        if self.inferType == 'EM':
            raise NotImplementedError('TODO')
        return dict(inferType=self.inferType,
                    K=self.K,
                    D=self.D,
                    )

    def fillSharedMemDictForLocalStep(self, ShMem=None):
        """ Get dict of shared mem arrays needed for parallel local step.

        Returns
        -------
        ShMem : dict of RawArray objects
        """
        if ShMem is None:
            ShMem = dict()
        if 'nu' in ShMem:
            fillSharedMemArray(ShMem['nu'], self.Post.nu)
            fillSharedMemArray(ShMem['kappa'], self.Post.kappa)
            fillSharedMemArray(ShMem['m'], self.Post.m)
            fillSharedMemArray(ShMem['beta'], self.Post.beta)
            fillSharedMemArray(ShMem['E_logL'], self._E_logL('all'))

        else:
            ShMem['nu'] = numpyToSharedMemArray(self.Post.nu)
            ShMem['kappa'] = numpyToSharedMemArray(self.Post.kappa)
            # Post.m is strided, so we need to copy it to do shared mem.
            ShMem['m'] = numpyToSharedMemArray(self.Post.m.copy())
            ShMem['beta'] = numpyToSharedMemArray(self.Post.beta.copy())
            ShMem['E_logL'] = numpyToSharedMemArray(self._E_logL('all'))

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



    def calcSmoothedMu(self, X, W=None):
        ''' Compute smoothed estimate of mean of statistic xxT.

        Args
        ----
        X : 2D array, size N x D

        Returns
        -------
        Mu_1 : 2D array, size D
            Expected value of Var[ X[n,d] ]
        Mu_2 : 1D array, size D
            Expected value of Mean[ X[n] ]
        '''
        if X is None:
            Mu1 = self.Prior.beta / self.Prior.nu
            Mu2 = self.Prior.m
            return Mu1, Mu2

        if X.ndim == 1:
            X = X[np.newaxis,:]
        N, D = X.shape
        # Compute suff stats
        if W is None:
            sum_wxx = np.sum(np.square(X), axis=0)
            sum_wx = np.sum(X, axis=0)
            sum_w = X.shape[0]
        else:
            W = as1D(W)
            sum_wxx = np.dot(W, np.square(X))
            sum_wx = np.dot(W, X)
            sum_w = np.sum(W)

        post_kappa = self.Prior.kappa + sum_w
        post_m = (self.Prior.m * self.Prior.kappa + sum_wx) / post_kappa
        Mu_2 = post_m

        prior_kmm = self.Prior.kappa * (self.Prior.m * self.Prior.m)
        post_kmm = post_kappa * (post_m * post_m)
        post_beta = sum_wxx + self.Prior.beta + prior_kmm - post_kmm
        Mu_1 = post_beta / (self.Prior.nu + sum_w)

        assert Mu_1.ndim == 1
        assert Mu_1.shape == (D,)
        assert Mu_2.shape == (D,)
        return Mu_1, Mu_2

    def calcSmoothedBregDiv(
            self, X, Mu, W=None,
            eps=1e-10,
            smoothFrac=0.0,
            includeOnlyFastTerms=False,
            DivDataVec=None,
            returnDivDataVec=False,
            return1D=False,
            **kwargs):
        ''' Compute Bregman divergence between data X and clusters Mu.

        Smooth the data via update with prior parameters.

        Args
        ----
        X : 2D array, size N x D
        Mu : list of size K, or tuple

        Returns
        -------
        Div : 2D array, N x K
            Div[n,k] = smoothed distance between X[n] and Mu[k]
        '''
        # Parse X
        if X.ndim < 2:
            X = X[np.newaxis,:]
        assert X.ndim == 2
        N = X.shape[0]
        D = X.shape[1]
        # Parse Mu
        if isinstance(Mu, tuple):
            Mu = [Mu]
        assert isinstance(Mu, list)
        K = len(Mu)
        assert Mu[0][0].size == D
        assert Mu[0][1].size == D

        prior_x = self.Prior.m
        prior_varx = self.Prior.beta / (self.Prior.nu)
        VarX = eps * prior_varx

        Div = np.zeros((N, K))
        for k in xrange(K):
            muVar_k = Mu[k][0]
            muMean_k = Mu[k][1]
            logdet_MuVar_k = np.sum(np.log(muVar_k))
            squareDiff_X_Mu_k = np.square(X - muMean_k)
            tr_k = np.sum((VarX + squareDiff_X_Mu_k) / \
                muVar_k[np.newaxis,:], axis=1)
            Div[:,k] = \
                + 0.5 * logdet_MuVar_k \
                + 0.5 * tr_k
        # Only enter here if exactly computing Div,
        # If just need it up to additive constant, skip this part.
        if not includeOnlyFastTerms:
            if DivDataVec is None:
                # Compute DivDataVec : 1D array of size N
                # This is the per-row additive constant indep. of k. 
                DivDataVec = -0.5 * D * np.ones(N)
                logdet_VarX = np.sum(np.log(VarX))
                DivDataVec -= 0.5 * logdet_VarX
        
            Div += DivDataVec[:,np.newaxis]
        # Apply per-atom weights to divergences.
        if W is not None:
            assert W.ndim == 1
            assert W.size == N
            Div *= W[:,np.newaxis]
        # Verify divergences are strictly non-negative 
        if not includeOnlyFastTerms:
            minDiv = Div.min()
            if minDiv < 0:
                if minDiv < -1e-6:
                    raise AssertionError(
                        "Expected Div.min() to be positive or" + \
                        " indistinguishable from zero. Instead " + \
                        " minDiv=% .3e" % (minDiv))
                np.maximum(Div, 0, out=Div)
                minDiv = Div.min()
            assert minDiv >= 0
        if return1D:
            Div = Div[:,0]
        if returnDivDataVec:
            return Div, DivDataVec
        return Div


    def calcBregDivFromPrior(self, Mu, smoothFrac=0.0):
        ''' Compute Bregman divergence between Mu and prior mean.

        Returns
        -------
        Div : 1D array, size K
            Div[k] = distance between Mu[k] and priorMu
        '''
        assert isinstance(Mu, list)
        K = len(Mu)
        assert K >= 1
        assert Mu[0][0].ndim == 1
        assert Mu[0][1].ndim == 1
        D = Mu[0][0].size
        assert D == Mu[0][1].size

        priorMuVar = self.Prior.beta / self.Prior.nu
        priorMuMean = self.Prior.m

        priorN_ZMG = (1-smoothFrac) * self.Prior.nu
        priorN_FVG = (1-smoothFrac) * self.Prior.kappa

        Div_ZMG = np.zeros(K) # zero-mean gaussian
        Div_FVG = np.zeros(K) # fixed variance gaussian
 
        logdet_priorMuVar = np.sum(np.log(priorMuVar))
        for k in xrange(K):
            MuVar_k = Mu[k][0]
            MuMean_k = Mu[k][1]

            logdet_MuVar_k = np.sum(np.log(MuVar_k))

            Div_ZMG[k] = 0.5 * logdet_MuVar_k + \
                - 0.5 * logdet_priorMuVar + \
                + 0.5 * np.sum(priorMuVar / MuVar_k) + \
                - 0.5
            squareDiff = np.square(priorMuMean - MuMean_k)
            Div_FVG[k] = 0.5 * np.sum(squareDiff / MuVar_k)

        return priorN_ZMG * Div_ZMG + priorN_FVG * Div_FVG
    # .... end class


def MAPEstParams_inplace(nu, beta, m, kappa=0):
    ''' MAP estimate parameters mu, Sigma given Normal-Wishart hyperparameters
    '''
    D = m.size
    mu = m
    sigma = beta / (nu[:, np.newaxis] + 2)
    return mu, sigma


def c_Func(nu, beta, m, kappa):
    ''' Evaluate cumulant function at given params.

    Returns
    --------
    c : scalar real value of cumulant function at provided args
    '''
    D = m.size
    c1D = - 0.5 * LOGTWOPI \
        - 0.5 * LOGTWO * nu \
        - gammaln(0.5 * nu) \
        + 0.5 * np.log(kappa) \
        + 0.5 * nu * np.log(beta)
    return np.sum(c1D)


def c_Diff(nu1, beta1, m1, kappa1,
           nu2, beta2, m2, kappa2):
    ''' Evaluate difference of cumulant functions c(params1) - c(params2)

    May be more numerically stable than directly using c_Func
    to find the difference.

    Returns
    -------
    diff : scalar real value of the difference in cumulant functions
    '''
    cDiff = - 0.5 * LOGTWO * (nu1 - nu2) \
        - gammaln(0.5 * nu1) + gammaln(0.5 * nu2) \
        + 0.5 * (np.log(kappa1) - np.log(kappa2)) \
        + 0.5 * (nu1 * np.log(beta1) - nu2 * np.log(beta2))
    return np.sum(cDiff)


def calcSummaryStats(Data, SS, LP, **kwargs):
    ''' Calculate summary statistics for given dataset and local parameters

    Returns
    --------
    SS : SuffStatBag object, with K components.
    '''
    X = Data.X
    if 'resp' in LP:
        resp = LP['resp']
        K = resp.shape[1]
        # 1/2: Compute mean statistic
        S_x = dotATB(resp, X)
        # 2/2: Compute expected outer-product statistic
        S_xx = calcRXX_withDenseResp(resp, X)
    else:
        spR = LP['spR']
        K = spR.shape[1]
        # 1/2: Compute mean statistic
        S_x = spR.T * X
        # 2/2: Compute expected outer-product statistic
        S_xx = calcSpRXX(X=X, spR_csr=spR)
    if SS is None:
        SS = SuffStatBag(K=K, D=Data.dim)
    # Expected mean for each state k
    SS.setField('x', S_x, dims=('K', 'D'))
    # Expected sum-of-squares for each state k
    SS.setField('xx', S_xx, dims=('K', 'D'))
    # Expected count for each k
    #  Usually computed by allocmodel. But sometimes not (eg TopicModel)
    if not hasattr(SS, 'N'):
        if 'resp' in LP:
            SS.setField('N', LP['resp'].sum(axis=0), dims='K')
        else:
            SS.setField('N', as1D(toCArray(LP['spR'].sum(axis=0))), dims='K')
    return SS


def calcLocalParams(Dslice, **kwargs):
    ''' Calculate all local parameters for provided data (or slice of data).

    Returns
    -------
    LP : dict of local params
    with field:
        * E_log_soft_ev : 2D array, N x K
    '''
    L = calcLogSoftEvMatrix_FromPost(Dslice, **kwargs)
    LP = dict(E_log_soft_ev=L)
    return LP


def calcLogSoftEvMatrix_FromPost(Dslice,
        Post=None,
        nu=None,
        beta=None,
        m=None,
        kappa=None,
        E_logdetL_K=None,
        K=None,
        **kwargs):
    ''' Calculate expected log soft ev matrix for variational.

    Returns
    ------
    L : 2D array, size N x K
    '''
    if Post is not None:
        nu = Post.nu
        beta = Post.beta
        m = Post.m
        kappa = Post.kappa

    if K is None:
        K = nu.size
    assert K == nu.size
    assert K == kappa.size
    L = np.zeros((Dslice.nObs, K))
    for k in xrange(K):
        if E_logdetL_K is None:
            E_logdetL_k = E_logdetL(nu=nu[k], beta=beta[k])
        else:
            E_logdetL_k = E_logdetL_K[k]

        L[:, k] = - 0.5 * Dslice.dim * LOGTWOPI \
            + 0.5 * np.sum(E_logdetL_k) \
            - 0.5 * _mahalDist_Post(
                Dslice.X, 
                nu=nu[k], 
                beta=beta[k],
                m=m[k],
                kappa=kappa[k])
    return L


def _mahalDist_Post(X, beta=None, m=None, nu=None, kappa=None, **kwargs):
    ''' Calc expected mahalonobis distance from comp k to each data atom

    Returns
    --------
    distvec : 1D array, size nObs
           distvec[n] gives E[ \Lam (x-\mu)^2 ] for comp k
    '''
    D = int(m.size)
    Xdiff = X - m
    np.square(Xdiff, out=Xdiff)
    dist = np.dot(Xdiff, nu / beta)
    dist += D / kappa
    return dist

def calcPostParamsFromSS(
        SS=None, 
        Prior=None,
        Post=None,
        returnParamBag=True,
        **kwargs):
    ''' Calc updated posterior parameters for all clusters from suff stats

    Returns
    --------
    nu : 1D array, size K
    beta : 2D array, size K x D
    m : 2D array, size K x D
    kappa : 1D array, size K
    '''
    K = SS.K
    D = SS.D

    nu = Prior.nu + SS.N
    kappa = Prior.kappa + SS.N
    m = (Prior.kappa * Prior.m + SS.x) / kappa[:, np.newaxis]
    beta = Prior.beta + SS.xx \
        + Prior.kappa * np.square(Prior.m) \
        - kappa[:, np.newaxis] * np.square(m)

    if not returnParamBag:
        return nu, beta, m, kappa
    return packParamBagForPost(
        nu=nu, beta=beta, m=m, kappa=kappa, Post=Post)



def calcPostParamsFromSSForComp(
        SS=None, 
        kA=None, kB=None,
        Prior=None,
        **kwargs):
    ''' Calc updated posterior parameters for all clusters from suff stats

    Returns
    --------
    nu : positive scalar
    beta : 1D array, size D
    m : 1D array, size D
    kappa : positive scalar
    '''
    K = SS.K
    D = SS.D

    if kB is None:
        SS_N_k = SS.N[kA]
        SS_x_k = SS.x[kA]
        SS_xx_k = SS.xx[kA]
    else:
        SS_N_k = SS.N[kA] + SS.N[kB]
        SS_x_k = SS.x[kA] + SS.x[kB]
        SS_xx_k = SS.xx[kA] + SS.xx[kB]

    nu = Prior.nu + SS_N_k
    kappa = Prior.kappa + SS_N_k
    m = (Prior.kappa * Prior.m + SS_x_k) / kappa
    beta = Prior.beta + SS_xx_k \
        + Prior.kappa * np.square(Prior.m) \
        - kappa * np.square(m)
    return nu, beta, m, kappa

def calcELBOFromSSAndPost(
        SS,
        Post=None, Prior=None,
        returnVec=0, afterMStep=False, **kwargs):
    """ Calculate obsModel's objective using suff stats SS and Post.

    Args
    -------
    SS : bnpy SuffStatBag
    afterMStep : boolean flag
        if 1, elbo calculated assuming M-step just completed

    Returns
    -------
    obsELBO : scalar float
        Equal to E[ log p(x) + log p(phi) - log q(phi)]
    """
    elbo = np.zeros(SS.K)
    for k in xrange(SS.K):
        elbo[k] = c_Diff(
            Prior.nu, Prior.beta, Prior.m, Prior.kappa,
            Post.nu[k], Post.beta[k], Post.m[k], Post.kappa[k],
            )
        if not afterMStep:
            aDiff = SS.N[k] + Prior.nu - Post.nu[k]
            bDiff = SS.xx[k] + Prior.beta \
                + Prior.kappa * np.square(Prior.m) \
                - Post.beta[k] \
                - Post.kappa[k] * np.square(Post.m[k])
            cDiff = SS.x[k] + Prior.kappa * Prior.m \
                - Post.kappa[k] * Post.m[k]
            dDiff = SS.N[k] + Prior.kappa - Post.kappa[k]
            elbo[k] += \
                + 0.5 * aDiff * np.sum(
                    E_logdetL(
                        nu=Post.nu[k],
                        beta=Post.beta[k])) \
                - 0.5 * np.inner(bDiff, E_L(
                        nu=Post.nu[k],
                        beta=Post.beta[k])) \
                + np.inner(cDiff, E_Lm(
                        nu=Post.nu[k],
                        beta=Post.beta[k],
                        m=Post.m[k])) \
                - 0.5 * dDiff * np.sum(E_Lm2(
                        nu=Post.nu[k],
                        beta=Post.beta[k],
                        m=Post.m[k],
                        kappa=Post.kappa[k]))
    elbo += -(0.5 * SS.D * LOGTWOPI) * SS.N
    if returnVec:
        return elbo
    return elbo.sum()

def E_logdetL(nu=None, beta=None, **kwargs):
    '''
    '''
    return LOGTWO - np.log(beta) + digamma(0.5 * nu)

def E_L(nu=None, beta=None, **kwargs):
    '''
    '''
    return nu / beta

def E_Lm(nu=None, beta=None, m=None, **kwargs):
    ''' Expected value of np.dot(L, m)

    Returns
    -------
    E_Lm : 1D array, size D
    '''
    return (nu / beta) * m

def E_Lm2(nu=None, beta=None, m=None, kappa=None, **kwargs):
    ''' Expected value of np.dot(m.T, np.dot(L,m))

    Returns
    -------
    E_Lm2 : 1D array, size D
    '''
    return 1.0 / kappa + (nu / beta) * (m * m)

def E_CovMat(nu=None, beta=None):
    ''' Get expected value of covariance matrix

    Returns
    --------
    E[ diag(sigma^2) ] : 2D array, size D x D
    '''
    return np.diag(beta / (nu - 2))

def packParamBagForPost(
        nu=None,
        beta=None,
        m=None,
        kappa=None,
        D=None,
        Post=None,
        **kwargs):
    '''
    '''
    m = as2D(m)
    beta = as2D(beta)

    if D is None:
        D = m.shape[1]

    if m.shape[1] != D:
        m = m.T.copy()
    if beta.shape[1] != D:
        beta = beta.T.copy()
    K, _ = m.shape
    if Post is None:
        Post = ParamBag(K=K, D=D)
    else:
        assert isinstance(Post, ParamBag)
        assert Post.K == K
        assert Post.D == D        
    Post.setField('nu', as1D(nu), dims=('K'))
    Post.setField('beta', beta, dims=('K', 'D'))
    Post.setField('m', m, dims=('K', 'D'))
    Post.setField('kappa', as1D(kappa), dims=('K'))
    return Post

def getStringSummaryOfPrior(Prior):
    ''' Get human-readable summary of specific prior

    Returns
    -------
    s : str
    '''
    if Prior.D > 2:
        sfx = ' ...'
    else:
        sfx = ''

    msg = 'independent Gauss-Wishart prior on each dimension\n'
    msg += "Wishart params \n"
    msg += "  nu = %.3g %s\n" % (Prior.nu, sfx)
    msg += "beta = %s %s\n" % (np2flatstr(Prior.beta[:2]), sfx)
    msg += "Expectations\n"
    msg += 'E[  mean[k]] = \n%s%s\n' % (np2flatstr(Prior.m[:2]), sfx)
    msg += 'E[ covar[k]] = \n'
    S = E_CovMat(nu=Prior.nu, beta=Prior.beta[:2])
    msg += str(S) + sfx
    msg = msg.replace('\n', '\n  ')
    return msg

def createParamBagForPrior(
        Data, D=0, nu=0, beta=None,
        m=None, kappa=None,
        MMat='zero', ECovMat=None, sF=1.0,
        Prior=None,
        **kwargs):
    ''' Initialize ParamBag of parameters which specify prior.

    Returns
    -------
    Prior : ParamBag
    '''
    if Data is None:
        D = int(D)
    else:
        D = int(Data.dim)
    nu = np.maximum(nu, D + 2)
    kappa = np.maximum(kappa, 1e-8)
    if beta is None:
        if ECovMat is None or isinstance(ECovMat, str):
            ECovMat = createECovMatFromUserInput(D, Data, ECovMat, sF)
        beta = np.diag(ECovMat) * (nu - 2)
    else:
        if beta.ndim == 0:
            beta = np.asarray([beta], dtype=np.float)
    if m is None:
        if MMat == 'data':
            m = np.sum(Data.X, axis=0)
        else:
            m = np.zeros(D)
    elif m.ndim < 1:
        m = np.asarray([m], dtype=np.float)
    if Prior is None:
        Prior = ParamBag(K=0, D=D)
    assert Prior.D == D
    Prior.setField('nu', nu, dims=None)
    Prior.setField('kappa', kappa, dims=None)
    Prior.setField('m', m, dims=('D'))
    Prior.setField('beta', beta, dims=('D'))
    return Prior


def calcHardMergeGapForPair(
        SS=None, Prior=None, Post=None, kA=0, kB=1, 
        cPost_K=None,
        cPrior=None,
        ):
    ''' Compute difference in ELBO objective after merging two clusters

    Uses caching if desired

    Returns
    -------
    Ldiff : scalar
        difference in ELBO from merging cluster indices kA and kB
    '''
    if cPost_K is None:
        cPost_K = [None for k in range(SS.K)]

    if cPost_K[kA] is None:
        cA = c_Func(
            Post.nu[kA], Post.beta[kA],
            Post.m[kA], Post.kappa[kA])
    else:
        cA = cPost_K[kA]

    if cPost_K[kB] is None:
        cB = c_Func(
            Post.nu[kB], Post.beta[kB],
            Post.m[kB], Post.kappa[kB])
        cPost_K[kB] = cB
    else:
        cB = cPost_K[kB]

    if cPrior is None:
        cPrior = c_Func(
            Prior.nu, Prior.beta,
            Prior.m, Prior.kappa)
    cAB = c_Func(*calcPostParamsFromSSForComp(SS, kA, kB, Prior))
    return cA + cB - cPrior - cAB, cPost_K, cPrior
