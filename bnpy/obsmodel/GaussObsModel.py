import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D, toCArray
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray
from bnpy.util.SparseRespStatsUtil import calcSpRXXT
from AbstractObsModel import AbstractObsModel


class GaussObsModel(AbstractObsModel):

    ''' Full-covariance gaussian data generation model for real vectors.

    Attributes for Prior (Normal-Wishart)
    --------
    nu : float
        degrees of freedom
    B : 2D array, size D x D
        scale parameters that set mean of parameter sigma
    m : 1D array, size D
        mean of the parameter mu
    kappa : float
        scalar precision on parameter mu

    Attributes for k-th component of EstParams (EM point estimates)
    ---------
    mu[k] : 1D array, size D
    Sigma[k] : 2D array, size DxD

    Attributes for k-th component of Post (VB parameter)
    ---------
    nu[k] : float
    B[k] : 1D array, size D
    m[k] : 1D array, size D
    kappa[k] : float

    '''

    def __init__(self, inferType='EM', D=0, min_covar=None,
                 Data=None,
                 **PriorArgs):
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
        self.createPrior(Data, **PriorArgs)
        self.Cache = dict()

    def createPrior(self, Data, nu=0, B=None,
                    m=None, kappa=None,
                    MMat='zero',
                    ECovMat=None, sF=1.0, **kwargs):
        ''' Initialize Prior ParamBag attribute.

        Post Condition
        ------
        Prior expected covariance matrix set to match provided value.
        '''
        D = self.D
        nu = np.maximum(nu, D + 2)
        if B is None:
            if ECovMat is None or isinstance(ECovMat, str):
                ECovMat = createECovMatFromUserInput(D, Data, ECovMat, sF)
            B = ECovMat * (nu - D - 1)
        if B.ndim == 1:
            B = np.asarray([B], dtype=np.float)
        elif B.ndim == 0:
            B = np.asarray([[B]], dtype=np.float)
        if m is None:
            if MMat == 'data':
                m = np.mean(Data.X, axis=0)
            else:
                m = np.zeros(D)
        elif m.ndim < 1:
            m = np.asarray([m], dtype=np.float)
        kappa = np.maximum(kappa, 1e-8)
        self.Prior = ParamBag(K=0, D=D)
        self.Prior.setField('nu', nu, dims=None)
        self.Prior.setField('kappa', kappa, dims=None)
        self.Prior.setField('m', m, dims=('D'))
        self.Prior.setField('B', B, dims=('D', 'D'))

    def get_mean_for_comp(self, k=None):
        if hasattr(self, 'EstParams'):
            return self.EstParams.mu[k]
        elif k is None or k == 'prior':
            return self.Prior.m
        else:
            return self.Post.m[k]

    def get_covar_mat_for_comp(self, k=None):
        if hasattr(self, 'EstParams'):
            return self.EstParams.Sigma[k]
        elif k is None or k == 'prior':
            return self._E_CovMat()
        else:
            return self._E_CovMat(k)

    def get_name(self):
        return 'Gauss'

    def get_info_string(self):
        return 'Gaussian with full covariance.'

    def get_info_string_prior(self):
        msg = 'Gauss-Wishart on mean and covar of each cluster\n'
        if self.D > 2:
            sfx = ' ...'
        else:
            sfx = ''
        S = self._E_CovMat()[:2, :2]
        msg += 'E[  mean[k] ] = \n %s %s\n' % (str(self.Prior.m[:2]), sfx)
        msg += 'E[ covar[k] ] = \n'
        msg += str(S) + sfx
        msg = msg.replace('\n', '\n  ')
        return msg

    def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                     mu=None, Sigma=None,
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
            Sigma = as3D(Sigma)
            K, D, D2 = Sigma.shape
            mu = as2D(mu)
            if mu.shape[0] != K:
                mu = mu.T
            assert mu.shape[0] == K
            assert mu.shape[1] == D
            self.EstParams = ParamBag(K=K, D=D)
            self.EstParams.setField('mu', mu, dims=('K', 'D'))
            self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))
            self.K = self.EstParams.K

    def setEstParamsFromPost(self, Post):
        ''' Convert from Post (nu, B, m, kappa) to EstParams (mu, Sigma),
             each EstParam is set to its posterior mean.
        '''
        self.EstParams = ParamBag(K=Post.K, D=Post.D)
        mu = Post.m.copy()
        Sigma = Post.B / (Post.nu[:, np.newaxis, np.newaxis] - Post.D - 1)
        self.EstParams.setField('mu', mu, dims=('K', 'D'))
        self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))
        self.K = self.EstParams.K

    def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                       nu=0, B=0, m=0, kappa=0,
                       **kwargs):
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
            m = as2D(m)
            if m.shape[1] != self.D:
                m = m.T.copy()
            K, _ = m.shape
            self.Post = ParamBag(K=K, D=self.D)
            self.Post.setField('nu', as1D(nu), dims=('K'))
            self.Post.setField('B', B, dims=('K', 'D', 'D'))
            self.Post.setField('m', m, dims=('K', 'D'))
            self.Post.setField('kappa', as1D(kappa), dims=('K'))
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
            N = N / K * np.ones(K)

        nu = self.Prior.nu + N
        B = np.zeros((K, D, D))
        for k in xrange(K):
            B[k] = (nu[k] - D - 1) * EstParams.Sigma[k]
        m = EstParams.mu.copy()
        kappa = self.Prior.kappa + N

        self.Post = ParamBag(K=K, D=D)
        self.Post.setField('nu', nu, dims=('K'))
        self.Post.setField('B', B, dims=('K', 'D', 'D'))
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

            This avoids numerical problems due to incremental add/subtract ops
            which can cause computations like
                x = 10.
                x += 1e-15
                x -= 10
                x -= 1e-15
            to be slightly different than zero instead of exactly zero.

            Returns
            -------
            None. SS.N updated in-place.
        '''
        np.maximum(SS.N, 0, out=SS.N)

    def incrementSS(self, SS, k, x):
        SS.x[k] += x
        SS.xxT[k] += np.outer(x, x)

    def decrementSS(self, SS, k, x):
        SS.x[k] -= x
        SS.xxT[k] -= np.outer(x, x)

    def calcSummaryStatsForContigBlock(self, Data, SS=None, a=0, b=0):
        ''' Calculate sufficient stats for a single contiguous block of data
        '''
        if SS is None:
            SS = SuffStatBag(K=1, D=Data.dim)

        SS.setField('N', (b - a) * np.ones(1), dims='K')
        SS.setField(
            'x', np.sum(Data.X[a:b], axis=0)[np.newaxis, :], dims=('K', 'D'))
        SS.setField(
            'xxT', dotATA(Data.X[a:b])[np.newaxis, :, :], dims=('K', 'D', 'D'))
        return SS

    def calcLogSoftEvMatrix_FromEstParams(self, Data, **kwargs):
        ''' Compute log soft evidence matrix for Dataset under EstParams.

        Returns
        ---------
        L : 2D array, N x K
        '''
        K = self.EstParams.K
        L = np.empty((Data.nObs, K))
        for k in xrange(K):
            L[:, k] = - 0.5 * self.D * LOGTWOPI \
                - 0.5 * self._logdetSigma(k)  \
                - 0.5 * self._mahalDist_EstParam(Data.X, k)
        return L

    def _mahalDist_EstParam(self, X, k):
        ''' Calc Mahalanobis distance from comp k to every row of X

        Args
        ---------
        X : 2D array, size N x D
        k : integer ID of comp

        Returns
        ----------
        dist : 1D array, size N
        '''
        Q = np.linalg.solve(self.GetCached('cholSigma', k),
                            (X - self.EstParams.mu[k]).T)
        Q *= Q
        return np.sum(Q, axis=0)

    def _cholSigma(self, k):
        ''' Calculate lower cholesky decomposition of Sigma[k]

        Returns
        --------
        L : 2D array, size D x D, lower triangular
            Sigma = np.dot(L, L.T)
        '''
        return scipy.linalg.cholesky(self.EstParams.Sigma[k], lower=1)

    def _logdetSigma(self, k):
        ''' Calculate log determinant of EstParam.Sigma for comp k

        Returns
        ---------
        logdet : scalar real
        '''
        return 2 * np.sum(np.log(np.diag(self.GetCached('cholSigma', k))))

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
        minCovMat = self.min_covar * np.eye(SS.D)
        covMat = np.tile(minCovMat, (SS.K, 1, 1))
        for k in xrange(SS.K):
            covMat[k] += SS.xxT[k] / SS.N[k] - np.outer(mu[k], mu[k])
        self.EstParams.setField('mu', mu, dims=('K', 'D'))
        self.EstParams.setField('Sigma', covMat, dims=('K', 'D', 'D'))
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
        PB = Prior.B + Prior.kappa * np.outer(Prior.m, Prior.m)

        m = np.empty((SS.K, SS.D))
        B = np.empty((SS.K, SS.D, SS.D))
        for k in xrange(SS.K):
            km_x = Prior.kappa * Prior.m + SS.x[k]
            m[k] = 1.0 / kappa[k] * km_x
            B[k] = PB + SS.xxT[k] - 1.0 / kappa[k] * np.outer(km_x, km_x)

        mu, Sigma = MAPEstParams_inplace(nu, B, m, kappa)
        self.EstParams.setField('mu', mu, dims=('K', 'D'))
        self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))
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

        nu, B, m, kappa = self.calcPostParams(SS)
        self.Post.setField('nu', nu, dims=('K'))
        self.Post.setField('kappa', kappa, dims=('K'))
        self.Post.setField('m', m, dims=('K', 'D'))
        self.Post.setField('B', B, dims=('K', 'D', 'D'))
        self.K = SS.K

    def calcPostParams(self, SS):
        ''' Calc updated params (nu, B, m, kappa) for all comps given suff stats

            These params define the common-form of the exponential family
            Normal-Wishart posterior distribution over mu, diag(Lambda)

            Returns
            --------
            nu : 1D array, size K
            B : 3D array, size K x D x D, each B[k] is symmetric and pos. def.
            m : 2D array, size K x D
            kappa : 1D array, size K
        '''
        Prior = self.Prior
        nu = Prior.nu + SS.N
        kappa = Prior.kappa + SS.N
        m = (Prior.kappa * Prior.m + SS.x) / kappa[:, np.newaxis]
        Bmm = Prior.B + Prior.kappa * np.outer(Prior.m, Prior.m)
        B = SS.xxT + Bmm[np.newaxis, :]
        for k in xrange(B.shape[0]):
            B[k] -= kappa[k] * np.outer(m[k], m[k])
        return nu, B, m, kappa

    def calcPostParamsForComp(self, SS, kA=None, kB=None):
        ''' Calc params (nu, B, m, kappa) for specific comp, given suff stats

            These params define the common-form of the exponential family
            Normal-Wishart posterior distribution over mu[k], diag(Lambda)[k]

            Returns
            --------
            nu : positive scalar
            B : 2D array, size D x D, symmetric and positive definite
            m : 1D array, size D
            kappa : positive scalar
        '''
        if kB is None:
            SN = SS.N[kA]
            Sx = SS.x[kA]
            SxxT = SS.xxT[kA]
        else:
            SN = SS.N[kA] + SS.N[kB]
            Sx = SS.x[kA] + SS.x[kB]
            SxxT = SS.xxT[kA] + SS.xxT[kB]
        Prior = self.Prior
        nu = Prior.nu + SN
        kappa = Prior.kappa + SN
        m = (Prior.kappa * Prior.m + Sx) / kappa
        B = Prior.B + SxxT \
            + Prior.kappa * np.outer(Prior.m, Prior.m) \
            - kappa * np.outer(m, m)
        return nu, B, m, kappa

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
        nu, Bnat, km, kappa = self.calcNaturalPostParams(SS)
        Post = self.Post
        Post.nu[:] = (1 - rho) * Post.nu + rho * nu
        Post.Bnat[:] = (1 - rho) * Post.Bnat + rho * Bnat
        Post.km[:] = (1 - rho) * Post.km + rho * km
        Post.kappa[:] = (1 - rho) * Post.kappa + rho * kappa
        self.convertPostToCommon()

    def calcNaturalPostParams(self, SS):
        ''' Calc  natural posterior parameters given suff stats SS.

        Returns
        --------
        nu : 1D array, size K
        Bnat : 3D array, size K x D x D
        km : 2D array, size K x D
        kappa : 1D array, size K
        '''
        Prior = self.Prior
        nu = Prior.nu + SS.N
        kappa = Prior.kappa + SS.N
        km = Prior.kappa * Prior.m + SS.x
        Bnat = (Prior.B + Prior.kappa * np.outer(Prior.m, Prior.m)) + SS.xxT
        return nu, Bnat, km, kappa

    def convertPostToNatural(self):
        ''' Convert current posterior params from common to natural form
        '''
        Post = self.Post
        assert hasattr(Post, 'nu')
        assert hasattr(Post, 'kappa')
        km = Post.m * Post.kappa[:, np.newaxis]
        Bnat = np.empty((self.K, self.D, self.D))
        for k in xrange(self.K):
            Bnat[k] = Post.B[k] + np.outer(km[k], km[k]) / Post.kappa[k]
        Post.setField('km', km, dims=('K', 'D'))
        Post.setField('Bnat', Bnat, dims=('K', 'D', 'D'))

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

        if hasattr(Post, 'B'):
            B = Post.B  # update in place, no reallocation!
        else:
            B = np.empty((self.K, self.D, self.D))
        for k in xrange(self.K):
            B[k] = Post.Bnat[k] - \
                np.outer(Post.km[k], Post.km[k]) / Post.kappa[k]
        Post.setField('B', B, dims=('K', 'D', 'D'))

    def calcLogSoftEvMatrix_FromPost(self, Data, **kwargs):
        ''' Calculate expected log soft ev matrix under Post.

        Returns
        ------
        L : 2D array, size N x K
        '''
        K = self.Post.K
        L = np.zeros((Data.nObs, K))
        for k in xrange(K):
            L[:, k] = - 0.5 * self.D * LOGTWOPI \
                + 0.5 * self.GetCached('E_logdetL', k)  \
                - 0.5 * self._mahalDist_Post(Data.X, k)
        return L

    def _mahalDist_Post(self, X, k):
        ''' Calc expected mahalonobis distance from comp k to each data atom

            Returns
            --------
            distvec : 1D array, size nObs
                   distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k
        '''
        Q = np.linalg.solve(self.GetCached('cholB', k),
                            (X - self.Post.m[k]).T)
        Q *= Q
        return self.Post.nu[k] * np.sum(Q, axis=0) \
            + self.D / self.Post.kappa[k]

    def calcELBO_Memoized(self, SS, returnVec=0, afterMStep=False, **kwargs):
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
            elbo[k] = c_Diff(Prior.nu,
                             self.GetCached('logdetB'),
                             Prior.m, Prior.kappa,
                             Post.nu[k],
                             self.GetCached('logdetB', k),
                             Post.m[k], Post.kappa[k],
                             )
            if not afterMStep:
                aDiff = SS.N[k] + Prior.nu - Post.nu[k]
                bDiff = SS.xxT[k] + Prior.B \
                                  + Prior.kappa * np.outer(Prior.m, Prior.m) \
                    - Post.B[k] \
                    - Post.kappa[k] * np.outer(Post.m[k], Post.m[k])
                cDiff = SS.x[k] + Prior.kappa * Prior.m \
                    - Post.kappa[k] * Post.m[k]
                dDiff = SS.N[k] + Prior.kappa - Post.kappa[k]
                elbo[k] += 0.5 * aDiff * self.GetCached('E_logdetL', k) \
                    - 0.5 * self._trace__E_L(bDiff, k) \
                    + np.inner(cDiff, self.GetCached('E_Lmu', k)) \
                    - 0.5 * dDiff * self.GetCached('E_muLmu', k)
        if returnVec:
            return elbo - (0.5 * SS.D * LOGTWOPI) * SS.N
        return elbo.sum() - 0.5 * np.sum(SS.N) * SS.D * LOGTWOPI

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
        Post = self.Post
        Prior = self.Prior
        cA = c_Func(Post.nu[kA], Post.B[kA], Post.m[kA], Post.kappa[kA])
        cB = c_Func(Post.nu[kB], Post.B[kB], Post.m[kB], Post.kappa[kB])
        cPrior = c_Func(Prior.nu, Prior.B, Prior.m, Prior.kappa)

        nu, B, m, kappa = self.calcPostParamsForComp(SS, kA, kB)
        cAB = c_Func(nu, B, m, kappa)
        return cA + cB - cPrior - cAB

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calculate change in ELBO for all candidate hard merge pairs

        Returns
        ---------
        Gap : 2D array, size K x K, upper-triangular entries non-zero
              Gap[j,k] : scalar change in ELBO after merge of k into j
        '''
        Post = self.Post
        Prior = self.Prior
        cPrior = c_Func(Prior.nu, Prior.B, Prior.m, Prior.kappa)
        c = np.zeros(SS.K)
        for k in xrange(SS.K):
            c[k] = c_Func(Post.nu[k], Post.B[k], Post.m[k], Post.kappa[k])

        Gap = np.zeros((SS.K, SS.K))
        for j in xrange(SS.K):
            for k in xrange(j + 1, SS.K):
                nu, B, m, kappa = self.calcPostParamsForComp(SS, j, k)
                cjk = c_Func(nu, B, m, kappa)
                Gap[j, k] = c[j] + c[k] - cPrior - cjk
        return Gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
        '''
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
        return Gaps

    def calcHardMergeGap_SpecificPairSS(self, SS1, SS2):
        ''' Calc change in ELBO for merge of two K=1 suff stat bags.

        Returns
        -------
        gap : scalar float
        '''
        assert SS1.K == 1
        assert SS2.K == 1

        Prior = self.Prior
        cPrior = c_Func(Prior.nu, Prior.B, Prior.m, Prior.kappa)

        # Compute cumulants of individual states 1 and 2
        nu1, B1, m1, kappa1 = self.calcPostParamsForComp(SS1, 0)
        nu2, B2, m2, kappa2 = self.calcPostParamsForComp(SS2, 0)
        c1 = c_Func(nu1, B1, m1, kappa1)
        c2 = c_Func(nu2, B2, m2, kappa2)

        # Compute cumulant of merged state 1&2
        SS12 = SS1 + SS2
        nu12, B12, m12, kappa12 = self.calcPostParamsForComp(SS12, 0)
        c12 = c_Func(nu12, B12, m12, kappa12)

        return c1 + c2 - cPrior - c12

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
               logM = log p( data assigned to comp kA )
                      computed up to an additive constant
        '''
        nu, beta, m, kappa = self.calcPostParamsForComp(SS, kA, kB)
        return -1 * c_Func(nu, beta, m, kappa)

    def calcMargLik(self, SS):
        ''' Calc log marginal likelihood across all comps, given suff stats

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
            nu, B, m, kappa = self.calcPostParamsForComp(SS, k)
            logp[k] = c_Diff(Prior.nu, Prior.B, Prior.m, Prior.kappa,
                             nu, B, m, kappa)
        return np.sum(logp) - 0.5 * np.sum(SS.N) * LOGTWOPI

    def calcPredProbVec_Unnorm(self, SS, x):
        ''' Calculate predictive probability that each comp assigns to vector x

            Returns
            --------
            p : 1D array, size K, all entries positive
                p[k] \propto p( x | SS for comp k)
        '''
        return self._calcPredProbVec_Fast(SS, x)

    def _calcPredProbVec_cFunc(self, SS, x):
        nu, B, m, kappa = self.calcPostParams(SS)
        pSS = SS.copy()
        pSS.N += 1
        pSS.x += x[np.newaxis, :]
        pSS.xxT += np.outer(x, x)[np.newaxis, :, :]
        pnu, pB, pm, pkappa = self.calcPostParams(pSS)
        logp = np.zeros(SS.K)
        for k in xrange(SS.K):
            logp[k] = c_Diff(nu[k], B[k], m[k], kappa[k],
                             pnu[k], pB[k], pm[k], pkappa[k])
        return np.exp(logp - np.max(logp))

    def _calcPredProbVec_Fast(self, SS, x):
        nu, B, m, kappa = self.calcPostParams(SS)
        kB = B
        kB *= ((kappa + 1) / kappa)[:, np.newaxis, np.newaxis]
        logp = np.zeros(SS.K)
        p = logp  # Rename so its not confusing what we're returning
        for k in xrange(SS.K):
            cholKB = scipy.linalg.cholesky(kB[k], lower=1)
            logdetKB = 2 * np.sum(np.log(np.diag(cholKB)))
            mVec = np.linalg.solve(cholKB, x - m[k])
            mDist_k = np.inner(mVec, mVec)
            logp[k] = -0.5 * logdetKB - 0.5 * \
                (nu[k] + 1) * np.log(1.0 + mDist_k)
        logp += gammaln(0.5 * (nu + 1)) - gammaln(0.5 * (nu + 1 - self.D))
        logp -= np.max(logp)
        np.exp(logp, out=p)
        return p

    def _Verify_calcPredProbVec(self, SS, x):
        ''' Verify that the predictive prob vector is correct,
              by comparing very different implementations
        '''
        pA = self._calcPredProbVec_Fast(SS, x)
        pC = self._calcPredProbVec_cFunc(SS, x)
        pA /= np.sum(pA)
        pC /= np.sum(pC)
        assert np.allclose(pA, pC)

    def _E_CovMat(self, k=None):
        if k is None:
            B = self.Prior.B
            nu = self.Prior.nu
        else:
            B = self.Post.B[k]
            nu = self.Post.nu[k]
        return B / (nu - self.D - 1)

    def _cholB(self, k=None):
        if k == 'all':
            retArr = np.zeros((self.K, self.D, self.D))
            for kk in xrange(self.K):
                retArr[kk] = self.GetCached('cholB', kk)
            return retArr
        elif k is None:
            B = self.Prior.B
        else:
            # k is one of [0, 1, ... K-1]
            B = self.Post.B[k]
        return scipy.linalg.cholesky(B, lower=True)

    def _logdetB(self, k=None):
        cholB = self.GetCached('cholB', k)
        return 2 * np.sum(np.log(np.diag(cholB)))

    def _E_logdetL(self, k=None):
        dvec = np.arange(1, self.D + 1, dtype=np.float)
        if k == 'all':
            dvec = dvec[:, np.newaxis]
            retVec = self.D * LOGTWO * np.ones(self.K)
            for kk in xrange(self.K):
                retVec[kk] -= self.GetCached('logdetB', kk)
            nuT = self.Post.nu[np.newaxis, :]
            retVec += np.sum(digamma(0.5 * (nuT + 1 - dvec)), axis=0)
            return retVec
        elif k is None:
            nu = self.Prior.nu
        else:
            nu = self.Post.nu[k]
        return self.D * LOGTWO \
            - self.GetCached('logdetB', k) \
            + np.sum(digamma(0.5 * (nu + 1 - dvec)))

    def _trace__E_L(self, Smat, k=None):
        if k is None:
            nu = self.Prior.nu
            B = self.Prior.B
        else:
            nu = self.Post.nu[k]
            B = self.Post.B[k]
        return nu * np.trace(np.linalg.solve(B, Smat))

    def _E_Lmu(self, k=None):
        if k is None:
            nu = self.Prior.nu
            B = self.Prior.B
            m = self.Prior.m
        else:
            nu = self.Post.nu[k]
            B = self.Post.B[k]
            m = self.Post.m[k]
        return nu * np.linalg.solve(B, m)

    def _E_muLmu(self, k=None):
        if k is None:
            nu = self.Prior.nu
            kappa = self.Prior.kappa
            m = self.Prior.m
            B = self.Prior.B
        else:
            nu = self.Post.nu[k]
            kappa = self.Post.kappa[k]
            m = self.Post.m[k]
            B = self.Post.B[k]
        Q = np.linalg.solve(self.GetCached('cholB', k), m.T)
        return self.D / kappa + nu * np.inner(Q, Q)

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
            fillSharedMemArray(ShMem['cholB'], self._cholB('all'))
            fillSharedMemArray(ShMem['E_logdetL'], self._E_logdetL('all'))

        else:
            ShMem['nu'] = numpyToSharedMemArray(self.Post.nu)
            ShMem['kappa'] = numpyToSharedMemArray(self.Post.kappa)
            ShMem['m'] = numpyToSharedMemArray(self.Post.m.copy())
            ShMem['cholB'] = numpyToSharedMemArray(self._cholB('all'))
            ShMem['E_logdetL'] = numpyToSharedMemArray(self._E_logdetL('all'))

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
        Mu_1 : 2D array, size D x D
            Expected value of Cov[ X[n] ]
        Mu_2 : 1D array, size D
            Expected value of Mean[ X[n] ]
        '''
        if X is None:
            Mu1 = self.Prior.B / self.Prior.nu
            Mu2 = self.Prior.m
            return Mu1, Mu2

        if X.ndim == 1:
            X = X[np.newaxis,:]
        N, D = X.shape
        # Compute suff stats
        if W is None:
            sum_wxxT = np.dot(X.T, X)
            sum_wx = np.sum(X, axis=0)
            sum_w = X.shape[0]
        else:
            W = as1D(W)
            sqrtWX = np.sqrt(W)[:,np.newaxis] * X
            sum_wxxT = np.dot(sqrtWX.T, sqrtWX)
            sum_wx = np.dot(W, X)
            sum_w = np.sum(W)

        kappa = self.Prior.kappa + sum_w
        m = (self.Prior.m * self.Prior.kappa + sum_wx) / kappa
        Mu_2 = m

        prior_kmmT = self.Prior.kappa * np.outer(self.Prior.m, self.Prior.m)
        post_kmmT = kappa * np.outer(m,m)
        B = sum_wxxT + self.Prior.B + prior_kmmT - post_kmmT
        Mu_1 = B / (self.Prior.nu + sum_w)

        assert Mu_1.ndim == 2
        assert Mu_1.shape == (D, D,)
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
        assert Mu[0][0].shape[0] == D
        assert Mu[0][0].shape[1] == D
        assert Mu[0][1].size == D

        prior_x = self.Prior.m
        prior_covx = self.Prior.B / (self.Prior.nu)
        CovX = eps * prior_covx

        Div = np.zeros((N, K))
        for k in xrange(K):
            chol_CovMu_k = np.linalg.cholesky(Mu[k][0])
            logdet_CovMu_k = 2.0 * np.sum(np.log(np.diag(chol_CovMu_k)))
            tr_InvMu_CovX_k = np.trace(np.linalg.solve(
                Mu[k][0], CovX))
            XdiffMu_k = X - Mu[k][1]
            tr_InvMu_XdXdT_k = np.linalg.solve(chol_CovMu_k, XdiffMu_k.T)
            tr_InvMu_XdXdT_k *= tr_InvMu_XdXdT_k
            tr_InvMu_XdXdT_k = tr_InvMu_XdXdT_k.sum(axis=0)
            Div[:,k] = \
                + 0.5 * logdet_CovMu_k \
                + 0.5 * (tr_InvMu_CovX_k + tr_InvMu_XdXdT_k)

        if not includeOnlyFastTerms:
            if DivDataVec is None:
                # Compute DivDataVec : 1D array of size N
                # This is the per-row additive constant indep. of k. 
                DivDataVec = -0.5 * D * np.ones(N)
                s, logdet = np.linalg.slogdet(CovX)
                logdet_CovX = s * logdet
                DivDataVec -= 0.5 * logdet_CovX
        
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
        assert Mu[0][0].ndim == 2
        assert Mu[0][1].ndim == 1
        D = Mu[0][0].shape[0]
        assert D == Mu[0][0].shape[1]
        assert D == Mu[0][1].size

        priorCov = self.Prior.B / self.Prior.nu
        priorMu_2 = self.Prior.m

        priorN_ZMG = (1-smoothFrac) * self.Prior.nu
        priorN_FVG = (1-smoothFrac) * self.Prior.kappa

        Div_ZMG = np.zeros(K) # zero-mean gaussian
        Div_FVG = np.zeros(K) # fixed variance gaussian
 
        s, logdet = np.linalg.slogdet(priorCov)
        logdet_priorCov = s * logdet
        for k in xrange(K):
            Cov_k = Mu[k][0]
            s, logdet = np.linalg.slogdet(Cov_k)
            logdet_Cov_k = s * logdet
            Div_ZMG[k] = 0.5 * logdet_Cov_k + \
                - 0.5 * logdet_priorCov \
                + 0.5 * np.trace(np.linalg.solve(Cov_k, priorCov)) \
                - 0.5
            pmT = np.outer(priorMu_2 - Mu[k][1], priorMu_2 - Mu[k][1])
            Div_FVG[k] = 0.5 * np.trace(np.linalg.solve(Cov_k, pmT))
            
        return priorN_ZMG * Div_ZMG + priorN_FVG * Div_FVG
    # .... end class


def MAPEstParams_inplace(nu, B, m, kappa=0):
    ''' MAP estimate parameters mu, Sigma given Normal-Wishart hyperparameters
    '''
    D = m.size
    mu = m
    Sigma = B
    for k in xrange(B.shape[0]):
        Sigma[k] /= (nu[k] + D + 1)
    return mu, Sigma


def c_Func(nu, logdetB, m, kappa):
    ''' Evaluate cumulant function at given params.

    Returns
    --------
    c : scalar real value of cumulant function at provided args
    '''
    if logdetB.ndim >= 2:
        logdetB = np.log(np.linalg.det(logdetB))
    D = m.size
    dvec = np.arange(1, D + 1, dtype=np.float)
    return - 0.5 * D * LOGTWOPI \
           - 0.25 * D * (D - 1) * LOGPI \
           - 0.5 * D * LOGTWO * nu \
           - np.sum(gammaln(0.5 * (nu + 1 - dvec))) \
        + 0.5 * D * np.log(kappa) \
        + 0.5 * nu * logdetB


def c_Diff(nu1, logdetB1, m1, kappa1,
           nu2, logdetB2, m2, kappa2):
    ''' Evaluate difference of cumulant functions c(params1) - c(params2)

    May be more numerically stable than directly using c_Func
    to find the difference.

    Returns
    -------
    diff : scalar real value of the difference in cumulant functions
    '''
    if logdetB1.ndim >= 2:
        logdetB1 = np.log(np.linalg.det(logdetB1))
    if logdetB2.ndim >= 2:
        logdetB2 = np.log(np.linalg.det(logdetB2))
    D = m1.size
    dvec = np.arange(1, D + 1, dtype=np.float)
    return - 0.5 * D * LOGTWO * (nu1 - nu2) \
           - np.sum(gammaln(0.5 * (nu1 + 1 - dvec))) \
        + np.sum(gammaln(0.5 * (nu2 + 1 - dvec))) \
        + 0.5 * D * (np.log(kappa1) - np.log(kappa2)) \
        + 0.5 * (nu1 * logdetB1 - nu2 * logdetB2)


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
        S_xxT = np.zeros((K, Data.dim, Data.dim))
        sqrtResp_k = np.sqrt(resp[:, 0])
        sqrtRX_k = sqrtResp_k[:, np.newaxis] * Data.X
        S_xxT[0] = dotATA(sqrtRX_k)
        for k in xrange(1, K):
            np.sqrt(resp[:, k], out=sqrtResp_k)
            np.multiply(sqrtResp_k[:, np.newaxis], Data.X, out=sqrtRX_k)
            S_xxT[k] = dotATA(sqrtRX_k)
    else:
        spR = LP['spR']
        K = spR.shape[1]
        # 1/2: Compute mean statistic
        S_x = spR.T * X
        # 2/2: Compute expected outer-product statistic
        S_xxT = calcSpRXXT(X=X, spR_csr=spR)

    if SS is None:
        SS = SuffStatBag(K=K, D=Data.dim)
    # Expected mean for each state k
    SS.setField('x', S_x, dims=('K', 'D'))
    # Expected outer-product for each state k
    SS.setField('xxT', S_xxT, dims=('K', 'D', 'D'))
    # Expected count for each k
    #  Usually computed by allocmodel. But just in case...
    if not hasattr(SS, 'N'):
        if 'resp' in LP:
            SS.setField('N', LP['resp'].sum(axis=0), dims='K')
        else:
            SS.setField('N', as1D(toCArray(LP['spR'].sum(axis=0))), dims='K')
    return SS


def calcLocalParams(Dslice, **kwargs):
    L = calcLogSoftEvMatrix_FromPost(Dslice, **kwargs)
    LP = dict(E_log_soft_ev=L)
    return LP


def calcLogSoftEvMatrix_FromPost(Dslice, **kwargs):
    ''' Calculate expected log soft ev matrix for variational.

    Returns
    ------
    L : 2D array, size N x K
    '''
    K = kwargs['K']
    L = np.zeros((Dslice.nObs, K))
    for k in xrange(K):
        L[:, k] = - 0.5 * Dslice.dim * LOGTWOPI \
            + 0.5 * kwargs['E_logdetL'][k]  \
            - 0.5 * _mahalDist_Post(Dslice.X, k, **kwargs)
    return L


def _mahalDist_Post(X, k, D=None,
                    cholB=None,
                    m=None, nu=None, kappa=None, **kwargs):
    ''' Calc expected mahalonobis distance from comp k to each data atom

    Returns
    --------
    distvec : 1D array, size N
           distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k
    '''
    Q = np.linalg.solve(cholB[k],
                        (X - m[k]).T)
    Q *= Q
    return nu[k] * np.sum(Q, axis=0) + D / kappa[k]


def createECovMatFromUserInput(D=0, Data=None, ECovMat='eye', sF=1.0):
    ''' Create expected covariance matrix defining Wishart prior.

        The creation process follows user-specified criteria.

    Args
    --------
    D : positive integer, size of each observation
    Data : [optional] dataset to use to make Sigma in data-driven way
    ECovMat : string name of the procedure to use to create Sigma
        'eye' : make Sigma a multiple of the identity matrix
        'covdata' : set Sigma to a multiple of the data covariance matrix
        'fromtruelabels' : set Sigma to the empirical mean of the
            covariances for each true cluster in the dataset

    Returns
    --------
    Sigma : 2D array, size D x D
        Symmetric and positive definite.
    '''
    if Data is not None:
        assert D == Data.dim
    if ECovMat == 'eye':
        Sigma = sF * np.eye(D)
    elif ECovMat == 'covdata':
        Sigma = sF * np.cov(Data.X.T, bias=1)
    elif ECovMat == 'diagcovdata':
        CovMat = as2D(np.cov(Data.X.T, bias=1)) # as2D deals with case of D=1
        Sigma = sF * np.diag(np.diag(CovMat))
    elif ECovMat == 'covfirstdiff':
        if not hasattr(Data, 'Xprev'):
            raise ValueError(
                'covfirstdiff only applies to auto-regressive datasets')
        Xdiff = Data.X - Data.Xprev
        Sigma = sF * np.cov(Xdiff.T, bias=1)
    elif ECovMat == 'diagcovfirstdiff':
        if not hasattr(Data, 'Xprev'):
            raise ValueError(
                'covfirstdiff only applies to auto-regressive datasets')
        Xdiff = Data.X - Data.Xprev
        Sigma = sF * np.diag(np.diag(np.cov(Xdiff.T, bias=1)))

    elif ECovMat == 'fromtruelabels':
        ''' Set Cov Matrix Sigma using the true labels in empirical Bayes style
            Sigma = \sum_{c : class labels} w_c * SampleCov[ data from class c]
        '''
        if hasattr(Data, 'TrueLabels'):
            Z = Data.TrueLabels
        else:
            Z = Data.TrueParams['Z']
        Zvals = np.unique(Z)
        Kmax = len(Zvals)
        wHat = np.zeros(Kmax)
        SampleCov = np.zeros((Kmax, D, D))
        for kLoc, kVal in enumerate(Zvals):
            mask = Z == kVal
            wHat[kLoc] = np.sum(mask)
            SampleCov[kLoc] = np.cov(Data.X[mask].T, bias=1)
        wHat = wHat / np.sum(wHat)
        Sigma = 1e-8 * np.eye(D)
        for k in range(Kmax):
            Sigma += wHat[k] * SampleCov[k]
    else:
        raise ValueError('Unrecognized ECovMat procedure %s' % (ECovMat))
    return Sigma
