from builtins import *
import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D, toCArray
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray
from bnpy.util.SparseRespStatsUtil import calcSpRXXT
from .AbstractObsModel import AbstractObsModel
from .GaussObsModel import createECovMatFromUserInput


class ZeroMeanGaussObsModel(AbstractObsModel):

    ''' Zero-mean, full-covariance gaussian model for real vectors.

    Attributes for Prior (Normal-Wishart)
    --------
    nu : float
        degrees of freedom
    B : 2D array, size D x D
        scale parameters that set mean of parameter Sigma

    Attributes for k-th component of EstParams (EM point estimates)
    ---------
    Sigma[k] : 2D array, size DxD

    Attributes for k-th component of Post (VB parameter)
    ---------
    nu[k] : float
    B[k] : 1D array, size D

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
        else:
            B = as2D(B)
        self.Prior = ParamBag(K=0, D=D)
        self.Prior.setField('nu', nu, dims=None)
        self.Prior.setField('B', B, dims=('D', 'D'))

    def get_mean_for_comp(self, k):
        return np.zeros(self.D)

    def get_covar_mat_for_comp(self, k=None):
        if hasattr(self, 'EstParams'):
            return self.EstParams.Sigma[k]
        elif k is None or k == 'prior':
            return self._E_CovMat()
        else:
            return self._E_CovMat(k)

    def get_name(self):
        return 'ZeroMeanGauss'

    def get_info_string(self):
        return 'Gaussian with fixed zero means, full covariance.'

    def get_info_string_prior(self):
        msg = 'Wishart on prec matrix Lam\n'
        if self.D > 2:
            sfx = ' ...'
        else:
            sfx = ''
        S = self._E_CovMat()[:2, :2]
        msg += 'E[ CovMat[k] ] = \n'
        msg += str(S) + sfx
        msg = msg.replace('\n', '\n  ')
        return msg

    def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                     Sigma=None,
                     **kwargs):
        ''' Create EstParams ParamBag with fields Sigma
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
            K = Sigma.shape[0]
            self.EstParams = ParamBag(K=K, D=self.D)
            self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))
        self.K = self.EstParams.K

    def setEstParamsFromPost(self, Post):
        ''' Convert from Post (nu, B) to EstParams (Sigma),
             each EstParam is set to its posterior mean.
        '''
        D = Post.D
        self.EstParams = ParamBag(K=Post.K, D=D)
        Sigma = Post.B / (Post.nu - D - 1)[:, np.newaxis, np.newaxis]
        self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))
        self.K = self.EstParams.K

    def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                       nu=0, B=0,
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
            K = B.shape[0]
            self.Post = ParamBag(K=K, D=self.D)
            self.Post.setField('nu', as1D(nu), dims=('K'))
            self.Post.setField('B', B, dims=('K', 'D', 'D'))
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
        B = np.zeros((K, D, D))
        for k in range(K):
            B[k] = (nu[k] - D - 1) * EstParams.Sigma[k]
        self.Post = ParamBag(K=K, D=D)
        self.Post.setField('nu', nu, dims=('K'))
        self.Post.setField('B', B, dims=('K', 'D', 'D'))
        self.K = K

    def calcSummaryStats(self, Data, SS, LP, **kwargs):
        ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        return calcSummaryStats(Data, SS, LP, **kwargs)

    def calcSummaryStatsForContigBlock(self, Data, SS=None,
                                       a=None, b=None, **kwargs):
        ''' Calculate summary statistics for specific block of dataset

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        SS = SuffStatBag(K=1, D=Data.dim)

        # Expected count
        SS.setField('N', (b - a) * np.ones(1, dtype=np.float64), dims='K')

        # Expected outer-product
        xxT = dotATA(Data.X[a:b])[np.newaxis, :, :]
        SS.setField('xxT', xxT, dims=('K', 'D', 'D'))
        return SS

    def forceSSInBounds(self, SS):
        ''' Force count vector N to remain positive
        '''
        np.maximum(SS.N, 0, out=SS.N)

    def incrementSS(self, SS, k, x):
        SS.xxT[k] += np.outer(x, x)

    def decrementSS(self, SS, k, x):
        SS.xxT[k] -= np.outer(x, x)

    def calcLogSoftEvMatrix_FromEstParams(self, Data, **kwargs):
        ''' Compute log soft evidence matrix for Dataset under EstParams.

        Returns
        ---------
        L : 2D array, N x K
        '''
        K = self.EstParams.K
        L = np.empty((Data.nObs, K))
        for k in range(K):
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
        cholSigma_k = self.GetCached('cholSigma', k)
        Q = scipy.linalg.solve_triangular(
            cholSigma_k, X.T, lower=True, check_finite=False)
        Q *= Q
        return np.sum(Q, axis=0)

    def _cholSigma(self, k):
        ''' Calculate lower cholesky decomposition of Sigma for comp k

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

        minCovMat = self.min_covar * np.eye(SS.D)
        covMat = np.tile(minCovMat, (SS.K, 1, 1))
        for k in range(SS.K):
            covMat[k] += SS.xxT[k] / SS.N[k]
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
        B = np.empty((SS.K, SS.D, SS.D))
        for k in range(SS.K):
            B[k] = Prior.B + SS.xxT[k]

        Sigma = MAPEstParams_inplace(nu, B)
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

        # use 'Prior' not 'self.Prior', improves readability
        Prior = self.Prior
        Post = self.Post

        Post.setField('nu', Prior.nu + SS.N, dims=('K'))
        B = np.empty((SS.K, SS.D, SS.D))
        for k in range(SS.K):
            B[k] = Prior.B + SS.xxT[k]
        Post.setField('B', B, dims=('K', 'D', 'D'))
        self.K = SS.K

    def calcPostParams(self, SS):
        ''' Calc posterior parameters for all comps given suff stats

        Returns
        --------
        nu : 1D array, size K
        B : 3D array, size K x D x D, each B[k] is symmetric and pos. def.
        '''
        Prior = self.Prior
        nu = Prior.nu + SS.N
        B = Prior.B + SS.xxT
        return nu, B

    def calcPostParamsForComp(self, SS, kA=None, kB=None):
        ''' Calc params (nu, B, m, kappa) for specific comp, given suff stats

        Returns
        --------
        nu : positive scalar
        B : 2D array, size D x D, symmetric and positive definite
        '''
        if kB is None:
            SN = SS.N[kA]
            SxxT = SS.xxT[kA]
        else:
            SN = SS.N[kA] + SS.N[kB]
            SxxT = SS.xxT[kA] + SS.xxT[kB]
        Prior = self.Prior
        nu = Prior.nu + SN
        B = Prior.B + SxxT
        return nu, B

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

        nu, B = self.calcPostParams(SS)
        Post = self.Post
        Post.nu[:] = (1 - rho) * Post.nu + rho * nu
        Post.B[:] = (1 - rho) * Post.B + rho * B

    def convertPostToNatural(self):
        ''' Convert current posterior params from common to natural form

        Here, the Wishart common form is already equivalent to the natural form
        '''
        pass

    def convertPostToCommon(self):
        ''' Convert (current posterior params from natural to common form

        Here, the Wishart common form is already equivalent to the natural form
        '''
        pass

    def calcLogSoftEvMatrix_FromPost(self, Data, **kwargs):
        ''' Calculate expected log soft ev matrix under Post.

        Returns
        ------
        L : 2D array, size N x K
        '''
        K = self.Post.K
        L = np.zeros((Data.nObs, K))
        for k in range(K):
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
        cholB_k = self.GetCached('cholB', k)
        Q = scipy.linalg.solve_triangular(
            cholB_k, X.T, lower=True, check_finite=False)
        Q *= Q
        return self.Post.nu[k] * np.sum(Q, axis=0)

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
        for k in range(SS.K):
            elbo[k] = c_Diff(Prior.nu,
                             self.GetCached('logdetB'), self.D,
                             Post.nu[k],
                             self.GetCached('logdetB', k),
                             )
            if not afterMStep:
                aDiff = SS.N[k] + Prior.nu - Post.nu[k]
                bDiff = SS.xxT[k] + Prior.B - Post.B[k]
                elbo[k] += 0.5 * aDiff * self.GetCached('E_logdetL', k) \
                    - 0.5 * self._trace__E_L(bDiff, k)
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
        cPrior = c_Func(Prior.nu, self.GetCached('logdetB'), self.D)

        cA = c_Func(Post.nu[kA], self.GetCached('logdetB', kA), self.D)
        cB = c_Func(Post.nu[kB], self.GetCached('logdetB', kB), self.D)

        nu, B = self.calcPostParamsForComp(SS, kA, kB)
        cAB = c_Func(nu, B)
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
        cPrior = c_Func(Prior.nu, self.GetCached('logdetB'), self.D)

        c = np.zeros(SS.K)
        for k in range(SS.K):
            c[k] = c_Func(Post.nu[k], self.GetCached('logdetB', k), self.D)

        Gap = np.zeros((SS.K, SS.K))
        for j in range(SS.K):
            for k in range(j + 1, SS.K):
                nu, B = self.calcPostParamsForComp(SS, j, k)
                cjk = c_Func(nu, B)
                Gap[j, k] = c[j] + c[k] - cPrior - cjk
        return Gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
        '''
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
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
               logM = log p( data assigned to comp kA )
                      computed up to an additive constant
        '''
        nu, B = self.calcPostParamsForComp(SS, kA, kB)
        return -1 * c_Func(nu, B)

    def calcMargLik(self, SS):
        ''' Calc log marginal likelihood given suff stats

        Returns
        --------
        logM : scalar real
               logM = \sum_{k=1}^K log p( data assigned to comp k | Prior)
        '''
        return self.calcMargLik_CFuncForLoop(SS)

    def calcMargLik_CFuncForLoop(self, SS):
        Prior = self.Prior
        logp = np.zeros(SS.K)
        for k in range(SS.K):
            nu, B = self.calcPostParamsForComp(SS, k)
            logp[k] = c_Diff(Prior.nu, Prior.B, self.D,
                             nu, B)
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
        pSS.xxT += np.outer(x, x)[np.newaxis, :, :]
        pnu, pB, pm, pkappa = self.calcPostParams(pSS)
        logp = np.zeros(SS.K)
        for k in range(SS.K):
            logp[k] = c_Diff(nu[k], B[k], self.D,
                             pnu[k], pB[k])
        return np.exp(logp - np.max(logp))

    def _calcPredProbVec_Fast(self, SS, x):
        nu, B = self.calcPostParams(SS)
        logp = np.zeros(SS.K)
        p = logp  # Rename so its not confusing what we're returning
        for k in range(SS.K):
            cholB_k = scipy.linalg.cholesky(B[k], lower=1)
            logdetB_k = 2 * np.sum(np.log(np.diag(cholB_k)))
            mVec = np.linalg.solve(cholB_k, x)
            mDist_k = np.inner(mVec, mVec)
            logp[k] = -0.5 * logdetB_k - 0.5 * \
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
            for kk in range(self.K):
                retArr[kk] = self.GetCached('cholB', kk)
            return retArr
        elif k is None:
            B = self.Prior.B
        else:
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
            for kk in range(self.K):
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

    def getSmoothedMuForComp(self, k):
        ''' Compute smoothed mean vector for cluster k

        Returns
        -------
        Mu_k : 2D array, size D x D
        '''
        #return self.Post.B[k] / self.Post.nu[k]
        return self.get_covar_mat_for_comp(k)

    def calcSmoothedMu(self, X, W=None):
        ''' Compute smoothed estimate of mean of statistic xxT.

        Args
        ----
        X : 2D array, size N x D

        Returns
        -------
        Mu : 2D array, size D x D
        '''
        Prior_nu = self.Prior.nu - self.D - 1
        # Prior_nu = self.Prior.nu

        if X is None:
            Mu = self.Prior.B / (Prior_nu)
            return Mu
        if X.ndim == 1:
            X = X[np.newaxis,:]
        N, D = X.shape
        # Compute suff stats
        if W is None:
            sum_wxxT = np.dot(X.T, X)
            sum_w = X.shape[0]
        else:
            W = as1D(W)
            wX = np.sqrt(W)[:,np.newaxis] * X
            sum_wxxT = np.dot(wX.T, wX)
            sum_w = np.sum(W)
        Mu = (self.Prior.B + sum_wxxT) / (Prior_nu + sum_w)
        assert Mu.ndim == 2
        assert Mu.shape == (D, D,)
        return Mu

    def calcSmoothedBregDiv(
            self, X, Mu,
            W=None,
            smoothFrac=0.0,
            eps=1e-10,
            includeOnlyFastTerms=False,
            DivDataVec=None,
            returnDivDataVec=False,
            return1D=False,
            **kwargs):
        ''' Compute Bregman divergence between data X and clusters Mu.

        Smooth the data via update with prior parameters.

        Returns
        -------
        Div : 2D array, N x K
            Div[n,k] = smoothed distance between X[n] and Mu[k]
        '''
        if X.ndim < 2:
            X = X[np.newaxis,:]
        assert X.ndim == 2
        N = X.shape[0]
        D = X.shape[1]
        if not isinstance(Mu, list):
            Mu = [Mu]
        K = len(Mu)
        assert Mu[0].ndim == 2
        assert Mu[0].shape[0] == D
        assert Mu[0].shape[1] == D

        if smoothFrac == 0:
            smoothMu = eps * self.Prior.B / (self.Prior.nu - self.D - 1)
            smoothNu = 1.0 # + eps ??
        else:
            smoothMu = self.Prior.B
            smoothNu = 1 + self.Prior.nu - self.D - 1
        Div = np.zeros((N, K))
        for k in range(K):
            chol_Mu_k = np.linalg.cholesky(Mu[k])
            logdet_Mu_k = 2.0 * np.sum(np.log(np.diag(chol_Mu_k)))
            xxTInvMu_k = scipy.linalg.solve_triangular(
                chol_Mu_k, X.T, lower=True, check_finite=False)
            xxTInvMu_k *= xxTInvMu_k
            tr_xxTInvMu_k = np.sum(xxTInvMu_k, axis=0) / smoothNu
            Div[:,k] = 0.5 * logdet_Mu_k + \
                0.5 * tr_xxTInvMu_k
            if smoothFrac > 0:
                Div[:, k] += 0.5 * np.trace(
                    np.linalg.solve(Mu[k], smoothMu))

        if not includeOnlyFastTerms:
            if DivDataVec is None:
                # Compute DivDataVec : 1D array of size N
                # This is the per-row additive constant indep. of k.
                # We do lots of steps in-place, to save memory.

                # FAST VERSION: Use the matrix determinant lemma
                chol_SM = np.linalg.cholesky(smoothMu/smoothNu)
                logdet_SM = 2.0 * np.sum(np.log(np.diag(chol_SM)))
                xxTInvSM = scipy.linalg.solve_triangular(
                    chol_SM, X.T, lower=True)
                xxTInvSM *= xxTInvSM
                tr_xxTSM = np.sum(xxTInvSM, axis=0) / smoothNu
                assert tr_xxTSM.size == N
                DivDataVec = np.log(1.0 + tr_xxTSM)
                DivDataVec *= -0.5
                DivDataVec += -0.5 * D - 0.5 * logdet_SM
                # SLOW VERSION: use a naive for loop
                # DivDataVecS = -0.5 * D * np.ones(N)
                # for n in xrange(N):
                #    s, logdet_xxT_n = np.linalg.slogdet(
                #        (np.outer(X[n], X[n]) + smoothMu) / smoothNu)
                #    DivDataVecS[n] -= 0.5 * s * logdet_xxT_n

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
        '''
        logdet_xxT = np.zeros(N)
        tr_xxTInvMu = np.zeros((N, K))
        for n in xrange(N):
            if smoothFrac == 0:
                smooth_xxT = np.outer(X[n], X[n]) + eps * priorMu
            else:
                smooth_xxT = np.outer(X[n], X[n]) + self.Prior.B
                smooth_xxT /= (1.0 + self.Prior.nu)
            s, logdet = np.linalg.slogdet(smooth_xxT)
            logdet_xxT[n] = s * logdet

            for k in xrange(K):
                tr_xxTInvMu[n, k] = np.trace(
                    np.linalg.solve(Mu[k], smooth_xxT))

        Div = np.zeros((N, K))
        for k in xrange(K):
            chol_Mu_k = np.linalg.cholesky(Mu[k])
            logdet_Mu_k = 2.0 * np.sum(np.log(np.diag(chol_Mu_k)))

            Div[:,k] = -0.5 * D - 0.5 * logdet_xxT + \
                0.5 * logdet_Mu_k + \
                0.5 * tr_xxTInvMu[:, k]
        '''

    def calcBregDivFromPrior(self, Mu, smoothFrac=0.0):
        ''' Compute Bregman divergence between Mu and prior mean.

        Returns
        -------
        Div : 1D array, size K
            Div[k] = distance between Mu[k] and priorMu
        '''
        if not isinstance(Mu, list):
            Mu = [Mu]
        K = len(Mu)
        D = Mu[0].shape[0]
        assert D == Mu[0].shape[1]

        priorMu = self.Prior.B / self.Prior.nu
        priorN = (1-smoothFrac) * (self.Prior.nu)

        Div = np.zeros(K)
        s, logdet = np.linalg.slogdet(priorMu)
        logdet_prior = s * logdet
        for k in range(K):
            chol_Mu_k = np.linalg.cholesky(Mu[k])
            logdet_Mu_k = 2.0 * np.sum(np.log(np.diag(chol_Mu_k)))
            tr_PriorInvMu_k = np.trace(np.linalg.solve(Mu[k], priorMu))
            Div[k] = -0.5 * logdet_prior + 0.5 * logdet_Mu_k + \
                0.5 * tr_PriorInvMu_k - 0.5 * D
        return priorN * Div

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
            fillSharedMemArray(ShMem['cholB'], self._cholB('all'))
            fillSharedMemArray(ShMem['E_logdetL'], self._E_logdetL('all'))

        else:
            ShMem['nu'] = numpyToSharedMemArray(self.Post.nu)
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
    # .... end class


def MAPEstParams_inplace(nu, B):
    ''' MAP estimate parameters mu, Sigma given Wishart hyperparameters
    '''
    D = B.shape[-1]
    Sigma = B
    for k in range(B.shape[0]):
        Sigma[k] /= (nu[k] + D + 1)
    return Sigma


def c_Func(nu, logdetB, D=None):
    ''' Evaluate cumulant function at given params.

    Returns
    --------
    c : scalar real value of cumulant function at provided args
    '''
    if logdetB.ndim >= 2:
        D = logdetB.shape[-1]
        logdetB = np.log(np.linalg.det(logdetB))
    dvec = np.arange(1, D + 1, dtype=np.float)
    return - 0.5 * D * LOGTWO * nu \
        - np.sum(gammaln(0.5 * (nu + 1 - dvec))) \
        + 0.5 * nu * logdetB


def c_Diff(nu1, logdetB1, D, nu2, logdetB2):
    ''' Evaluate difference of cumulant functions c(params1) - c(params2)

    May be more numerically stable than directly using c_Func
    to find the difference.

    Returns
    -------
    diff : scalar real value of the difference in cumulant functions
    '''
    if logdetB1.ndim >= 2:
        assert D == logdetB1.shape[-1]
        logdetB1 = np.log(np.linalg.det(logdetB1))
        logdetB2 = np.log(np.linalg.det(logdetB2))
    dvec = np.arange(1, D + 1, dtype=np.float)
    return - 0.5 * D * LOGTWO * (nu1 - nu2) \
        - np.sum(gammaln(0.5 * (nu1 + 1 - dvec))) \
        + np.sum(gammaln(0.5 * (nu2 + 1 - dvec))) \
        + 0.5 * (nu1 * logdetB1 - nu2 * logdetB2)


def calcSummaryStats(Data, SS, LP, **kwargs):
    ''' Calculate summary statistics for given dataset and local parameters

    Returns
    --------
    SS : SuffStatBag object, with K components.
    '''
    X = Data.X
    D = Data.dim
    if 'resp' in LP:
        resp = LP['resp']
        K = resp.shape[1]
        # Compute expected outer-product statistic
        S_xxT = np.zeros((K, Data.dim, Data.dim))
        sqrtResp_k = np.sqrt(resp[:, 0])
        sqrtRX_k = sqrtResp_k[:, np.newaxis] * Data.X
        S_xxT[0] = dotATA(sqrtRX_k)
        for k in range(1, K):
            np.sqrt(resp[:, k], out=sqrtResp_k)
            np.multiply(sqrtResp_k[:, np.newaxis], Data.X, out=sqrtRX_k)
            S_xxT[k] = dotATA(sqrtRX_k)

        sqrtResp = np.sqrt(resp)
        xxT = np.zeros((K, D, D))
        for k in range(K):
            xxT[k] = dotATA(sqrtResp[:, k][:, np.newaxis] * Data.X)
        assert np.allclose(xxT, S_xxT)
    else:
        spR = LP['spR']
        K = spR.shape[1]
        # Compute expected outer-product statistic
        S_xxT = calcSpRXXT(X=X, spR_csr=spR)

    if SS is None:
        SS = SuffStatBag(K=K, D=D)
    # Expected outer-product for each state k
    SS.setField('xxT', S_xxT, dims=('K', 'D', 'D'))
    # Expected count for each k
    #  Usually computed by allocmodel. But sometimes not (eg TopicModel)
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
    for k in range(K):
        L[:, k] = - 0.5 * Dslice.dim * LOGTWOPI \
            + 0.5 * kwargs['E_logdetL'][k]  \
            - 0.5 * _mahalDist_Post(Dslice.X, k, **kwargs)
    return L


def _mahalDist_Post(X, k, cholB=None, nu=None, **kwargs):
    ''' Calc expected mahalonobis distance from comp k to each data atom

        Returns
        --------
        distvec : 1D array, size nObs
               distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k
    '''
    #Q = np.linalg.solve(cholB[k], X.T)
    Q = scipy.linalg.solve_triangular(cholB[k], X.T, lower=True)
    Q *= Q
    return nu[k] * np.sum(Q, axis=0)
