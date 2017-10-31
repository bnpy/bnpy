from builtins import *
import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray

from .AbstractObsModel import AbstractObsModel
from .GaussObsModel import createECovMatFromUserInput


class AutoRegGaussObsModel(AbstractObsModel):

    ''' First-order auto-regressive data generation model.

    Attributes for Prior (Matrix-Normal-Wishart)
    --------
    nu : float
        degrees of freedom
    B : 2D array, size D x D
        scale matrix that sets mean of parameter Sigma
    M : 2D array, size D x D
        sets mean of parameter A
    V : 2D array, size D x D
        scale matrix that sets covariance of parameter A

    Attributes for k-th component of EstParams (EM point estimates)
    ---------
    A[k] : 2D array, size D x D
        coefficient matrix for auto-regression.
    Sigma[k] : 2D array, size D x D
        covariance matrix.

    Attributes for k-th component of Post (VB parameter)
    ---------
    nu[k] : float
    B[k] : 2D array, size D x D
    M[k] : 2D array, size D x D
    V[k] : 2D array, size D x D
    '''

    def __init__(self, inferType='EM', D=None, E=None,
                 min_covar=None,
                 Data=None,
                 **PriorArgs):
        ''' Initialize bare obsmodel with valid prior hyperparameters.

        Resulting object lacks either EstParams or Post,
        which must be created separately (see init_global_params).
        '''
        # Set dimension D
        if Data is not None:
            D = Data.X.shape[1]
        else:
            assert D is not None
            D = int(D)
        self.D = D

        # Set dimension E
        if Data is not None:
            E = Data.Xprev.shape[1]
        else:
            assert E is not None
            E = int(E)
        self.E = E

        self.K = 0
        self.inferType = inferType
        self.min_covar = min_covar
        self.createPrior(Data, D=D, E=E, **PriorArgs)
        self.Cache = dict()

    def createPrior(
            self, Data,
            D=None, E=None,
            nu=0, B=None,
            M=None, V=None,
            ECovMat=None, sF=1.0,
            VMat='eye', sV=1.0, MMat='zero', sM=1.0,
            **kwargs):
        ''' Initialize Prior ParamBag attribute.

        Post Condition
        ------
        Prior expected covariance matrix set to match provided value.
        '''
        if Data is None:
            if D is None:
                raise ValueError("Need to specify dimension D")
            if E is None:
                raise ValueError("Need to specify dimension E")
        if Data is not None:
            if D is None:
                D = Data.X.shape[1]
            else:
                assert D == Data.X.shape[1]
            if E is None:
                E = Data.Xprev.shape[1]
            else:
                assert E == Data.Xprev.shape[1]

        nu = np.maximum(nu, D + 2)
        if B is None:
            if ECovMat is None or isinstance(ECovMat, str):
                ECovMat = createECovMatFromUserInput(D, Data, ECovMat, sF)
            B = ECovMat * (nu - D - 1)
        B = as2D(B)

        if M is None:
            if MMat == 'zero':
                M = np.zeros((D, E))
            elif MMat == 'eye':
                assert D == E
                M = sM * np.eye(D)
            else:
                raise ValueError('Unrecognized MMat: %s' % (MMat))
        else:
            M = as2D(M)

        if V is None:
            if VMat == 'eye':
                V = sV * np.eye(E)
            elif VMat == 'same':
                assert D == E
                V = sV * ECovMat
            else:
                raise ValueError('Unrecognized VMat: %s' % (VMat))
        else:
            V = as2D(V)

        self.Prior = ParamBag(K=0, D=D, E=E)
        self.Prior.setField('nu', nu, dims=None)
        self.Prior.setField('B', B, dims=('D', 'D'))
        self.Prior.setField('V', V, dims=('E', 'E'))
        self.Prior.setField('M', M, dims=('D', 'E'))

    def get_mean_for_comp(self, k=None):
        if hasattr(self, 'EstParams'):
            return np.diag(self.EstParams.A[k])
        elif k is None or k == 'prior':
            return np.diag(self.Prior.M)
        else:
            return np.diag(self.Post.M[k])

    def get_covar_mat_for_comp(self, k=None):
        if hasattr(self, 'EstParams'):
            return self.EstParams.Sigma[k]
        elif k is None or k == 'prior':
            return self._E_CovMat()
        else:
            return self._E_CovMat(k)

    def get_name(self):
        return 'AutoRegGauss'

    def get_info_string(self):
        return 'Auto-Regressive Gaussian with full covariance.'

    def get_info_string_prior(self):
        msg = 'MatrixNormal-Wishart on each mean/prec matrix pair: A, Lam\n'
        if self.D > 2:
            sfx = ' ...'
        else:
            sfx = ''
        M = self.Prior.M[:2, :2]
        S = self._E_CovMat()[:2, :2]
        msg += 'E[ A ] = \n'
        msg += str(M) + sfx + '\n'
        msg += 'E[ Sigma ] = \n'
        msg += str(S) + sfx
        msg = msg.replace('\n', '\n  ')
        return msg

    def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                     A=None, Sigma=None,
                     **kwargs):
        ''' Initialize EstParams attribute with fields A, Sigma.
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
            A = as3D(A)
            Sigma = as3D(Sigma)
            self.EstParams = ParamBag(K=A.shape[0], D=A.shape[1])
            self.EstParams.setField('A', A, dims=('K', 'D', 'D'))
            self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))

    def setEstParamsFromPost(self, Post):
        ''' Convert from Post to EstParams.
        '''
        D = Post.D
        self.EstParams = ParamBag(K=Post.K, D=D)
        A = Post.M.copy()
        Sigma = Post.B / (Post.nu - D - 1)[:, np.newaxis, np.newaxis]
        self.EstParams.setField('A', A, dims=('K', 'D', 'D'))
        self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))
        self.K = self.EstParams.K

    def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                       nu=0, B=0, M=0, V=0,
                       **kwargs):
        ''' Set Post attribute to provided values.
        '''
        self.ClearCache()
        if obsModel is not None:
            if hasattr(obsModel, 'Post'):
                self.Post = obsModel.Post.copy()
            else:
                self.setPostFromEstParams(obsModel.EstParams)
            self.K = self.Post.K
            return

        if LP is not None and Data is not None:
            SS = self.calcSummaryStats(Data, None, LP)

        if SS is not None:
            self.updatePost(SS)
        else:
            M = as3D(M)
            B = as3D(B)
            V = as3D(V)

            K, D, E = M.shape
            assert D == self.D
            assert E == self.E
            self.Post = ParamBag(K=K, D=self.D, E=self.E)
            self.Post.setField('nu', as1D(nu), dims=('K'))
            self.Post.setField('B', B, dims=('K', 'D', 'D'))
            self.Post.setField('M', M, dims=('K', 'D', 'E'))
            self.Post.setField('V', V, dims=('K', 'E', 'E'))
        self.K = self.Post.K

    def setPostFromEstParams(self, EstParams, Data=None, N=None):
        ''' Set Post attribute values based on provided EstParams.
        '''
        K = EstParams.K
        D = EstParams.D
        if Data is not None:
            N = Data.nObsTotal
        N = np.asarray(N, dtype=np.float)
        if N.ndim == 0:
            N = N / K * np.ones(K)

        nu = self.Prior.nu + N
        B = EstParams.Sigma * (nu - D - 1)[:, np.newaxis, np.newaxis]
        M = EstParams.A.copy()
        V = as3D(self.Prior.V)

        self.Post = ParamBag(K=K, D=D)
        self.Post.setField('nu', nu, dims=('K'))
        self.Post.setField('B', B, dims=('K', 'D', 'D'))
        self.Post.setField('M', M, dims=('K', 'D', 'D'))
        self.Post.setField('V', V, dims=('K', 'D', 'D'))
        self.K = self.Post.K

    def calcSummaryStats(self, Data, SS, LP, **kwargs):
        """ Fill in relevant sufficient stats fields into provided SS.

        Returns
        -------
        SS : bnpy.suffstats.SuffStatBag
        """
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

        Post Condition
        -------
        Field N is guaranteed to be positive.
        '''
        np.maximum(SS.N, 0, out=SS.N)

    def incrementSS(self, SS, k, x):
        pass

    def decrementSS(self, SS, k, x):
        pass

    def calcSummaryStatsForContigBlock(self, Data, SS=None, a=0, b=0):
        ''' Calculate sufficient stats for a single contiguous block of data
        '''
        D = Data.X.shape[1]
        E = Data.Xprev.shape[1]

        if SS is None:
            SS = SuffStatBag(K=1, D=D, E=E)
        elif not hasattr(SS, 'E'):
            SS._Fields.E = E

        ppT = dotATA(Data.Xprev[a:b])[np.newaxis, :, :]
        xxT = dotATA(Data.X[a:b])[np.newaxis, :, :]
        pxT = dotATB(Data.Xprev[a:b], Data.X[a:b])[np.newaxis, :, :]

        SS.setField('N', (b - a) * np.ones(1), dims='K')
        SS.setField('xxT', xxT, dims=('K', 'D', 'D'))
        SS.setField('ppT', ppT, dims=('K', 'E', 'E'))
        SS.setField('pxT', pxT, dims=('K', 'E', 'D'))
        return SS

    def calcLogSoftEvMatrix_FromEstParams(self, Data, **kwargs):
        ''' Compute log soft evidence matrix for Dataset under EstParams.

        Returns
        ---------
        L : 2D array, size N x K
            L[n,k] = log p( data n | EstParams for comp k )
        '''
        K = self.EstParams.K
        L = np.empty((Data.nObs, K))
        for k in range(K):
            L[:, k] = - 0.5 * self.D * LOGTWOPI \
                - 0.5 * self._logdetSigma(k)  \
                - 0.5 * self._mahalDist_EstParam(Data.X, Data.Xprev, k)
        return L

    def _mahalDist_EstParam(self, X, Xprev, k):
        ''' Calc Mahalanobis distance from comp k to every row of X.

        Args
        -----
        X : 2D array, size N x D
        k : integer ID of comp

        Returns
        ------
        dist : 1D array, size N
        '''
        deltaX = X - np.dot(Xprev, self.EstParams.A[k].T)
        Q = np.linalg.solve(self.GetCached('cholSigma', k),
                            deltaX.T)
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
        minCovMat = self.min_covar * np.eye(SS.D)
        A = np.zeros((SS.K, self.D, self.D))
        Sigma = np.zeros((SS.K, self.D, self.D))
        for k in range(SS.K):
            # Add small pos multiple of identity to make invertible
            # TODO: This is source of potential stability issues.
            A[k] = np.linalg.solve(SS.ppT[k] + minCovMat,
                                   SS.pxT[k]).T
            Sigma[k] = SS.xxT[k] \
                - 2 * np.dot(SS.pxT[k].T, A[k].T) \
                + np.dot(A[k], np.dot(SS.ppT[k], A[k].T))
            Sigma[k] /= SS.N[k]
            # Sigma[k] = 0.5 * (Sigma[k] + Sigma[k].T) # symmetry!
            Sigma[k] += minCovMat
        self.EstParams.setField('A', A, dims=('K', 'D', 'D'))
        self.EstParams.setField('Sigma', Sigma, dims=('K', 'D', 'D'))
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
        raise NotImplemented('TODO')

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Update uses the variational objective.

        Post Condition
        ---------
        Attributes K and Post updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D, E=SS.E)
        elif not hasattr(self.Post, 'E'):
            self.Post.E = SS.E

        nu, B, M, V = self.calcPostParams(SS)
        self.Post.setField('nu', nu, dims=('K'))
        self.Post.setField('B', B, dims=('K', 'D', 'D'))
        self.Post.setField('M', M, dims=('K', 'D', 'E'))
        self.Post.setField('V', V, dims=('K', 'E', 'E'))
        self.K = SS.K

    def calcPostParams(self, SS):
        ''' Calc updated posterior params for all comps given suff stats

        These params define the common-form of the exponential family
        Normal-Wishart posterior distribution over mu, diag(Lambda)

        Returns
        --------
        nu : 1D array, size K
        B : 3D array, size K x D x D
            each B[k] symmetric and positive definite
        M : 3D array, size K x D x E
        V : 3D array, size K x E x E
        '''
        Prior = self.Prior
        nu = Prior.nu + SS.N

        B_MVM = Prior.B + np.dot(Prior.M, np.dot(Prior.V, Prior.M.T))
        B = SS.xxT + B_MVM[np.newaxis, :]
        V = SS.ppT + Prior.V[np.newaxis, :]
        M = np.zeros((SS.K, SS.D, SS.E))
        for k in range(B.shape[0]):
            M[k] = np.linalg.solve(
                V[k], SS.pxT[k] + np.dot(Prior.V, Prior.M.T)).T
            B[k] -= np.dot(M[k], np.dot(V[k], M[k].T))
        return nu, B, M, V

    def calcPostParamsForComp(self, SS, kA=None, kB=None):
        ''' Calc params (nu, B, m, kappa) for specific comp, given suff stats

        These params define the common-form of the exponential family
        Normal-Wishart posterior distribution over mu[k], diag(Lambda)[k]

        Returns
        --------
        nu : positive scalar
        B : 2D array, size D x D, symmetric and positive definite
        M : 2D array, size D x D
        V : 2D array, size D x D
        '''
        if kA is not None and kB is not None:
            N = SS.N[kA] + SS.N[kB]
            xxT = SS.xxT[kA] + SS.xxT[kB]
            ppT = SS.ppT[kA] + SS.ppT[kB]
            pxT = SS.pxT[kA] + SS.pxT[kB]
        elif kA is not None:
            N = SS.N[kA]
            xxT = SS.xxT[kA]
            ppT = SS.ppT[kA]
            pxT = SS.pxT[kA]
        else:
            raise ValueError('Need to specify specific component.')
        Prior = self.Prior
        nu = Prior.nu + N
        B_MVM = Prior.B + np.dot(Prior.M, np.dot(Prior.V, Prior.M.T))
        B = xxT + B_MVM
        V = ppT + Prior.V
        M = np.linalg.solve(V, pxT + np.dot(Prior.V, Prior.M.T)).T
        B -= np.dot(M, np.dot(V, M.T))
        return nu, B, M, V

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
        n_nu, n_V, n_VMT, n_B = self.calcNaturalPostParams(SS)
        Post = self.Post
        Post.nu[:] = (1 - rho) * Post.nu + rho * n_nu
        Post.V[:] = (1 - rho) * Post.V + rho * n_V

        Post.n_VMT[:] = (1 - rho) * Post.n_VMT + rho * n_VMT
        Post.n_B[:] = (1 - rho) * Post.n_B + rho * n_B
        self.convertPostToCommon()

    def calcNaturalPostParams(self, SS):
        ''' Calc updated natural params for all comps given suff stats

        These params define the natural-form of the exponential family
        Normal-Wishart posterior distribution over mu, Lambda

        Returns
        --------
        nu : 1D array, size K
        Bnat : 3D array, size K x D x D
        '''
        Prior = self.Prior
        VMT = np.dot(Prior.V, Prior.M.T)
        MVMT = np.dot(Prior.M, VMT)

        n_nu = Prior.nu + SS.N
        n_V = Prior.V + SS.ppT
        n_VMT = VMT + SS.pxT
        n_B = Prior.B + MVMT + SS.xxT
        return n_nu, n_V, n_VMT, n_B

    def convertPostToNatural(self):
        ''' Convert current posterior params from common to natural form

        Post Condition
        --------
        Attribute Post has new fields n_VMT, n_B.
        '''
        Post = self.Post
        # These two are done implicitly
        # Post.setField('nu', Post.nu, dims=None)
        # Post.setField('V', Post.nu, dims=None)
        VMT = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            VMT[k] = np.dot(Post.V[k], Post.M[k].T)
        Post.setField('n_VMT', VMT, dims=('K', 'D', 'D'))

        Bnat = np.empty((self.K, self.D, self.D))
        for k in range(self.K):
            Bnat[k] = Post.B[k] + np.dot(Post.M[k], VMT[k])
        Post.setField('n_B', Bnat, dims=('K', 'D', 'D'))

    def convertPostToCommon(self):
        ''' Convert current posterior params from natural to common form

        Post Condition
        --------
        Attribute Post has new fields n_VMT, n_B.
        '''
        Post = self.Post
        # These two are done implicitly
        # Post.setField('nu', Post.nu, dims=None)
        # Post.setField('V', Post.nu, dims=None)

        M = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            M[k] = np.linalg.solve(Post.V[k], Post.n_VMT[k]).T
        Post.setField('M', M, dims=('K', 'D', 'D'))

        B = np.empty((self.K, self.D, self.D))
        for k in range(self.K):
            B[k] = Post.n_B[k] - np.dot(Post.M[k], Post.n_VMT[k])
        Post.setField('B', B, dims=('K', 'D', 'D'))

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
                - 0.5 * self._mahalDist_Post(Data.X, Data.Xprev, k)
        return L

    def _mahalDist_Post(self, X, Xprev, k):
        ''' Calc expected mahalonobis distance from comp k to each data atom

        Returns
        --------
        distvec : 1D array, size nObs
               distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k
        '''
        # Calc: (x-M*xprev)' * B * (x-M*xprev)
        deltaX = X - np.dot(Xprev, self.Post.M[k].T)
        Q = np.linalg.solve(self.GetCached('cholB', k),
                            deltaX.T)
        Q *= Q

        # Calc: xprev' * V * xprev
        Qprev = np.linalg.solve(self.GetCached('cholV', k),
                                Xprev.T)
        Qprev *= Qprev

        return self.Post.nu[k] * np.sum(Q, axis=0) \
            + self.D * np.sum(Qprev, axis=0)

    def calcELBO_Memoized(self, SS, afterMStep=False, **kwargs):
        ''' Calculate obsModel's objective using suff stats SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag
        afterMStep : boolean flag
            if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        obsELBO : scalar float
            Equal to E[ log p(x) + log p(phi) - log q(phi)]
        '''
        elbo = np.zeros(SS.K)
        Post = self.Post
        Prior = self.Prior
        for k in range(SS.K):
            elbo[k] = c_Diff(Prior.nu,
                             self.GetCached('logdetB'),
                             Prior.M,
                             self.GetCached('logdetV'),
                             Post.nu[k],
                             self.GetCached('logdetB', k),
                             Post.M[k],
                             self.GetCached('logdetV', k),
                             )
            if not afterMStep:
                aDiff = SS.N[k] + Prior.nu - Post.nu[k]
                bDiff = SS.xxT[k] + Prior.B + \
                    np.dot(Prior.M, np.dot(Prior.V, Prior.M.T)) - \
                    Post.B[k] - \
                    np.dot(Post.M[k], np.dot(Post.V[k], Post.M[k].T))
                cDiff = SS.pxT[k] + np.dot(Prior.V, Prior.M.T) - \
                    np.dot(Post.V[k], Post.M[k].T)
                dDiff = SS.ppT[k] + Prior.V - Post.V[k]
                elbo[k] += 0.5 * aDiff * self.GetCached('E_logdetL', k) \
                    - 0.5 * self._trace__E_L(bDiff, k) \
                    + self._trace__E_LA(cDiff, k) \
                    - 0.5 * self._trace__E_ALA(dDiff, k)
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
        cA = c_Func(Post.nu[kA], Post.B[kA], Post.M[kA], Post.V[kA])
        cB = c_Func(Post.nu[kB], Post.B[kB], Post.M[kB], Post.V[kB])

        cPrior = c_Func(Prior.nu, Prior.B, Prior.M, Prior.V)
        nu, B, M, V = self.calcPostParamsForComp(SS, kA, kB)
        cAB = c_Func(nu, B, M, V)
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
        cPrior = c_Func(Prior.nu, Prior.B, Prior.M, Prior.V)
        c = np.zeros(SS.K)
        for k in range(SS.K):
            c[k] = c_Func(Post.nu[k], Post.B[k], Post.M[k], Post.V[k])

        Gap = np.zeros((SS.K, SS.K))
        for j in range(SS.K):
            for k in range(j + 1, SS.K):
                nu, B, M, V = self.calcPostParamsForComp(SS, j, k)
                cjk = c_Func(nu, B, M, V)
                Gap[j, k] = c[j] + c[k] - cPrior - cjk
        return Gap

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
               Gaps[j] = scalar change in ELBO after merge of PairList[j]
        '''
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
        return Gaps

    def calcHardMergeGap_SpecificPairSS(self, SS1, SS2):
        ''' Calc change in ELBO for merger of two components.
        '''
        assert SS1.K == 1
        assert SS2.K == 1

        Prior = self.Prior
        cPrior = c_Func(Prior.nu, Prior.B, Prior.M, Prior.V)

        # Compute cumulants of individual states 1 and 2
        c1 = c_Func(*self.calcPostParamsForComp(SS1, 0))
        c2 = c_Func(*self.calcPostParamsForComp(SS2, 0))

        # Compute cumulant of merged state 1&2
        SS12 = SS1 + SS2
        c12 = c_Func(*self.calcPostParamsForComp(SS12, 0))
        return c1 + c2 - cPrior - c12

    def calcLogMargLikForComp(self, SS, kA, kB=None, **kwargs):
        ''' Calc log marginal likelihood of data assigned to component

        Up to an additive constant that depends on the prior.

        Requires Data pre-summarized into sufficient stats for each comp.
        If multiple comp IDs are provided,
        we combine into a "merged" component.

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
        nu, B, M, V = self.calcPostParamsForComp(SS, kA, kB)
        return -1 * c_Func(nu, B, M, V)

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
        for k in range(SS.K):
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
        raise NotImplementedError('TODO')

    def _calcPredProbVec_Fast(self, SS, x):
        raise NotImplementedError('TODO')

    def _Verify_calcPredProbVec(self, SS, x):
        raise NotImplementedError('TODO')

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
            retMat = np.zeros((self.K, self.D, self.D))
            for k in range(self.K):
                retMat[k] = scipy.linalg.cholesky(self.Post.B[k], lower=True)
            return retMat
        elif k is None:
            B = self.Prior.B
        else:
            B = self.Post.B[k]
        return scipy.linalg.cholesky(B, lower=True)

    def _logdetB(self, k=None):
        cholB = self.GetCached('cholB', k)
        return 2 * np.sum(np.log(np.diag(cholB)))

    def _cholV(self, k=None):
        if k == 'all':
            retMat = np.zeros((self.K, self.D, self.D))
            for k in range(self.K):
                retMat[k] = scipy.linalg.cholesky(self.Post.V[k], lower=True)
            return retMat
        elif k is None:
            V = self.Prior.V
        else:
            V = self.Post.V[k]
        return scipy.linalg.cholesky(V, lower=True)

    def _logdetV(self, k=None):
        cholV = self.GetCached('cholV', k)
        return 2 * np.sum(np.log(np.diag(cholV)))

    def _E_logdetL(self, k=None):
        dvec = np.arange(1, self.D + 1, dtype=np.float)
        if k is 'all':
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

    def _E_LA(self, k=None):
        if k is None:
            nu = self.Prior.nu
            B = self.Prior.B
            M = self.Prior.M
        else:
            nu = self.Post.nu[k]
            B = self.Post.B[k]
            M = self.Post.M[k]
        return nu * np.linalg.solve(B, M)

    def _E_ALA(self, k=None):
        if k is None:
            nu = self.Prior.nu
            M = self.Prior.M
            B = self.Prior.B
            V = self.Prior.V
        else:
            nu = self.Post.nu[k]
            M = self.Post.M[k]
            B = self.Post.B[k]
            V = self.Post.V[k]
        Q = np.linalg.solve(self.GetCached('cholB', k), M)
        return self.D * np.linalg.inv(V) \
            + nu * np.dot(Q.T, Q)

    def _trace__E_L(self, S, k=None):
        if k is None:
            nu = self.Prior.nu
            B = self.Prior.B
        else:
            nu = self.Post.nu[k]
            B = self.Post.B[k]
        return nu * np.trace(np.linalg.solve(B, S))

    def _trace__E_LA(self, S, k=None):
        E_LA = self._E_LA(k)
        return np.trace(np.dot(E_LA, S))

    def _trace__E_ALA(self, S, k=None):
        E_ALA = self._E_ALA(k)
        return np.trace(np.dot(E_ALA, S))

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
            fillSharedMemArray(ShMem['M'], self.Post.M)
            fillSharedMemArray(ShMem['cholV'], self._cholV('all'))
            fillSharedMemArray(ShMem['cholB'], self._cholB('all'))
            fillSharedMemArray(ShMem['E_logdetL'], self._E_logdetL('all'))

        else:
            ShMem['nu'] = numpyToSharedMemArray(self.Post.nu)
            ShMem['M'] = numpyToSharedMemArray(self.Post.M)
            ShMem['cholV'] = numpyToSharedMemArray(self._cholV('all'))
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


def MAPEstParams_inplace(nu, B, m, kappa=0):
    ''' MAP estimate parameters mu, Sigma given Normal-Wishart hyperparameters
    '''
    raise NotImplementedError('TODO')


def c_Func(nu, logdetB, M, logdetV):
    ''' Evaluate cumulant function c at given parameters.

    c is the cumulant of the MatrixNormal-Wishart, using common params.

    Returns
    --------
    c : scalar real value of cumulant function at provided args
    '''
    if logdetB.ndim >= 2:
        logdetB = np.log(np.linalg.det(logdetB))
    if logdetV.ndim >= 2:
        logdetV = np.log(np.linalg.det(logdetV))
    D, E = M.shape
    dvec = np.arange(1, D + 1, dtype=np.float)
    return - 0.25 * D * (D - 1) * LOGPI \
        - 0.5 * D * LOGTWO * nu \
        - np.sum(gammaln(0.5 * (nu + 1 - dvec))) \
        + 0.5 * nu * logdetB \
        - 0.5 * D * E * LOGTWOPI \
        + 0.5 * E * logdetV


def c_Diff(nu, logdetB, M, logdetV,
           nu2, logdetB2, M2, logdetV2):
    return c_Func(nu, logdetB, M, logdetV) \
        - c_Func(nu2, logdetB2, M2, logdetV2)


def calcSummaryStats(Data, SS, LP,
                     **kwargs):
    ''' Calculate sufficient statistics for local params at data slice.

    Returns
    -------
    SS
    '''
    X = Data.X
    Xprev = Data.Xprev
    resp = LP['resp']
    K = resp.shape[1]
    D = Data.X.shape[1]
    E = Data.Xprev.shape[1]

    if SS is None:
        SS = SuffStatBag(K=K, D=D, E=E)
    elif not hasattr(SS, 'E'):
        SS._Fields.E = E

    # Expected count for each k
    #  Usually computed by allocmodel. But just in case...
    if not hasattr(SS, 'N'):
        SS.setField('N', np.sum(resp, axis=0), dims='K')

    # Expected outer products
    sqrtResp = np.sqrt(resp)
    xxT = np.empty((K, D, D))
    ppT = np.empty((K, E, E))
    pxT = np.empty((K, E, D))
    for k in range(K):
        sqrtResp_k = sqrtResp[:, k][:, np.newaxis]
        xxT[k] = dotATA(sqrtResp_k * Data.X)
        ppT[k] = dotATA(sqrtResp_k * Data.Xprev)
        pxT[k] = np.dot(Data.Xprev.T, resp[:, k][:, np.newaxis] * Data.X)
    SS.setField('xxT', xxT, dims=('K', 'D', 'D'))
    SS.setField('ppT', ppT, dims=('K', 'E', 'E'))
    SS.setField('pxT', pxT, dims=('K', 'E', 'D'))
    return SS


def calcLocalParams(Dslice, **kwargs):
    ''' Compute local parameters for provided data slice.

    Returns
    -------
    LP : dict of local params
    '''
    L = calcLogSoftEvMatrix_FromPost(Dslice, **kwargs)
    LP = dict(E_log_soft_ev=L)
    return LP


def calcLogSoftEvMatrix_FromPost(Dslice,
                                 E_logdetL=None,
                                 **kwargs):
    ''' Calculate expected log soft ev matrix for variational.

    Returns
    ------
    L : 2D array, size N x K
    '''
    K = kwargs['K']
    L = np.zeros((Dslice.nObs, K))
    for k in range(K):
        L[:, k] = - 0.5 * Dslice.dim * LOGTWOPI \
            + 0.5 * E_logdetL[k]  \
            - 0.5 * _mahalDist_Post(Dslice.X, Dslice.Xprev, k, **kwargs)
    return L


def _mahalDist_Post(X, Xprev, k, D=None,
                    cholB=None, cholV=None, M=None,
                    nu=None, **kwargs):
    ''' Calc expected mahalonobis distance from comp k to each data atom

    Returns
    --------
    distvec : 1D array, size N
           distvec[n] gives E[ (x-\mu) \Lam (x-\mu) ] for comp k
    '''
    deltaX = X - np.dot(Xprev, M[k].T)
    Q = np.linalg.solve(cholB[k], deltaX.T)
    Q *= Q

    # Calc: xprev' * V * xprev
    Qprev = np.linalg.solve(cholV[k], Xprev.T)
    Qprev *= Qprev
    return nu[k] * np.sum(Q, axis=0) + D * np.sum(Qprev, axis=0)
